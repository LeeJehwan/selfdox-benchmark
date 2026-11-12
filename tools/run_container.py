"""Run a team's agent on a split, then validate and score its output (Phase 3).

    uv run python -m tools.run_container --image team7/agent@sha256:... --split hidden \
        [--runtime docker|apptainer|venv] [--models /path/to/weights] [--gpus 1]
        [--timeout 5400] [--repeat k]

    # the lab cluster has no container runtime: run the team's agent folder in a venv
    uv run python -m tools.run_container --runtime venv --image submissions/team7 \
        --split test [--python /path/to/shared/venv/bin/python] [--isolation auto]

The container gets exactly what the rules allow (docs/participants.md):
  /input   the split's images, read-only
  /models  the allowed model weights, read-only (optional)
  /output  an empty folder for predictions.json and images/
  no network, a GPU limit, a wall-clock limit (the container is killed when it runs out)
Docker images run with `docker run --network none`; an Apptainer image (.sif, e.g. built
from the team's docker-archive) runs with `--containall --net --network none`.

--runtime venv runs an agent folder (the agent_template/ layout) without a container,
for the lab cluster, which has neither Apptainer nor docker access:
  - the command is `python run.py --input {input} --output {output}`, or the list in the
    folder's selfdox_run.json ({"command": [...]}, placeholders {input}, {output},
    {models}); "python" is replaced by the venv's interpreter
  - the interpreter is a fresh venv in runs/<name>/venv with the folder's
    requirements.txt installed (setup.log), or --python to reuse an existing one
  - the inputs are copied into the run folder, so the originals can't be changed
  - the network is cut with a user + network namespace (`unshare -rn`) where the node
    allows it (--isolation netns). The lab cluster does not (Ubuntu 24.04 AppArmor blocks
    unprivileged user namespaces), so --isolation auto falls back to `python`: a
    sitecustomize.py on PYTHONPATH that refuses every non-loopback connection and DNS
    lookup in the agent and in any Python process it starts (a non-Python program could
    still get out, so we also read the code). --isolation env only sets offline
    variables (HF_HUB_OFFLINE, a dead proxy). score.json records which one ran
  - the whole process group is killed at the time limit
  - the folder's sha256 (every file but venvs and caches) stands in for the image digest
Afterwards the output is validated and scored against the split's GT, and
runs/<name>/<split>/ keeps the command, the log, the validation report and the score.

Only a missing or malformed predictions.json fails the run (every image then scores 0).
A problem with one prediction (GPS left, image missing, a box outside the photo, wrong
size...) is a warning: that image scores 0 and the rest are scored as usual. score.json
also records the image digest (what exactly ran), the GPU name and the peak GPU memory.
With --repeat k the container runs k times (run1/ ... runk/) and score.json adds the
mean and spread over runs and pass^k (selfdox.metrics.multi_run). Official runs are single
runs (docs/participants.md §7).

Splits: dev (data/dev), test (data/test images, data/private/test_gt.json) and hidden
(data/private/hidden/images, data/private/hidden_gt.json).
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

SPLITS = {"dev": ("data/dev/images", "data/dev/gt.json"),
          "test": ("data/test/images", "data/private/test_gt.json"),
          "hidden": ("data/private/hidden/images", "data/private/hidden_gt.json")}


def docker_cmd(image, inputs, out, models=None, gpus=1, name="selfdox-run", cpus=8,
               memory="32g"):
    cmd = ["docker", "run", "--rm", "--name", name, "--network", "none",
           "--cpus", str(cpus), "--memory", memory, "--pids-limit", "1024",
           "--security-opt", "no-new-privileges",
           "-v", f"{Path(inputs).resolve()}:/input:ro", "-v", f"{Path(out).resolve()}:/output"]
    if models:
        cmd += ["-v", f"{Path(models).resolve()}:/models:ro"]
    if gpus:
        cmd += ["--gpus", str(gpus)]
    if hasattr(os, "getuid"):
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
    return cmd + [image]


def apptainer_cmd(image, inputs, out, models=None, gpus=1):
    binds = [f"{Path(inputs).resolve()}:/input:ro", f"{Path(out).resolve()}:/output"]
    if models:
        binds.append(f"{Path(models).resolve()}:/models:ro")
    cmd = ["apptainer", "run", "--containall", "--cleanenv", "--net", "--network", "none",
           "--bind", ",".join(binds)]
    if gpus:
        cmd.append("--nv")
    return cmd + [image]


DEFAULT_COMMAND = ["python", "run.py", "--input", "{input}", "--output", "{output}"]
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache"}


def agent_command(agent_dir):
    """The agent folder's command template (selfdox_run.json, else DEFAULT_COMMAND)."""
    spec = Path(agent_dir) / "selfdox_run.json"
    if spec.is_file():
        cmd = json.loads(spec.read_text()).get("command")
        if not (isinstance(cmd, list) and cmd and all(isinstance(c, str) for c in cmd)):
            raise SystemExit(f"{spec}: 'command' must be a non-empty list of strings")
        return cmd
    return list(DEFAULT_COMMAND)


def netns_available():
    """Can this user cut the network with a user + network namespace?"""
    if not shutil.which("unshare"):
        return False
    try:
        return subprocess.run(["unshare", "-rn", "true"], capture_output=True,
                              timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


# Loaded by every Python process of a venv run (--isolation python). Loopback and Unix
# sockets stay open: vLLM and other servers talk to their own workers over them.
SITECUSTOMIZE = '''"""SelfDox runner: no network for this agent (loopback and Unix sockets only)."""
import ipaddress as _ip
import socket as _socket

_MSG = "network disabled by the SelfDox runner"


def _local(host):
    if host in (None, "", "localhost"):
        return True
    try:
        return _ip.ip_address(str(host).split("%")[0]).is_loopback
    except ValueError:
        return False


def _check(sock, addr):
    if sock.family in (_socket.AF_INET, _socket.AF_INET6) and isinstance(addr, tuple) \\
            and not _local(addr[0]):
        raise OSError(101, _MSG)


_connect, _connect_ex, _sendto = (_socket.socket.connect, _socket.socket.connect_ex,
                                  _socket.socket.sendto)


def connect(self, addr):
    _check(self, addr)
    return _connect(self, addr)


def connect_ex(self, addr):
    _check(self, addr)
    return _connect_ex(self, addr)


def sendto(self, data, *args):
    _check(self, args[-1])
    return _sendto(self, data, *args)


_getaddrinfo = _socket.getaddrinfo


def getaddrinfo(host, *args, **kwargs):
    if not _local(host if not isinstance(host, bytes) else host.decode()):
        raise _socket.gaierror(-2, _MSG)
    return _getaddrinfo(host, *args, **kwargs)


_socket.socket.connect, _socket.socket.connect_ex = connect, connect_ex
_socket.socket.sendto, _socket.getaddrinfo = sendto, getaddrinfo
'''


def python_block_dir(root):
    """A folder with the network-blocking sitecustomize.py, for PYTHONPATH."""
    d = Path(root) / "_selfdox_site"
    d.mkdir(parents=True, exist_ok=True)
    (d / "sitecustomize.py").write_text(SITECUSTOMIZE)
    return d


def venv_cmd(agent_dir, inputs, out, models=None, python=sys.executable, isolation="netns"):
    """The agent folder's command with the paths filled in, run by `python`, inside a
    network-less namespace when isolation == "netns"."""
    fill = {"input": str(Path(inputs).resolve()), "output": str(Path(out).resolve()),
            "models": str(Path(models).resolve()) if models else ""}
    cmd = [c.format(**fill) for c in agent_command(agent_dir)]
    if cmd[0] in ("python", "python3"):
        # the agent runs from its own folder, so a relative interpreter path would break
        cmd[0] = str(Path(python).absolute()) if os.sep in str(python) else str(python)
    return (["unshare", "-rn"] + cmd) if isolation == "netns" else cmd


def offline_env(gpus=1, base=None):
    """Environment for a venv run: no proxy to the outside, Hugging Face offline, and the
    first `gpus` GPUs (unless the scheduler already set CUDA_VISIBLE_DEVICES)."""
    base = dict(os.environ if base is None else base)
    env = {k: base[k] for k in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "USER",
                                "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH") if k in base}
    dead = "http://127.0.0.1:9"
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "http_proxy": dead,
                "https_proxy": dead, "HTTP_PROXY": dead, "HTTPS_PROXY": dead,
                "PYTHONUNBUFFERED": "1"})
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(max(gpus, 0))) or ""
    return env


def prepare_venv(agent_dir, venv_dir, log_path, base_python=sys.executable):
    """A fresh venv with the agent folder's requirements.txt; returns its python."""
    venv_dir = Path(venv_dir).absolute()
    py = venv_dir / "bin" / "python"
    env = {**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    with open(log_path, "w") as log:
        if not py.exists():
            subprocess.run([str(base_python), "-m", "venv", str(venv_dir)], check=True,
                           stdout=log, stderr=subprocess.STDOUT)
        req = Path(agent_dir) / "requirements.txt"
        if req.is_file():
            subprocess.run([str(py), "-m", "pip", "install", "-r", str(req)], check=True,
                           stdout=log, stderr=subprocess.STDOUT, env=env)
    return py


def tree_digest(folder):
    """sha256 over the folder's files (relative path and bytes), venvs and caches left out."""
    h = hashlib.sha256()
    root = Path(folder)
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_dir() or SKIP_DIRS & set(rel.parts):
            continue
        h.update(str(rel).encode() + b"\0")
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        h.update(b"\0")
    return {"sha256_tree": h.hexdigest()}


def run_name(image):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(image).name if image.endswith(".sif")
                  else image).strip("_")[:80]


class GpuMonitor:
    """Peak GPU memory (MiB, summed over GPUs) while the container runs, by polling
    nvidia-smi. Does nothing where nvidia-smi is missing."""

    def __init__(self, every=2.0):
        self.every, self.peak_mib, self._stop = every, None, threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _sample(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                  "--format=csv,noheader,nounits"], capture_output=True,
                                 text=True, timeout=10).stdout
            used = sum(int(x) for x in out.split())
        except (OSError, subprocess.SubprocessError, ValueError):
            return
        self.peak_mib = max(self.peak_mib or 0, used)

    def _loop(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.every)

    def __enter__(self):
        if shutil.which("nvidia-smi"):
            self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t.is_alive():
            self._t.join(timeout=15)


def gpu_names():
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def image_digest(image, runtime):
    """What exactly ran: the Docker image id and repo digests, or the .sif file's sha256."""
    if runtime == "apptainer" or image.endswith(".sif"):
        p = Path(image)
        if not p.is_file():
            return None
        h = hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return {"sha256": h.hexdigest()}
    if not shutil.which("docker"):
        return None
    try:
        out = subprocess.run(["docker", "image", "inspect", "--format",
                              "{{json .Id}}|{{json .RepoDigests}}", image],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or "|" not in out.stdout:
        return None
    iid, repo = out.stdout.strip().split("|", 1)
    return {"id": json.loads(iid), "repo_digests": json.loads(repo) or []}


def run_container(cmd, log_path, timeout, kill_cmd=None, env=None, cwd=None):
    """Run with a wall-clock limit. Returns (exit code or None on timeout, seconds). The
    command gets its own process group, and on timeout the whole group is killed (a venv
    agent's worker processes too)."""
    t = time.time()
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=cwd,
                             start_new_session=True)
        try:
            code = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if kill_cmd:
                subprocess.run(kill_cmd, capture_output=True)
            try:
                os.killpg(p.pid, 9)
            except (ProcessLookupError, PermissionError):
                p.kill()
            p.wait()
            code = None
    return code, time.time() - t


def failed_run(gt):
    """Every image at 0, for a run without a usable predictions.json."""
    from selfdox.metrics import ImageScore, image_outcome, photo_group
    out = []
    for im in gt["images"]:
        r = ImageScore(id=im["id"], category=im.get("category", ""), score=0.0, parts={},
                       hard_fail="run_failed", expected_verdict=im.get("expected_verdict"),
                       group=photo_group(im))
        r.outcome = image_outcome(r)
        out.append(r)
    return out


def check_and_score(out, gt_path):
    """(run errors, warnings, score summary, per-image results). Only a missing or
    malformed predictions.json is a run error (every image scores 0); a problem with one
    prediction is a warning and only that image scores 0."""
    from selfdox.metrics import aggregate
    from selfdox.schema import gt_sizes, load_gt, submission_problems
    from selfdox.score import score_submission
    gt = load_gt(gt_path)
    errors, problems, warnings = submission_problems(out, gt_sizes(gt))
    if errors:
        results = failed_run(gt)
        return errors, warnings, aggregate(results), results
    from selfdox.score import invalid_predictions
    zero, _ = invalid_predictions(out, gt)
    for iid, msgs in problems.items():
        warnings += [m + (" -> image scored 0" if iid in zero else "") for m in msgs]
    _, _, agg, results = score_submission(gt_path, out, zero=zero)
    return [], warnings, agg, results


def run_once(args, runtime, inputs, gt, out_root, name, digest=None):
    """One container run into out_root/output, then validation and scoring."""
    out = out_root / "output"
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; remove it or pass another --name")
    env = cwd = None
    isolation = "container"
    if runtime == "docker":
        cname = f"selfdox-{name}-{args.split}-{out_root.name}"[:60]
        cmd = docker_cmd(args.image, inputs, out, args.models, args.gpus, name=cname)
        kill = ["docker", "kill", cname]
    elif runtime == "apptainer":
        cmd = apptainer_cmd(args.image, inputs, out, args.models, args.gpus)
        kill = None
    else:
        isolation = venv_isolation(args.isolation)
        local_in = out_root / "input"
        if not args.dry_run:
            shutil.rmtree(local_in, ignore_errors=True)
            shutil.copytree(inputs, local_in)
        cmd = venv_cmd(args.image, local_in, out, args.models, venv_python(args),
                       isolation=isolation)
        env, cwd, kill = offline_env(args.gpus), str(Path(args.image).resolve()), None
        if isolation == "python":
            env["PYTHONPATH"] = str(python_block_dir(out_root).resolve())
    (out_root / "command.txt").write_text(shlex.join(cmd) + "\n")
    print(shlex.join(cmd))
    if args.dry_run:
        return None
    with GpuMonitor() as gpu:
        code, secs = run_container(cmd, out_root / "container.log", args.timeout, kill,
                                   env=env, cwd=cwd)
    status = "timeout" if code is None else ("ok" if code == 0 else f"exit {code}")
    print(f"container: {status} after {secs:.0f}s (log: {out_root / 'container.log'})")
    errors, warnings, agg, results = check_and_score(out, gt)
    (out_root / "validate.txt").write_text(
        "\n".join([f"WARNING {w}" for w in warnings] + [f"ERROR   {e}" for e in errors]) + "\n")
    report = {"image": args.image, "image_digest": digest, "runtime": runtime,
              "isolation": isolation, "split": args.split, "status": status,
              "seconds": round(secs),
              "gpu": gpu_names(), "peak_gpu_mib": gpu.peak_mib,
              "errors": errors, "warnings": warnings, "summary": agg,
              "images": [asdict(r) for r in results]}
    (out_root / "score.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    if errors:
        print(f"run failed ({len(errors)} error(s), every image scored 0); see "
              f"{out_root / 'validate.txt'}")
    else:
        print(f"score {agg['score']:.2f}  pass rate {agg['pass_rate']:.3f}  false-safe rate "
              f"{agg['false_safe_rate']:.3f}  ({len(warnings)} warning(s)) -> "
              f"{out_root / 'score.json'}")
    return report, results


def venv_isolation(choice):
    """netns, python or env for a venv run (auto: netns where the node allows it, else the
    Python-level block)."""
    if choice in ("env", "python"):
        return choice
    ok = netns_available()
    if choice == "netns" and not ok:
        raise SystemExit("--isolation netns: `unshare -rn` does not work on this node")
    return "netns" if ok else "python"


def venv_python(args):
    if args.python:
        return args.python
    if args.dry_run:
        return "python"
    root = Path(args.runs) / (args.name or run_name(args.image))
    root.mkdir(parents=True, exist_ok=True)
    return prepare_venv(args.image, root / "venv", root / "setup.log")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run a team's agent on a SelfDox split.")
    ap.add_argument("--image", required=True, help="Docker image (name@sha256:... preferred), "
                    "a .sif for Apptainer, or an agent folder for --runtime venv")
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--runtime", choices=["docker", "apptainer", "venv"])
    ap.add_argument("--python", help="venv runtime: use this interpreter instead of a fresh "
                    "venv with the folder's requirements.txt")
    ap.add_argument("--isolation", choices=["auto", "netns", "python", "env"], default="auto",
                    help="venv runtime: how to cut the network (default: netns if the node "
                         "allows it, else the Python-level block)")
    ap.add_argument("--models", help="folder with the allowed model weights (mounted at /models)")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=5400, help="seconds per run (default 90 min)")
    ap.add_argument("--repeat", type=int, default=1, help="run the container this many times")
    ap.add_argument("--root", default=".", help="benchmark root (holds data/)")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--name", help="run name (default: from the image)")
    ap.add_argument("--dry-run", action="store_true", help="print the command only")
    args = ap.parse_args(argv)
    runtime = args.runtime or ("apptainer" if args.image.endswith(".sif") else
                               "venv" if Path(args.image).is_dir() else "docker")
    inputs, gt = (Path(args.root) / p for p in SPLITS[args.split])
    name = args.name or run_name(args.image)
    split_root = Path(args.runs) / name / args.split
    roots = [split_root] if args.repeat == 1 else \
        [split_root / f"run{i}" for i in range(1, args.repeat + 1)]
    if runtime == "venv":
        args.python = None if args.dry_run and not args.python else venv_python(args)
        digest = tree_digest(args.image)
    else:
        digest = None if args.dry_run else image_digest(args.image, runtime)
    reports = []
    for root in roots:
        got = run_once(args, runtime, inputs, gt, root, name, digest)
        if got is not None:
            reports.append(got)
    if args.dry_run:
        return 0
    if args.repeat > 1:
        from selfdox.metrics import multi_run
        multi = multi_run([r for _, r in reports])
        summary = {"image": args.image, "image_digest": digest, "split": args.split,
                   "multi_run": multi,
                   "runs": [{k: rep[k] for k in ("status", "seconds", "peak_gpu_mib", "errors")}
                            | {"score": rep["summary"]["score"],
                               "pass_rate": rep["summary"]["pass_rate"],
                               "n_warnings": len(rep["warnings"])} for rep, _ in reports]}
        (split_root / "score.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
        print(f"{multi['k']} runs: score {multi['score_mean']:.2f} +- {multi['score_std']:.2f}  "
              f"pass@1 {multi['pass_at_1']:.3f}  pass^{multi['k']} {multi['pass_hat_k']:.3f} "
              f"-> {split_root / 'score.json'}")
    return 1 if any(rep["errors"] for rep, _ in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
