"""Check and pack a Phase 3 submission.

    python -m selfdox.submit --agent my_agent --team team7 --contact you@example.com \
        --model "Qwen/Qwen3-VL-4B-Instruct@ebb281e (bf16)" --tools "EasyOCR 1.7.2" \
        --temperature 0 --seed 0 [--test-sub outputs_test] [--smoke 3] [--out submission.zip]

Checks the agent folder the way the organizers will run it (docs/participants.md §6):
  - it has run.py or a selfdox_run.json command, and a requirements.txt
  - no answer keys inside, no file over 50 MB (weights come from {models}, not the folder)
  - --smoke N: runs the folder's command with the official runner's settings (offline
    environment, Python-level network block, a copy of N dev photos) using this Python,
    then validates the output. It does not install requirements.txt; run it from a venv
    where they are installed.
  - --test-sub: validates your own test predictions against data/test/images; only
    predictions.json goes into the zip (we run your agent on test ourselves).
Writes a zip with the agent folder, metadata.json (team, contact, model build, tools,
decoding, the folder's sha256) and test_predictions.json. Attach it to a "Submission"
issue on the benchmark repository.
"""
import argparse
import datetime
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from . import config as C
from .schema import input_sizes, submission_errors

MAX_FILE_MB = 50
MAX_ZIP_MB = 25          # GitHub issue attachments
SKIP = {".git", "__pycache__", ".venv", "venv", "outputs", ".mypy_cache", ".pytest_cache"}


def agent_files(folder):
    root = Path(folder)
    return [p for p in sorted(root.rglob("*"))
            if p.is_file() and not SKIP & set(p.relative_to(root).parts)]


def check_folder(folder):
    """(errors, warnings) for an agent folder."""
    folder = Path(folder)
    errors, warnings = [], []
    if not folder.is_dir():
        return [f"{folder} is not a folder"], []
    if not (folder / "run.py").is_file() and not (folder / "selfdox_run.json").is_file():
        errors.append("no run.py and no selfdox_run.json: we would not know how to start it")
    if not (folder / "requirements.txt").is_file():
        warnings.append("no requirements.txt: the venv will have only the standard library")
    for p in agent_files(folder):
        rel = p.relative_to(folder)
        if p.name.endswith("gt.json") or p.name in ("test_gt.json", "hidden_gt.json"):
            errors.append(f"{rel}: answer keys must not be in the agent folder")
        if p.stat().st_size > MAX_FILE_MB * 2 ** 20:
            errors.append(f"{rel}: {p.stat().st_size / 2 ** 20:.0f} MB; files over "
                          f"{MAX_FILE_MB} MB are not allowed (weights come from {{models}})")
    return errors, warnings


def smoke(folder, n, dev_images="data/dev/images", models=None):
    """Run the folder's command like the official venv run on n dev photos; returns
    (errors, warnings) of the validated output."""
    from tools.run_container import offline_env, python_block_dir, venv_cmd
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "input").mkdir()
        photos = sorted(Path(dev_images).glob("*.jpg"))[:n]
        for p in photos:
            shutil.copy(p, tmp / "input" / p.name)
        out = tmp / "output"
        out.mkdir()
        cmd = venv_cmd(folder, tmp / "input", out, models, sys.executable, isolation="python")
        env = offline_env(1)
        env["PYTHONPATH"] = str(python_block_dir(tmp))
        r = subprocess.run(cmd, cwd=str(Path(folder).resolve()), env=env, capture_output=True,
                           text=True, timeout=60 * 30)
        if r.returncode != 0:
            return [f"the agent exited with code {r.returncode}:\n{r.stderr[-2000:]}"], []
        return submission_errors(out, input_sizes(tmp / "input"))


def tree_sha256(folder):
    from tools.run_container import tree_digest
    return tree_digest(folder)["sha256_tree"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check and pack a SelfDox submission.")
    ap.add_argument("--agent", required=True, help="agent folder (agent_template layout)")
    ap.add_argument("--team", required=True)
    ap.add_argument("--contact", required=True, help="an email we can reach you at")
    ap.add_argument("--model", required=True, help='model build, e.g. "Qwen/Qwen3-VL-4B-'
                    'Instruct@ebb281e (bf16)", or "none"')
    ap.add_argument("--tools", default="", help="other tools and models, with versions")
    ap.add_argument("--temperature", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--test-sub", help="your own outputs on data/test/images (optional)")
    ap.add_argument("--smoke", type=int, default=0, help="run the agent on N dev photos first")
    ap.add_argument("--models", help="folder to pass as {models} for --smoke")
    ap.add_argument("--out", help="zip to write (default: submission_<team>.zip)")
    args = ap.parse_args(argv)
    errors, warnings = check_folder(args.agent)
    if not errors and args.smoke:
        e, w = smoke(args.agent, args.smoke, models=args.models)
        errors += [f"smoke run: {x}" for x in e]
        warnings += [f"smoke run: {x}" for x in w]
    if args.test_sub:
        e, w = submission_errors(args.test_sub, input_sizes("data/test/images"))
        errors += [f"test predictions: {x}" for x in e]
        warnings += [f"test predictions: {x}" for x in w]
    for w in warnings:
        print(f"WARNING {w}")
    for e in errors:
        print(f"ERROR   {e}")
    if errors:
        print(f"\n{len(errors)} error(s); nothing written.")
        return 1
    meta = {"team": args.team, "contact": args.contact, "model": args.model,
            "tools": args.tools, "temperature": args.temperature, "seed": args.seed,
            "agent_sha256": tree_sha256(args.agent), "scorer_version": C.VERSION,
            "packed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    out = Path(args.out or f"submission_{args.team}.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("metadata.json", json.dumps(meta, ensure_ascii=False, indent=1))
        for p in agent_files(args.agent):
            z.write(p, f"agent/{p.relative_to(args.agent)}")
        if args.test_sub:
            z.write(Path(args.test_sub) / "predictions.json", "test_predictions.json")
    mb = out.stat().st_size / 2 ** 20
    print(f"{out} ({mb:.1f} MB), agent sha256 {meta['agent_sha256'][:12]}")
    if mb > MAX_ZIP_MB:
        print(f"WARNING over {MAX_ZIP_MB} MB: GitHub won't attach it; share a link in the "
              "Submission issue instead")
    print("Next: open a 'Submission' issue on the benchmark repository and attach this zip.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
