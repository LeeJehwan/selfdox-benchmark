"""Check and pack a Phase 3 submission.

    python -m selfdox.submit --agent my_agent --team team7 --contact you@example.com \
        --model "Qwen/Qwen3-VL-4B-Instruct@ebb281e" [--model ...] --tools "EasyOCR 1.7.2" \
        --temperature 0 --seed 0 --test-sub outputs_test [--smoke 3] [--out submission.zip]

Checks the agent folder the way the organizers will run it (docs/participants.md §6):
  - it has run.py or a selfdox_run.json command, and a requirements.txt
  - no answer keys inside, no file over 50 MB (weights come from {models}, not the folder)
  - --smoke N: runs the folder's command with the official runner's settings (offline
    environment, Python-level network block, a copy of N dev photos) using this Python,
    then validates the output. It does not install requirements.txt; run it from a venv
    where they are installed.
  - --test-sub: validates your own outputs on data/test/images and also writes
    test_outputs_<team>.zip next to the submission zip: predictions.json and the edited
    images (JPEG files as they are, other formats saved as JPEG quality 92). The
    site scores it for the interim board.
  - --model: each open-weight model the agent loads, as Hugging Face repo@revision, each
    at most 4B parameters (repeat for several; "none" if it uses none).
  - --weights: a weight file that is not on Hugging Face (e.g. an OCR detector), as
    "<https url>#sha256=<hex>" (repeat for several); we download and check it into
    {models} before the final run. Files under 50 MB may ship in the folder instead.
Writes a zip with the agent folder, metadata.json (team, contact, models, tools, decoding,
the folder's sha256) and test_predictions.json. Upload both files on the submission site,
https://huggingface.co/spaces/wpghks7/selfdox (sign in with Hugging Face, enter your team
code, add submission_<team>.zip and test_outputs_<team>.zip). After the deadline we run the agent folder
of your last submission ourselves on test and hidden (docs/participants.md §6-7).
"""
import argparse
import datetime
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from . import config as C
from .schema import input_sizes, submission_errors

MAX_FILE_MB = 50
WEIGHTS_RE = re.compile(r"^https://\S+#sha256=[0-9a-f]{64}$")
SITE_URL = "https://huggingface.co/spaces/wpghks7/selfdox"
MAX_ZIP_MB = 30          # the submission site's limit
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


def pack_outputs(sub_dir, out):
    """test_outputs zip for the bot: predictions.json with every edited_image pointing to
    images/<id>.jpg, and those images (JPEG kept as is, anything else saved as JPEG q92)."""
    from PIL import Image

    from .io import safe_join
    sub_dir = Path(sub_dir)
    doc = json.loads((sub_dir / "predictions.json").read_text())
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for p in doc["predictions"]:
            src = safe_join(sub_dir, p["edited_image"])
            if src is None or not src.is_file():
                raise SystemExit(f"{p['id']}: edited_image {p['edited_image']!r} is not a file "
                                 "inside the outputs folder")
            name = f"images/{p['id']}.jpg"
            if src.suffix.lower() in (".jpg", ".jpeg"):
                z.write(src, name)
            else:
                buf = io.BytesIO()
                Image.open(src).convert("RGB").save(buf, "JPEG", quality=92)
                z.writestr(name, buf.getvalue())
            p["edited_image"] = name
        z.writestr("predictions.json", json.dumps(doc, ensure_ascii=False, indent=1))
    return out


def tree_sha256(folder):
    """sha256 of exactly the files that go into the zip, computed the way
    tools.run_container.tree_digest does it, so the digest of the unpacked agent/ folder
    is the same (an outputs/ folder, left out of the zip, does not change it)."""
    import hashlib
    h, root = hashlib.sha256(), Path(folder)
    for p in agent_files(root):
        h.update(str(p.relative_to(root)).encode() + b"\0")
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        h.update(b"\0")
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check and pack a SelfDox submission.")
    ap.add_argument("--agent", required=True, help="agent folder (agent_template layout)")
    ap.add_argument("--team", required=True)
    ap.add_argument("--contact", default="", help="how to reach you (optional; it goes into "
                    "metadata.json, which is public on the submissions dataset)")
    ap.add_argument("--model", required=True, action="append",
                    help='an open-weight model the agent loads, at most 4B parameters (the '
                    'size it is published as), as Hugging Face repo@revision, e.g. '
                    '"Qwen/Qwen3-VL-4B-Instruct@ebb281e"; repeat for several, or "none"')
    ap.add_argument("--weights", action="append", default=[],
                    help='a weight file not on Hugging Face, "<https url>#sha256=<hex>" '
                    "(repeat for several); we download it into {models} for the final run")
    ap.add_argument("--tools", default="", help="other tools and models, with versions")
    ap.add_argument("--temperature", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--test-sub", help="your agent's outputs on data/test/images: also "
                    "writes test_outputs_<team>.zip for the submission site")
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
    for w in args.weights:
        if not WEIGHTS_RE.match(w):
            errors.append(f"--weights {w!r}: write it as <https url>#sha256=<64 hex digits>")
    for w in warnings:
        print(f"WARNING {w}")
    for e in errors:
        print(f"ERROR   {e}")
    if errors:
        print(f"\n{len(errors)} error(s); nothing written.")
        return 1
    models = [m for m in args.model if m.strip().lower() != "none"]
    meta = {"team": args.team, "contact": args.contact, "models": models,
            "weights": args.weights,
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
        print(f"WARNING over {MAX_ZIP_MB} MB: the site rejects it; "
              "make the agent folder smaller")
    if args.test_sub:
        outputs = pack_outputs(args.test_sub, out.with_name(f"test_outputs_{args.team}.zip"))
        print(f"{outputs} ({outputs.stat().st_size / 2 ** 20:.0f} MB): upload it on the "
              "submission site too (limit 300 MB)")
    print(f"Next: upload {out.name} and the test outputs zip on {SITE_URL} (sign in with "
          "Hugging Face, enter your team code). The score appears in 5 to 10 minutes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
