"""The quick check and the scoring, made of the submission bot's functions
(tools/bot_score.py), so the site scores exactly like the bot did."""
import json
import os
import subprocess
from pathlib import Path

from selfdox.schema import submission_format_errors
from tools import bot_score as B

Refused = B.Refused          # its message goes to the team
OPENSSL = ["-aes-256-cbc", "-pbkdf2", "-md", "sha256", "-iter", "200000", "-salt"]  # = encrypt_key
RESULT_KEYS = ("raw", "raw_ci", "score", "ci", "cleared", "leak", "false_safe", "n_safe", "pass")


def prepare_key(enc_path, passphrase, images_dir, workdir):
    """Decrypt the test key into workdir next to a link to the test photos (the key's "file"
    paths are images/<name>); returns the key's path. The key is never published."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    gt = workdir / "gt.json"
    subprocess.run(["openssl", "enc", "-d", *OPENSSL, "-in", str(enc_path), "-out", str(gt),
                    "-pass", "env:SELFDOX_KEY"], check=True, capture_output=True,
                   env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "SELFDOX_KEY": passphrase})
    link = workdir / "images"
    if not link.exists():
        link.symlink_to(Path(images_dir).resolve())
    return gt


def test_ids(gt_path):
    return {im["id"] for im in json.loads(Path(gt_path).read_text())["images"]}


def quick_check(agent_zip, outputs_zip, ids, workdir):
    """(agent metadata, warnings), or Refused: the agent zip's files match its sha256, the
    outputs unpack safely, predictions.json has the format and test photo ids."""
    meta = B.agent_metadata(agent_zip)
    sub = B.safe_unzip_outputs(outputs_zip, Path(workdir) / "sub")
    doc, errors = submission_format_errors(sub)
    if errors:
        raise Refused("predictions.json does not match the format: "
                      + "; ".join(B.quote(e, 160) for e in errors[:3])
                      + ". Run `selfdox.validate` on your outputs")
    got = {p.get("id") for p in doc["predictions"]}
    if not got & set(ids):
        raise Refused("predictions.json has none of the test photo ids; run your agent on "
                      "data/test/images")
    missing = len(set(ids) - got)
    return meta, ([f"{missing} test photo(s) have no prediction and will score 0"]
                  if missing else [])


def score(outputs_zip, gt_path, refs, workdir):
    """Aggregates for the board and the score card; never per-photo numbers."""
    sub = B.safe_unzip_outputs(outputs_zip, Path(workdir) / "sub")
    stats, info = B.score_outputs(gt_path, sub, refs)
    return {**{k: stats[k] for k in RESULT_KEYS}, "n_zero": info["n_zero"] + info["n_unscored"]}
