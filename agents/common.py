"""Shared runner for the reference agents.

Each agent is a function (path) -> (edited, clues, verdict) where edited is a PIL image
(saved without metadata) or None (input file copied byte for byte).
"""
import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

from selfdox import config as C
from selfdox.io import metadata_report


def has_gps(path):
    return bool({"gps_exif", "gps_xmp"} & set(metadata_report(path).problems))


def run(name, agent, argv=None, fmt="png"):
    ap = argparse.ArgumentParser(description=f"Reference agent {name}")
    ap.add_argument("--input", "--inputs", dest="inputs", default="data/dev/images")
    ap.add_argument("--output", "--out", dest="out", default=f"outputs/{name}")
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    preds = []
    for p in sorted(Path(args.inputs).glob("*.jpg")):
        edited, clues, verdict = agent(p)
        if edited is None:
            rel = f"images/{p.name}"
            shutil.copyfile(p, out / rel)
        else:
            rel = f"images/{p.stem}.{fmt}"
            kw = {"quality": 75, "subsampling": "4:2:0"} if fmt == "jpg" else {}
            edited.convert("RGB").save(out / rel, "JPEG" if fmt == "jpg" else "PNG", **kw)
        preds.append({"id": p.stem, "clues": clues, "edited_image": rel, "verdict": verdict})
        print(f"{p.stem}: {len(clues)} clue(s), {verdict}")
    (out / "predictions.json").write_text(json.dumps(
        {"agent_name": name, "benchmark_version": C.VERSION, "predictions": preds},
        ensure_ascii=False, indent=1))


def open_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")
