"""SelfDox agent template. Replace find_clues() with your detector; hide() and the verdict
rule work as they are.

    python run.py --input <photos> --output <folder>

For every input photo this writes an edited copy (same size, PNG, no metadata) and one
entry in predictions.json. predictions.json is rewritten after every photo (to a temp
file, then renamed), so a run stopped by the time limit still leaves a valid file.
Check your output with:
    python -m selfdox.validate --sub <folder> --inputs <photos>
"""
import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageFilter

BENCHMARK_VERSION = "0.3"     # the answer-key format version this agent was built for


def find_clues(img):
    """Return the location clues you see, as a list of
    {"bbox": [x, y, w, h], "type": ..., "text": ..., "severity": "critical" | "supporting",
     "hideable": True | False}. `severity` and `hideable` are only used by this template's
    verdict rule; they are not part of the submission. Replace this stub with your detector
    (OCR, a VLM, ...)."""
    return []


def hide(img, clues):
    """Blur every hideable clue box. A margin helps: landmarks and skylines are judged on
    coarse structure that reaches past a tight box (docs/participants.md, FAQ)."""
    for c in clues:
        if not c.get("hideable", True):
            continue
        x, y, w, h = c["bbox"]
        m = round(0.25 * min(w, h)) if c.get("type") in ("landmark", "skyline_view", "facade") \
            else round(0.1 * min(w, h))
        x0, y0 = max(0, x - m), max(0, y - m)
        x1, y1 = min(img.width, x + w + m), min(img.height, y + h + m)
        r = max(6.0, 0.35 * min(w, h))
        img.paste(img.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(r)), (x0, y0))
    return img


def verdict(clues, recheck):
    """The reference agent's rule (agents/e_qwen.py): a critical clue that can't be hidden
    -> cannot_hide; something still found after hiding -> still_risky; otherwise safe.
    "safe" is only worth it when you really looked: a false "safe" costs 10 points."""
    if any(c.get("severity") == "critical" and not c.get("hideable", True) for c in clues):
        return "cannot_hide"
    return "still_risky" if recheck else "safe"


def has_gps(path):
    with Image.open(path) as im:
        return bool(im.getexif().get_ifd(0x8825))


def write_predictions(out, preds):
    tmp = out / "predictions.json.tmp"
    tmp.write_text(json.dumps({"agent_name": "template", "benchmark_version": BENCHMARK_VERSION,
                               "predictions": preds}, ensure_ascii=False, indent=1))
    os.replace(tmp, out / "predictions.json")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "--inputs", dest="input", required=True)
    ap.add_argument("--output", "--out", dest="output", required=True)
    args = ap.parse_args(argv)
    out = Path(args.output)
    (out / "images").mkdir(parents=True, exist_ok=True)
    preds = []
    write_predictions(out, preds)                    # a valid file from the start
    for p in sorted(Path(args.input).glob("*.jpg")):
        with Image.open(p) as im:
            img = im.convert("RGB")                  # pixels only: EXIF (incl. GPS) is dropped
        clues = find_clues(img)
        img = hide(img, clues)
        left = [c for c in find_clues(img) if c.get("hideable", True)]   # look again
        img.save(out / "images" / f"{p.stem}.png")   # PNG without metadata, same size
        report = [{k: c[k] for k in ("bbox", "type", "text") if k in c} for c in clues]
        if has_gps(p):
            report.append({"bbox": None, "type": "exif_gps", "reason": "GPS in EXIF (removed)"})
        preds.append({"id": p.stem, "clues": report, "edited_image": f"images/{p.stem}.png",
                      "verdict": verdict(clues, left)})
        write_predictions(out, preds)


if __name__ == "__main__":
    main()
