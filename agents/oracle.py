"""Upper bound ("oracle"): an agent that knows the answer key.

    uv run python -m agents.oracle --gt <gt.json> --inputs <images> --out outputs/oracle300

For every photo: blur each hideable answer-key clue with reference agent E's blur (sigma
0.35 x the box's short side), report exactly the key's clues, and answer the expected
verdict (`cannot_hide` where a person said so, else `safe`). Saved as PNG without
metadata, so GPS is gone. It shows how far the answer key itself lets an agent go; rerun
it whenever the key changes.

Margins: a text box is padded by 10% of its size, which stays inside the margin edit
economy forgives (E_DILATE_FRAC). A landmark, skyline or facade is padded by 25% of its
size: the scorer judges those on coarse structure that reaches past a tight box, so a
margin is needed to hide them, and it costs a little edit economy.
"""
import argparse
import json
from pathlib import Path

from PIL import Image

from agents.e_qwen import blur_box, pad_box
from selfdox import config as C

PAD = {"text": 0.10, "non_text": 0.25}


def oracle(im, img):
    for c in im["clues"]:
        if c.get("bbox") is not None and c["hideable"]:
            frac = PAD["non_text" if c["type"] in C.NON_TEXT_TYPES else "text"]
            blur_box(img, pad_box(c["bbox"], img.width, img.height, frac=frac))
    clues = [{"bbox": c["bbox"], "type": c["type"]} for c in im["clues"]]
    verdict = "cannot_hide" if im.get("expected_verdict") == "cannot_hide" else "safe"
    return img, clues, verdict


def main(argv=None):
    ap = argparse.ArgumentParser(description="Oracle agent (knows the answer key)")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--inputs", required=True, help="folder with <id>.jpg")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    preds = []
    for im in json.loads(Path(args.gt).read_text())["images"]:
        with Image.open(Path(args.inputs) / f"{im['id']}.jpg") as src:
            img = src.convert("RGB")
        img, clues, verdict = oracle(im, img)
        rel = f"images/{im['id']}.png"
        img.save(out / rel, "PNG")
        preds.append({"id": im["id"], "clues": clues, "edited_image": rel, "verdict": verdict})
    (out / "predictions.json").write_text(json.dumps(
        {"agent_name": "oracle", "benchmark_version": C.VERSION, "predictions": preds},
        ensure_ascii=False, indent=1))
    print(f"{len(preds)} photos -> {out}")


if __name__ == "__main__":
    main()
