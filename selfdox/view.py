"""See why each photo scored what it did: one HTML page per run.

    python -m selfdox.view --gt data/dev/gt.json --sub outputs --results results.json \
        --out view/

For every photo the page shows the edited image with the answer-key boxes drawn on it:
green = hidden, red = still recoverable, grey = not hideable (leave it), and your
reported boxes dashed in blue. Next to it: score, outcome, verdict vs. true state and
the parts (L, E, R, V). Photos are sorted worst first. `results.json` is the
`selfdox.score --out` file of the same run. Only works where you have the answer key
(dev); it opens offline in any browser.
"""
import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw

from .io import safe_join

THUMB = 900
COLORS = {True: (30, 170, 70), False: (220, 40, 40), None: (150, 150, 150)}


def _dashed_rect(draw, box, color, width, dash=10):
    x0, y0, x1, y1 = box
    for a, b, horizontal in ((x0, x1, True), (y0, y1, False)):
        for t in range(int(a), int(b), dash * 2):
            e = min(t + dash, b)
            if horizontal:
                draw.line([(t, y0), (e, y0)], fill=color, width=width)
                draw.line([(t, y1), (e, y1)], fill=color, width=width)
            else:
                draw.line([(x0, t), (x0, e)], fill=color, width=width)
                draw.line([(x1, t), (x1, e)], fill=color, width=width)


def overlay(img, gt_image, result, pred):
    """The edited image, scaled down, with key boxes (by neutralized state) and the agent's
    reported boxes."""
    img = img.convert("RGB")
    k = min(1.0, THUMB / max(img.size))
    img = img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))))
    d = ImageDraw.Draw(img)
    state = {c["id"]: c.get("neutralized") for c in (result or {}).get("clues", [])}
    for c in gt_image["clues"]:
        if c.get("bbox") is None:
            continue
        x, y, w, h = (v * k for v in c["bbox"])
        hidden = state.get(c["id"]) if c["hideable"] else None
        d.rectangle([x, y, x + w, y + h], outline=COLORS[hidden],
                    width=4 if c["severity"] == "critical" else 2)
    for c in (pred or {}).get("clues", []):
        if c.get("bbox"):
            x, y, w, h = (v * k for v in c["bbox"])
            _dashed_rect(d, (x, y, x + w, y + h), (40, 110, 230), 2)
    return img


def page(rows):
    css = ("body{font:14px system-ui,sans-serif;margin:16px;background:#f6f7f9;color:#151922}"
           ".p{display:flex;flex-wrap:wrap;gap:14px;background:#fff;border:1px solid #dfe3ea;"
           "border-radius:10px;padding:12px;margin:12px 0}.p img{max-width:min(100%,900px);"
           "height:auto}.m{min-width:240px;flex:1}.m td{padding:2px 8px 2px 0}"
           ".leak{color:#c62828;font-weight:700}.ruined{color:#b26a00;font-weight:700}"
           ".cleared{color:#1b7a3a;font-weight:700}code{font-size:12px}")
    legend = ("<p>Answer-key boxes: <b style='color:#1eaa46'>green</b> hidden, "
              "<b style='color:#dc2828'>red</b> still recoverable, <b style='color:#969696'>"
              "grey</b> not hideable (leave it); thick = critical. Your reported boxes are "
              "dashed blue. Worst photos first.</p>")
    body = []
    for r in rows:
        parts = r["parts"]
        cells = "".join(f"<tr><td>{k}</td><td>{'–' if v is None else f'{v:.3f}'}</td></tr>"
                        for k, v in parts.items())
        miss = [f"{c['id']} {c['type']}" for c in r["clues"]
                if c["hideable"] and not c["neutralized"]]
        body.append(
            f"<div class='p'><img src='{html.escape(r['thumb'])}' loading='lazy'>"
            f"<div class='m'><h3>{html.escape(r['id'])} · {r['score']:.1f}</h3>"
            f"<p class='{r['outcome']}'>{html.escape(str(r['outcome']).replace('_', ' '))}</p>"
            f"<p>verdict <b>{html.escape(str(r['verdict']))}</b>, true state "
            f"<b>{html.escape(str(r['true_state']))}</b>, expected "
            f"{html.escape(str(r['expected_verdict']))}, group {html.escape(str(r['group']))}"
            f"</p>{'<p>hard fail: <code>' + html.escape(r['hard_fail']) + '</code></p>' if r['hard_fail'] else ''}"
            f"<p>not hidden: {html.escape(', '.join(miss)) or 'none'}</p>"
            f"<table>{cells}</table></div></div>")
    return (f"<!doctype html><meta charset='utf-8'><title>SelfDox results</title>"
            f"<style>{css}</style><h1>SelfDox results</h1>{legend}{''.join(body)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="HTML view of a scored SelfDox run")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--sub", required=True, help="the submission folder that was scored")
    ap.add_argument("--results", required=True, help="selfdox.score --out file of that run")
    ap.add_argument("--out", default="view")
    args = ap.parse_args(argv)
    gt = {im["id"]: im for im in json.loads(Path(args.gt).read_text())["images"]}
    res = json.loads(Path(args.results).read_text())
    if "multi_run" in res:
        res = res["runs"][0]
    sub = Path(args.sub)
    preds = {p["id"]: p for p in json.loads((sub / "predictions.json").read_text())["predictions"]}
    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    rows = []
    for r in sorted(res["images"], key=lambda r: r["score"]):
        pred = preds.get(r["id"])
        path = safe_join(sub, pred["edited_image"]) if pred else None
        if path is None or not path.is_file():
            path = Path(args.gt).parent / gt[r["id"]]["file"]
        with Image.open(path) as im:
            overlay(im, gt[r["id"]], r, pred).save(out / "img" / f"{r['id']}.jpg", quality=85)
        rows.append({**r, "thumb": f"img/{r['id']}.jpg"})
    (out / "index.html").write_text(page(rows), encoding="utf-8")
    print(f"{len(rows)} photos -> {out / 'index.html'}")


if __name__ == "__main__":
    main()
