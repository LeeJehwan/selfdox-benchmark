"""Reference agent E: Qwen3-VL-4B finds location clues, blurs them, re-checks, gives a verdict.

Runs under the proposed Phase 3 rules (docs/design.md §2.1): the only model is
Qwen/Qwen3-VL-4B-Instruct at a pinned revision, run offline with vLLM, no fine-tuning.
bf16 needs about 9 GB of GPU memory (one RTX 3090 is enough).

    HF_HUB_OFFLINE=1 python agents/e_qwen.py --input <images> --output outputs/E_qwen

Self-contained on purpose (Pillow, numpy and vLLM only; no selfdox import) so it runs on
the lab cluster and doubles as the model example for Phase 3 teams.

Per photo:
  1. find      the model lists clues as bbox_2d on a 0-1000 grid -> pixel boxes
  2. hide      a strong Gaussian blur over every hideable clue box (padded)
  3. re-check  the model looks at the edited photo; hideable clues not already blurred
               are blurred too (at most 2 re-checks)
  4. verdict   an unhideable critical clue -> cannot_hide; the last re-check still found
               something (or could not be parsed) -> still_risky; otherwise safe
The edited photo is saved as a new JPEG, so no metadata (and no GPS) is carried over.
Every step runs as one vLLM batch over a chunk of photos; finished photos are skipped
when the run is restarted, and predictions.json is rewritten (atomically) after every
chunk, so a run stopped by the time limit still leaves a valid file. Each prediction
carries an optional `meta` with its cost: model calls, prompt and completion tokens, and
seconds (the wall time of the batches it was in, split evenly over their photos).

Decoding is free (no JSON-schema constraint) and parsed leniently. With a schema that
wraps the list in {"clues": [...]}, Qwen3-VL-4B answered {"clues": []} for every photo of
a 5-photo probe, while the same prompt without the constraint listed 2-15 clues per photo
(the model prefers a bare JSON list). Unparsable answers are retried once with a
frequency penalty; a cut-off list keeps its complete items.
"""
import argparse
import base64
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
BENCHMARK_VERSION = "0.3"
TYPES = ["shop_sign", "address_plate", "apt_name", "apt_block", "street_sign", "transit",
         "phone_number", "utility_mark", "parcel_label", "school_name", "landmark",
         "skyline_view", "facade", "reflection", "other"]
MAX_RECHECKS = 2

FIND_PROMPT = """This photo was taken in South Korea. You are a privacy assistant: find every visible thing that would help a stranger find WHERE this photo was taken, so it can be hidden.

For each one give:
- bbox_2d: a tight box [x1, y1, x2, y2] on a 0-1000 grid. Box each sign or line of text separately.
- type: one of shop_sign, address_plate, apt_name, apt_block, street_sign, transit, phone_number, utility_mark, parcel_label, school_name, landmark, skyline_view, facade, reflection, other
- text: the legible text as written (keep Korean as Korean), or "" if there is none
- severity: "critical" if it alone narrows the location to about 1 km (a shop name, a phone number, a road-name address plate, a station or bus stop name, an apartment or school name, a famous landmark), else "supporting"
- hideable: false only if hiding it would destroy the photo (the landmark or skyline is the main subject), else true
- reason: a few words

List each thing once. Do NOT list: license plates, faces, car makes, product logos, menus, prices, opening hours, generic words ("Coffee", "SALE"), warning or instruction signs without a place name, arrows or icons, text too blurry to read. Blurred areas are already hidden; ignore them.
If nothing reveals the location, return an empty list.

Answer with JSON only."""

# The item format, for reference and tests. Not used as a decoding constraint (see above).
SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["clues"],
    "properties": {"clues": {"type": "array", "maxItems": 15, "items": {
        "type": "object", "additionalProperties": False,
        "required": ["bbox_2d", "type", "text", "severity", "hideable", "reason"],
        "properties": {
            "bbox_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 4,
                        "maxItems": 4},
            "type": {"enum": TYPES}, "text": {"type": "string"},
            "severity": {"enum": ["critical", "supporting"]}, "hideable": {"type": "boolean"},
            "reason": {"type": "string"}}}}}}


# ---------- geometry and editing (pure, testable without a GPU) ----------

def to_px(b, W, H):
    """bbox_2d [x1, y1, x2, y2] on 0-1000 -> [x, y, w, h] pixels, clipped, or None.
    Free decoding sometimes gives strings, None or the wrong number of values (seen on
    the cluster); such boxes are skipped."""
    try:
        vals = [float(v) for v in b]
    except (TypeError, ValueError):
        return None
    if len(vals) != 4:
        return None
    x1, y1, x2, y2 = (min(1000, max(0, v)) for v in vals)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x, y = round(x1 * W / 1000), round(y1 * H / 1000)
    w, h = round(x2 * W / 1000) - x, round(y2 * H / 1000) - y
    w, h = min(w, W - x), min(h, H - y)
    return [x, y, w, h] if w >= 4 and h >= 4 else None


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def covered(box, hidden):
    """True if the box's center lies in an already blurred box or overlaps one a lot."""
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    return any(h[0] <= cx <= h[0] + h[2] and h[1] <= cy <= h[1] + h[3] or iou(box, h) > 0.5
               for h in hidden)


def pad_box(box, W, H, frac=0.12, minimum=4):
    x, y, w, h = box
    mx, my = max(minimum, round(frac * w)), max(minimum, round(frac * h))
    x0, y0 = max(0, x - mx), max(0, y - my)
    return [x0, y0, min(W, x + w + mx) - x0, min(H, y + h + my) - y0]


def blur_box(img, box):
    """Blur strong enough for the scorer: sigma 0.35 x the box's short side (the
    calibration hides every line from 0.15 x the line height on)."""
    x, y, w, h = box
    r = max(6.0, 0.35 * min(w, h))
    m = int(3 * r)
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(img.width, x + w + m), min(img.height, y + h + m)
    patch = img.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(r))
    img.paste(patch.crop((x - x0, y - y0, x - x0 + w, y - y0 + h)), (x, y))
    return img


def clean(clues, W, H):
    """Model JSON clues -> pixel clues, deduplicated."""
    out = []
    for c in clues or []:
        box = to_px(c.get("bbox_2d", [0, 0, 0, 0]), W, H)
        if box is None or any(iou(box, o["bbox"]) > 0.8 for o in out):
            continue
        out.append({"bbox": box, "type": c.get("type", "other"), "text": c.get("text", ""),
                    "severity": c.get("severity", "supporting"),
                    "hideable": bool(c.get("hideable", True)), "reason": c.get("reason", "")})
    return out


def verdict(all_clues, last_new, last_failed):
    if any(c["severity"] == "critical" and not c["hideable"] for c in all_clues):
        return "cannot_hide"
    if last_failed or last_new:
        return "still_risky"
    return "safe"


def has_gps(path):
    with Image.open(path) as im:
        return bool(im.getexif().get_ifd(0x8825))


def data_url(img):
    buf = BytesIO()
    img.save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------- the model ----------

class VLM:
    def __init__(self, model, revision, max_pixels, seed=0, max_tokens=2048):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model, revision=revision, dtype="bfloat16", max_model_len=16384,
                       gpu_memory_utilization=0.90, seed=seed,
                       limit_mm_per_prompt={"image": 1, "video": 0},
                       mm_processor_kwargs={"max_pixels": max_pixels})
        self.sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=seed)
        self.retry = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=seed,
                                    frequency_penalty=0.5)

    def find(self, images):
        """[PIL image] -> [(clue list or None if unparsable, raw text, usage)], usage =
        {"model_calls", "prompt_tokens", "completion_tokens"} for that image."""
        convs = [[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url(im)}},
            {"type": "text", "text": FIND_PROMPT}]}] for im in images]
        got = self.llm.chat(convs, self.sp, use_tqdm=False)
        outs = [o.outputs[0].text for o in got]
        usage = [_usage(o) for o in got]
        res = [parse_clues(t) for t in outs]
        bad = [i for i, r in enumerate(res) if r is None]
        if bad:                    # repetition loops: retry once with a frequency penalty
            again = self.llm.chat([convs[i] for i in bad], self.retry, use_tqdm=False)
            for i, o in zip(bad, again):
                outs[i] = o.outputs[0].text
                res[i] = parse_clues(outs[i])
                usage[i] = {k: usage[i][k] + v for k, v in _usage(o).items()}
        return list(zip(res, outs, usage))


def _usage(o):
    """Token counts of one vLLM RequestOutput."""
    return {"model_calls": 1, "prompt_tokens": len(o.prompt_token_ids or []),
            "completion_tokens": len(o.outputs[0].token_ids or [])}


def parse_clues(text):
    """The clue list from a free-form answer: a JSON list, or {"clues": [...]}, optionally in
    ```json fences. A list cut off mid-way keeps its complete items. None if nothing parses."""
    t = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", (text or "").strip())
    starts = [i for i in (t.find("["), t.find("{")) if i >= 0]
    if not starts:
        return None
    t = t[min(starts):]
    try:
        v = json.loads(t)
    except json.JSONDecodeError:
        v = _salvage(t)
    if isinstance(v, dict):
        v = v.get("clues")
    if not isinstance(v, list):
        return None
    return [c for c in v if isinstance(c, dict)]


def _salvage(t):
    """Complete objects from the front of a list that was cut off (e.g. max_tokens)."""
    i = t.find("[")
    if i < 0:
        return None
    dec, items, pos = json.JSONDecoder(), [], i + 1
    while True:
        while pos < len(t) and t[pos] in " \t\r\n,":
            pos += 1
        if pos >= len(t) or t[pos] != "{":
            break
        try:
            obj, pos = dec.raw_decode(t, pos)
        except json.JSONDecodeError:
            break
        items.append(obj)
    return items if items else None


# ---------- the agent loop ----------

def run_chunk(vlm, paths, out, trace):
    imgs, states = [], []
    for p in paths:
        with Image.open(p) as im:
            img = ImageOps.exif_transpose(im).convert("RGB") if im.getexif().get(0x0112, 1) != 1 \
                else im.convert("RGB")
        imgs.append(img)
        states.append({"id": p.stem, "gps": has_gps(p), "clues": [], "hidden": [],
                       "last_new": [], "last_failed": False, "rounds": 0,
                       "meta": {"model_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                "seconds": 0.0}})
    todo = list(range(len(paths)))
    for rnd in range(1 + MAX_RECHECKS):
        if not todo:
            break
        t = time.time()
        found = vlm.find([imgs[i] for i in todo])
        share = (time.time() - t) / len(todo)
        nxt = []
        for i, (clues, raw, usage) in zip(todo, found):
            s, img = states[i], imgs[i]
            s["rounds"] = rnd + 1
            for k, v in usage.items():
                s["meta"][k] += v
            s["meta"]["seconds"] += share
            s["last_failed"] = clues is None
            new = [c for c in clean(clues, img.width, img.height)
                   if not (rnd and covered(c["bbox"], s["hidden"]))]
            trace.write(json.dumps({"id": s["id"], "round": rnd, "raw": raw}, ensure_ascii=False)
                        + "\n")
            s["clues"].extend(new)
            hide = [c for c in new if c["hideable"]]
            for c in hide:
                box = pad_box(c["bbox"], img.width, img.height)
                blur_box(img, box)
                s["hidden"].append(box)
            s["last_new"] = hide
            if hide and rnd < MAX_RECHECKS:
                nxt.append(i)
        todo = nxt
    preds = []
    for s, img in zip(states, imgs):
        rel = f"images/{s['id']}.jpg"
        img.save(out / rel, "JPEG", quality=90)
        clues = [{"bbox": c["bbox"], "type": c["type"], "text": c["text"], "reason": c["reason"]}
                 for c in s["clues"]]
        if s["gps"]:
            clues.append({"bbox": None, "type": "exif_gps", "reason": "GPS in EXIF (removed)"})
        meta = {**s["meta"], "seconds": round(s["meta"]["seconds"], 2)}
        preds.append({"id": s["id"], "clues": clues, "edited_image": rel,
                      "verdict": verdict(s["clues"], s["last_new"], s["last_failed"]),
                      "meta": meta, "rounds": s["rounds"]})
    return preds


def write_predictions(out, paths, done):
    """predictions.json with every photo done so far, in input order; written to a temp
    file and renamed, so a reader (or a killed run) never sees half a file."""
    preds = [{k: v for k, v in done[p.stem].items() if k != "rounds"} for p in paths
             if p.stem in done]
    tmp = out / "predictions.json.tmp"
    tmp.write_text(json.dumps(
        {"agent_name": "E_qwen3vl4b", "benchmark_version": BENCHMARK_VERSION,
         "model": f"{MODEL}@{REVISION}", "predictions": preds}, ensure_ascii=False, indent=1))
    os.replace(tmp, out / "predictions.json")
    return preds


def main(argv=None):
    ap = argparse.ArgumentParser(description="SelfDox reference agent E (Qwen3-VL-4B)")
    ap.add_argument("--input", "--inputs", dest="inputs", required=True)
    ap.add_argument("--output", "--out", dest="out", default="outputs/E_qwen")
    ap.add_argument("--model", default=MODEL, help="HF id or a local folder (e.g. /models/...)")
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--max-pixels", type=int, default=2048 * 2048)
    ap.add_argument("--chunk", type=int, default=10,
                    help="photos per vLLM batch; predictions.json is rewritten after each")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path(args.inputs).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        paths = paths[:args.limit]
    part = out / "predictions.partial.jsonl"
    done = {}
    if part.exists():
        for line in part.open(encoding="utf-8"):
            r = json.loads(line)
            done[r["id"]] = r
    todo = [p for p in paths if p.stem not in done]
    print(f"{len(paths)} photos, {len(done)} done, {len(todo)} to go", flush=True)
    write_predictions(out, paths, done)        # a valid file from the start
    if todo:
        revision = None if Path(args.model).exists() else args.revision
        vlm = VLM(args.model, revision, args.max_pixels)
        with part.open("a", encoding="utf-8") as pf, \
                (out / "trace.jsonl").open("a", encoding="utf-8") as trace:
            for k in range(0, len(todo), args.chunk):
                t = time.time()
                for r in run_chunk(vlm, todo[k:k + args.chunk], out, trace):
                    pf.write(json.dumps(r, ensure_ascii=False) + "\n")
                    done[r["id"]] = r
                pf.flush()
                write_predictions(out, paths, done)
                print(f"chunk {k // args.chunk + 1}: {min(k + args.chunk, len(todo))}/"
                      f"{len(todo)} in {time.time() - t:.0f}s", flush=True)
    preds = write_predictions(out, paths, done)
    verdicts = {}
    for p in preds:
        verdicts[p["verdict"]] = verdicts.get(p["verdict"], 0) + 1
    print(f"wrote {len(preds)} predictions -> {out}/predictions.json  verdicts {verdicts}")


if __name__ == "__main__":
    main()
