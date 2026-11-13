"""Per-image scoring (PU, R, V, hard fails) and aggregation. See docs/design.md §7."""
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

from . import config as C
from .io import ImageError, load_rgb, metadata_report, safe_join
from .neutralize import clue_neutralized

GPS_PROBLEMS = {"gps_exif", "gps_xmp"}

# Reported next to the score (none of this changes it): what each photo means to the person
# posting it, checked in this order (the first that applies wins).
OUTCOMES = ["leak", "ruined", "cleared", "correct_warning", "unneeded_warning"]
#   leak              a false "safe" (also on a photo a person marked cannot_hide), or GPS left
#   ruined            no usable photo (any other hard fail) or edit economy E < PASS_E
#   cleared           "safe", truly safe, E >= PASS_E: the user can post it
#   correct_warning   still_risky / cannot_hide on a photo that is not safe
#   unneeded_warning  a warning on a photo that was already safe after the edit
GROUPS = ["needs_edit", "leave_alone", "unhideable"]
#   unhideable   expected verdict cannot_hide, or a critical clue that can't be hidden
#   needs_edit   otherwise, a hideable critical clue with a box
#   leave_alone  the rest (nothing or only supporting clues to hide; GPS may still be there)
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 0


@dataclass
class ImageScore:
    id: str
    category: str
    score: float
    parts: dict
    verdict: str | None = None
    true_state: str | None = None
    false_safe: bool = False
    hard_fail: str | None = None
    clues: list = field(default_factory=list)
    passed: bool = False
    expected_verdict: str | None = None
    group: str | None = None
    outcome: str | None = None


# the verdict that earns full verdict points in each true state
RIGHT_VERDICT = {"safe": "safe", "unsafe": "still_risky", "unhideable": "cannot_hide"}


def image_passed(r):
    """No hard fail, every hideable critical clue hidden, E >= PASS_E, and the right verdict
    for the true state (RIGHT_VERDICT)."""
    if r.hard_fail or not r.parts or r.false_safe:
        return False
    if r.parts.get("E") is None or r.parts["E"] < C.PASS_E:
        return False
    if r.true_state not in RIGHT_VERDICT or r.verdict != RIGHT_VERDICT[r.true_state]:
        return False
    return all(c["neutralized"] for c in r.clues if c["hideable"] and c["severity"] == "critical")


def edit_score(E):
    """What edit economy E is worth in the score: max(0, 1 - E_PENALTY * (1 - E))."""
    return max(0.0, 1.0 - C.E_PENALTY * (1.0 - E))


def photo_group(gt):
    """needs_edit, leave_alone or unhideable, from the answer key alone."""
    px = [c for c in gt.get("clues", []) if c.get("bbox") is not None]
    if gt.get("expected_verdict") == "cannot_hide" or \
            any(c["severity"] == "critical" and not c["hideable"] for c in px):
        return "unhideable"
    if any(c["severity"] == "critical" and c["hideable"] for c in px):
        return "needs_edit"
    return "leave_alone"


def image_outcome(r):
    """One of OUTCOMES for a scored image (see the definitions at the top)."""
    hf = r.hard_fail or ""
    gps_left = any(g in hf for g in GPS_PROBLEMS)
    # true_state is None when the edited photo could not be judged (missing, wrong size...)
    false_safe = r.verdict == "safe" and r.true_state is not None and (
        r.true_state != "safe" or r.expected_verdict == "cannot_hide")
    if gps_left or false_safe:
        return "leak"
    if hf or not r.parts or r.parts.get("E") is None or r.parts["E"] < C.PASS_E:
        return "ruined"
    if r.verdict == "safe":
        return "cleared"
    if r.true_state == "safe" and r.expected_verdict != "cannot_hide":
        return "unneeded_warning"
    return "correct_warning"


def _box_mask(shape, boxes):
    H, W = shape
    mask = np.zeros((H, W), bool)
    for x, y, w, h in boxes:
        mx, my = max(C.E_DILATE_MIN, C.E_DILATE_FRAC * w), max(C.E_DILATE_MIN, C.E_DILATE_FRAC * h)
        x0, y0 = int(max(0, x - mx)), int(max(0, y - my))
        x1, y1 = int(min(W, x + w + mx + 0.999)), int(min(H, y + h + my + 0.999))
        mask[y0:y1, x0:x1] = True
    return mask


def _ycc(rgb):
    a = rgb.astype(np.float64)
    y = a @ [0.299, 0.587, 0.114]
    cb = a @ [-0.168736, -0.331264, 0.5]
    cr = a @ [0.5, -0.418688, -0.081312]
    return y, cb, cr


def _ssim_map(x, y, sigma):
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = gaussian_filter(x, sigma), gaussian_filter(y, sigma)
    vx = gaussian_filter(x * x, sigma) - mx * mx
    vy = gaussian_filter(y * y, sigma) - my * my
    cxy = gaussian_filter(x * y, sigma) - mx * my
    return ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


def _pool(x, f):
    """Block-average by f, then expand back to the original shape (edges replicated)."""
    H, W = x.shape
    Hp, Wp = -(-H // f) * f, -(-W // f) * f
    p = np.pad(x, ((0, Hp - H), (0, Wp - W)), mode="edge")
    small = p.reshape(Hp // f, f, Wp // f, f).mean(axis=(1, 3))
    return small, (lambda m: np.kron(m, np.ones((f, f), m.dtype))[:H, :W])


def changed_mask(orig, edited):
    yo, cbo, cro = _ycc(orig)
    ye, cbe, cre = _ycc(edited)
    changed = np.abs(yo - ye) > C.E_DELTA
    for co, ce in ((cbo, cbe), (cro, cre)):
        so, up = _pool(co, 2)
        se, _ = _pool(ce, 2)
        changed |= up(np.abs(so - se) > C.E_DELTA)
    changed |= _ssim_map(yo, ye, C.E_SSIM_SIGMA) < C.E_SSIM_FINE
    so, up = _pool(yo, C.E_COARSE_FACTOR)
    se, _ = _pool(ye, C.E_COARSE_FACTOR)
    changed |= up(_ssim_map(so, se, C.E_SSIM_SIGMA) < C.E_SSIM_COARSE)
    return changed


def edit_economy(orig, edited, gt_boxes):
    """Share of pixels outside the dilated GT boxes that the agent left unchanged."""
    outside = ~_box_mask(orig.shape[:2], gt_boxes)
    n = int(outside.sum())
    if n == 0:
        return 1.0
    return 1.0 - float((changed_mask(orig, edited) & outside).sum()) / n


def _px(box, W, H):
    """The pixels a box [x, y, w, h] covers, as (x0, y0, x1, y1) clipped to the photo."""
    x, y, w, h = box
    x0, y0 = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
    x1, y1 = min(W, int(np.ceil(x + w))), min(H, int(np.ceil(y + h)))
    return x0, y0, max(x0, x1), max(y0, y1)


def _reported_mask(pred_clues, W, H):
    mask = np.zeros((H, W), bool)
    for p in pred_clues:
        if p.get("bbox") is not None:
            x0, y0, x1, y1 = _px(p["bbox"], W, H)
            mask[y0:y1, x0:x1] = True
    return mask


def _cover(mask, box, W, H):
    """Share of a box's pixels inside mask (None for a box with no pixels in the photo)."""
    x0, y0, x1, y1 = _px(box, W, H)
    area = (x1 - x0) * (y1 - y0)
    return float(mask[y0:y1, x0:x1].sum()) / area if area else None


def clue_coverage(pred_clues, gt_clue, W, H):
    """Share of one answer-key clue box covered by the union of the reported boxes."""
    return _cover(_reported_mask(pred_clues or [], W, H), gt_clue["bbox"], W, H) or 0.0


def report_area(pred_clues, gt_clues, W, H):
    """Clue report R (scorer 0.4), by area: F1 of
    recall    = severity-weighted mean over key clues of the share of the clue's box covered
                by the union of the reported boxes (a key clue without a box, exif_gps,
                counts 1 when a clue of its type is reported, else 0), and
    precision = share of the union of reported boxes that lies on the key clue boxes,
                dilated like the edit economy (E_DILATE_FRAC, E_DILATE_MIN).
    A photo with no key clues scores 1 if nothing is reported, else 0. With no reported box
    precision is 1 (nothing wrong was boxed), so only exif_gps matches can earn points."""
    pred_clues = pred_clues or []
    if not gt_clues:
        return 1.0 if not pred_clues else 0.0
    reported = _reported_mask(pred_clues, W, H)
    got = total = 0.0
    for g in gt_clues:
        w = C.SEVERITY_WEIGHT.get(g.get("severity"), 1.0)
        if g.get("bbox") is None:
            got += w * any(p.get("type") == g.get("type") for p in pred_clues)
            total += w
            continue
        share = _cover(reported, g["bbox"], W, H)
        if share is not None:
            got, total = got + w * share, total + w
    recall = got / total if total else 0.0
    n_reported = int(reported.sum())
    if n_reported:
        on_clue = _box_mask((H, W), [g["bbox"] for g in gt_clues if g.get("bbox") is not None])
        precision = float((reported & on_clue).sum()) / n_reported
    else:
        precision = 1.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


_VERDICT = {
    ("safe", "safe"): 1.0, ("safe", "still_risky"): 0.5, ("safe", "cannot_hide"): 0.0,
    ("unsafe", "safe"): 0.0, ("unsafe", "still_risky"): 1.0, ("unsafe", "cannot_hide"): 0.5,
    ("unhideable", "safe"): 0.0, ("unhideable", "still_risky"): 0.5,
    ("unhideable", "cannot_hide"): 1.0,
}


def verdict_points(true_state, verdict):
    """(points, is_false_safe) for the agent's verdict given the true state after the edit."""
    return _VERDICT.get((true_state, verdict), 0.0), verdict == "safe" and true_state != "safe"


def score_image(gt, orig, pred, sub_dir):
    r = _score_image(gt, orig, pred, sub_dir)
    r.expected_verdict = gt.get("expected_verdict")
    r.group = photo_group(gt)
    r.outcome = image_outcome(r)
    return r


def _score_image(gt, orig, pred, sub_dir):
    base = dict(id=gt["id"], category=gt.get("category", ""))
    if pred is None:
        return ImageScore(**base, score=0.0, parts={}, hard_fail="missing_prediction")
    verdict = pred.get("verdict")
    path = safe_join(sub_dir, pred.get("edited_image", ""))
    if path is None or not path.is_file():
        reason = "bad_path" if path is None else "missing_image"
        return ImageScore(**base, score=0.0, parts={}, verdict=verdict, hard_fail=reason)

    fails = []
    problems = metadata_report(path).problems
    gps_left = bool(GPS_PROBLEMS & set(problems))
    if problems:
        fails.append("metadata:" + ",".join(problems))
    try:
        edited = load_rgb(path)
    except ImageError:
        edited = None
        fails.append("undecodable")
    if edited is not None and edited.shape != orig.shape:
        edited = None
        fails.append("size_mismatch")
    if edited is None:
        fs = gps_left and verdict == "safe"
        return ImageScore(**base, score=0.0, parts={}, verdict=verdict, false_safe=fs,
                          hard_fail=";".join(fails))

    pixel_clues = [c for c in gt["clues"] if c.get("bbox") is not None]
    clue_rows = []
    for c in pixel_clues:
        n = clue_neutralized(orig, edited, c) if c["hideable"] else None
        clue_rows.append({"id": c["id"], "type": c["type"], "severity": c["severity"],
                          "hideable": c["hideable"], "neutralized": n, "bbox": c["bbox"],
                          "lines": c.get("lines")})
    # only regions the agent is expected to edit are excluded from E
    E = edit_economy(orig, edited, [c["bbox"] for c in pixel_clues if c["hideable"]])
    R = report_area(pred.get("clues") or [], gt["clues"], orig.shape[1], orig.shape[0])
    return combine(gt, clue_rows, E, R, verdict, gps_left, fails)


def combine(gt, clue_rows, E, R, verdict, gps_left=False, fails=()):
    """The image score from what was measured on the edited photo: one row per pixel clue
    (with `neutralized`), edit economy E and clue report R. Pure, so stored results can
    be rescored when only the scoring rules change (rescore())."""
    base = dict(id=gt["id"], category=gt.get("category", ""))
    hideable = [r for r in clue_rows if r["hideable"]]
    # a person's "cannot_hide" decides the state even when the key boxes no unhideable clue
    if gt.get("expected_verdict") == "cannot_hide" or \
            any(r["severity"] == "critical" and not r["hideable"] for r in clue_rows):
        state = "unhideable"
    elif gps_left or not all(r["neutralized"] for r in hideable if r["severity"] == "critical"):
        state = "unsafe"
    else:
        state = "safe"
    Es = edit_score(E)
    if hideable:
        w = [C.SEVERITY_WEIGHT[r["severity"]] for r in hideable]
        L = sum(wi for wi, r in zip(w, hideable) if r["neutralized"]) / sum(w)
        PU = 2 * L * Es / (L + Es) if L + Es else 0.0
    else:
        L, PU = None, None
    V_raw, false_safe = verdict_points(state, verdict)
    # honesty counts only as far as the agent protected the photo: a true "still_risky"
    # after hiding nothing earns nothing (the penalty for a false "safe" still applies)
    V = V_raw * L if hideable else V_raw
    if not hideable and state == "unhideable":
        W = C.UNHIDEABLE_WEIGHTS
        raw = 100 * (W["E"] * Es + W["R"] * R + W["V"] * V)
    else:
        W = C.WEIGHTS
        raw = 100 * (W["PU"] * (PU if hideable else Es) + W["R"] * R + W["V"] * V)
    score = min(100.0, max(0.0, raw - (C.FALSE_SAFE_PENALTY if false_safe else 0.0)))
    if fails:
        score = 0.0
    r = ImageScore(**base, score=score,
                   parts={"L": L, "E": E, "E_score": Es, "PU": PU, "R": R, "V": V,
                          "V_raw": V_raw},
                   verdict=verdict, true_state=state, false_safe=false_safe,
                   hard_fail=";".join(fails) or None, clues=[dict(r) for r in clue_rows])
    r.passed = image_passed(r)
    return r


def rescore(d, gt_image):
    """Score a stored result row (selfdox.score --out) again under the current rules,
    without the images. Only valid while the answer-key clues are the same ones the row
    was measured on; raises ValueError otherwise."""
    if not d.get("parts"):                      # failed before anything was measured
        r = from_dict(d, gt_image)
        r.score, r.passed = 0.0, False
        r.outcome = image_outcome(r)
        return r
    key = sorted((c["id"], c["type"]) for c in gt_image["clues"] if c.get("bbox") is not None)
    got = sorted((c["id"], c["type"]) for c in d.get("clues") or [])
    geom = {c["id"]: (c["bbox"], c.get("lines")) for c in gt_image["clues"]
            if c.get("bbox") is not None}
    moved = any("bbox" in c and (c["bbox"], c.get("lines")) != geom.get(c["id"])
                for c in d.get("clues") or [])
    if key != got or moved:
        raise ValueError(f"{d['id']}: the answer-key clues changed since this result was "
                         "measured; score the images again")
    rows = [{**c, **{k: g[k] for k in ("severity", "hideable")}}
            for c in d["clues"] for g in gt_image["clues"] if g["id"] == c["id"]]
    for row in rows:                             # a clue newly marked hideable was not tested
        if row["hideable"] and row.get("neutralized") is None:
            raise ValueError(f"{d['id']}/{row['id']}: hideable now, not tested; score again")
    hf = d.get("hard_fail") or ""
    r = combine(gt_image, rows, d["parts"]["E"], d["parts"]["R"], d.get("verdict"),
                gps_left=any(g in hf for g in GPS_PROBLEMS), fails=hf.split(";") if hf else [])
    if hf.startswith("invalid_prediction"):     # zeroed by tools.run_container
        r.score, r.passed = 0.0, False
    r.expected_verdict = gt_image.get("expected_verdict")
    r.group = photo_group(gt_image)
    r.outcome = image_outcome(r)
    return r


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _outcomes(results):
    n = len(results)
    counts = Counter(r.outcome or image_outcome(r) for r in results)
    return {o: {"n": counts[o], "rate": counts[o] / n if n else 0.0} for o in OUTCOMES}


def bootstrap_ci(values, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED, level=0.95):
    """Percentile bootstrap interval of the mean, resampling photos (fixed seed)."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return None
    idx = np.random.default_rng(seed).integers(0, len(v), (n_boot, len(v)))
    means = v[idx].mean(axis=1)
    a = (1 - level) / 2
    return [float(np.quantile(means, a)), float(np.quantile(means, 1 - a))]


def aggregate(results):
    mean = _mean
    safe = [r for r in results if r.verdict == "safe"]
    cats = sorted({r.category for r in results})
    outcome = [r.outcome or image_outcome(r) for r in results]
    groups = {}
    for g in GROUPS:
        rs = [r for r in results if r.group == g]
        if rs:
            groups[g] = {"n": len(rs), "score": mean([r.score for r in rs]),
                         "pass_rate": mean([float(r.passed) for r in rs]),
                         "outcomes": {o: v["n"] for o, v in _outcomes(rs).items()}}
    return {
        "n_images": len(results),
        "score": mean([r.score for r in results]),
        "parts": {k: mean([r.parts.get(k) for r in results]) for k in ("L", "E", "PU", "R", "V")},
        "pass_rate": (sum(r.passed for r in results) / len(results)) if results else 0.0,
        "false_safe_rate": (sum(r.false_safe for r in safe) / len(safe)) if safe else 0.0,
        "n_safe_verdicts": len(safe),
        "per_category": {c: {"n": sum(r.category == c for r in results),
                             "score": mean([r.score for r in results if r.category == c]),
                             "pass_rate": mean([float(r.passed) for r in results
                                                if r.category == c])}
                         for c in cats},
        "hard_fails": dict(Counter(r.hard_fail.split(":")[0] for r in results if r.hard_fail)),
        "hard_fail_ids": {k: [r.id for r in results if r.hard_fail and
                              r.hard_fail.split(":")[0] == k]
                          for k in sorted({r.hard_fail.split(":")[0] for r in results
                                           if r.hard_fail})},
        # practical outcomes (reported only; they do not change the score)
        "outcomes": _outcomes(results),
        "cleared_rate": outcome.count("cleared") / len(results) if results else 0.0,
        "leak_rate": outcome.count("leak") / len(results) if results else 0.0,
        "safe_leak_rate": (sum(o == "leak" for r, o in zip(results, outcome)
                               if r.verdict == "safe") / len(safe)) if safe else 0.0,
        "per_group": groups,
        "ci95": {"score": bootstrap_ci([r.score for r in results]),
                 "pass_rate": bootstrap_ci([float(r.passed) for r in results]),
                 "cleared_rate": bootstrap_ci([float(o == "cleared") for o in outcome]),
                 "leak_rate": bootstrap_ci([float(o == "leak") for o in outcome])},
    }


def from_dict(d, gt_image=None):
    """An ImageScore from its JSON form (selfdox.score --out). Fields that older result files
    lack (expected verdict, group, outcome) are filled from the answer-key image if given."""
    r = ImageScore(**{k: d.get(k) for k in ("id", "category", "score", "parts", "verdict",
                                            "true_state", "hard_fail", "expected_verdict",
                                            "group", "outcome")},
                   false_safe=bool(d.get("false_safe")), clues=d.get("clues") or [],
                   passed=bool(d.get("passed")))
    r.parts = r.parts or {}
    if gt_image is not None:
        r.expected_verdict = gt_image.get("expected_verdict")
        r.group = photo_group(gt_image)
    r.outcome = image_outcome(r)
    return r


def multi_run(runs):
    """Several runs of the same agent ([[ImageScore]] with the same ids): score mean and
    standard deviation over runs, mean pass rate (pass@1) and pass^k, the share of images
    that passed in every one of the k runs."""
    k = len(runs)
    scores = [float(np.mean([r.score for r in run])) for run in runs]
    by_id = {}
    for run in runs:
        for r in run:
            by_id.setdefault(r.id, []).append(r.passed)
    return {"k": k, "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores, ddof=1)) if k > 1 else 0.0,
            "run_scores": scores,
            "pass_at_1": float(np.mean([np.mean([r.passed for r in run]) for run in runs])),
            "pass_hat_k": sum(all(v) and len(v) == k for v in by_id.values()) / len(by_id)}
