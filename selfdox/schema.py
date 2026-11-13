"""Loading and checking the GT file and agent submissions.

The JSON Schemas in selfdox/schemas/ are the published format; this module adds the
checks a schema can't express (ids, boxes inside the image, files on disk).
"""
import json
from importlib.resources import files
from pathlib import Path

from jsonschema import Draft202012Validator
from PIL import Image

from . import config as C
from .io import ImageError, load_rgb, metadata_report, safe_join

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _validator(name):
    return Draft202012Validator(json.loads(files("selfdox").joinpath("schemas", name).read_text()))


def _schema_errors(validator, doc):
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
            for e in sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path))]


def _inside(box, W, H):
    x, y, w, h = box
    return w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= W and y + h <= H


def gt_errors(doc):
    errs = _schema_errors(_validator("gt.schema.json"), doc)
    if errs:
        return errs
    seen = set()
    for im in doc["images"]:
        iid, W, H = im["id"], im["width"], im["height"]
        if iid in seen:
            errs.append(f"{iid}: duplicate id")
        seen.add(iid)
        clue_ids = set()
        for c in im["clues"]:
            where = f"{iid}/{c['id']}"
            if c["id"] in clue_ids:
                errs.append(f"{where}: duplicate clue id")
            clue_ids.add(c["id"])
            if c["type"] == "exif_gps":
                if c["bbox"] is not None:
                    errs.append(f"{where}: exif_gps clue must have bbox null")
                continue
            if c["bbox"] is None:
                errs.append(f"{where}: bbox is required (only exif_gps may be null)")
                continue
            if not _inside(c["bbox"], W, H):
                errs.append(f"{where}: bbox {c['bbox']} is outside the {W}x{H} image")
            for ln in c.get("lines", []):
                if not _inside(ln, W, H):
                    errs.append(f"{where}: line {ln} is outside the image")
                elif min(ln[2], ln[3]) < C.MIN_LINE_PX:
                    errs.append(f"{where}: line {ln} is smaller than {C.MIN_LINE_PX} px")
        for b in im.get("privacy_regions", []):
            if not _inside(b, W, H):
                errs.append(f"{iid}: privacy region {b} is outside the image")
    return errs


CLUE_CATEGORIES = {"easy", "medium", "hard", "cannot_hide"}
GT_WARNINGS = {
    "cannot_hide_no_stuck_clue":
        "expected verdict cannot_hide, but no boxed clue is critical and not hideable: the "
        "key should box what can't be hidden (scorer 0.3 follows the expected verdict, but "
        "an agent that reports the landmark gets no R credit for it)",
    "clue_category_no_box":
        "a clue category, but no clue has a box, so the photo scores like a no-clue photo",
    "safe_with_stuck_clue":
        "expected verdict safe, but a boxed clue is critical and not hideable: either the "
        "place is the subject (expected cannot_hide) or the clue can be hidden",
}


def image_warnings(im):
    """Codes from GT_WARNINGS for one answer-key image (valid GT, but likely a mistake)."""
    px = [c for c in im["clues"] if c.get("bbox") is not None]
    out = []
    if im.get("expected_verdict") == "cannot_hide" and \
            not any(c["severity"] == "critical" and not c["hideable"] for c in px):
        out.append("cannot_hide_no_stuck_clue")
    if im.get("category") in CLUE_CATEGORIES and not px:
        out.append("clue_category_no_box")
    if im.get("expected_verdict") == "safe" and \
            any(c["severity"] == "critical" and not c["hideable"] for c in px):
        out.append("safe_with_stuck_clue")
    return out


def gt_warnings(doc):
    """Answer-key checks that are not errors: {"images": {code: [ids]}, "unchecked_clues",
    "clues"}. Clues count as checked when a person reviewed them (review "checked")."""
    images = {code: [] for code in GT_WARNINGS}
    for im in doc["images"]:
        for code in image_warnings(im):
            images[code].append(im["id"])
    clues = [c for im in doc["images"] for c in im["clues"]]
    return {"images": images, "clues": len(clues),
            "unchecked_clues": sum(c.get("review") != "checked" for c in clues)}


def format_gt_warnings(w):
    """Printable lines for gt_warnings()."""
    lines = []
    for code, ids in w["images"].items():
        if ids:
            more = " ..." if len(ids) > 12 else ""
            lines.append(f"{len(ids)} photo(s): {GT_WARNINGS[code]}: {' '.join(ids[:12])}{more}")
    if w["unchecked_clues"]:
        lines.append(f"{w['unchecked_clues']} of {w['clues']} clues not checked by a person yet")
    return lines


def load_gt(path):
    doc = json.loads(Path(path).read_text())
    errs = gt_errors(doc)
    if errs:
        raise ValueError("invalid GT:\n  " + "\n  ".join(errs))
    return doc


def input_sizes(images_dir):
    """{id: (width, height)} for the benchmark input images in a folder."""
    sizes = {}
    for p in sorted(Path(images_dir).iterdir()):
        if p.suffix.lower() in IMAGE_SUFFIXES:
            with Image.open(p) as im:
                sizes[p.stem] = im.size
    return sizes


def gt_sizes(doc):
    return {im["id"]: (im["width"], im["height"]) for im in doc["images"]}


def submission_format_errors(sub_dir):
    """(doc, errors) for predictions.json: file present, valid JSON, matches the schema."""
    p = Path(sub_dir) / "predictions.json"
    if not p.is_file():
        return None, [f"{p} not found"]
    try:
        doc = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        return None, [f"predictions.json is not valid JSON: {e}"]
    return doc, _schema_errors(_validator("submission.schema.json"), doc)


def submission_problems(sub_dir, sizes):
    """(run errors, {image id: [problems]}, warnings) for outputs/predictions.json and the
    edited images it points to. Run errors (no file, bad JSON, wrong schema) stop the whole
    submission; an image problem concerns one prediction only."""
    sub_dir = Path(sub_dir)
    doc, errors = submission_format_errors(sub_dir)
    if errors:
        return errors, {}, []
    warnings, seen, problems = [], set(), {}

    def bad(k, iid, msg):
        problems.setdefault(iid, []).append(f"predictions[{k}] ({iid}): {msg}")

    if doc["benchmark_version"] != C.BENCHMARK_VERSION:
        warnings.append(f"benchmark_version {doc['benchmark_version']!r} != "
                        f"{C.BENCHMARK_VERSION!r}")
    for k, pr in enumerate(doc["predictions"]):
        iid = pr["id"]
        if iid in seen:
            bad(k, iid, "duplicate id")
            continue
        seen.add(iid)
        if iid not in sizes:
            bad(k, iid, "unknown id")
            continue
        W, H = sizes[iid]
        for c in pr["clues"]:
            if c["bbox"] is not None and not _inside(c["bbox"], W, H):
                bad(k, iid, f"clue bbox {c['bbox']} is outside the {W}x{H} image")
        path = safe_join(sub_dir, pr["edited_image"])
        if path is None:
            bad(k, iid, "edited_image path must stay inside the submission folder")
            continue
        if not path.is_file():
            bad(k, iid, f"edited_image not found: {pr['edited_image']}")
            continue
        for code in metadata_report(path).problems:
            bad(k, iid, f"metadata problem: {code} (hard fail)")
        try:
            arr = load_rgb(path)
        except ImageError as e:
            bad(k, iid, str(e))
            continue
        if (arr.shape[1], arr.shape[0]) != (W, H):
            bad(k, iid, f"size {arr.shape[1]}x{arr.shape[0]} != input {W}x{H}")
    missing = sorted(set(sizes) - seen)
    if missing:
        more = " ..." if len(missing) > 10 else ""
        warnings.append(f"no prediction for {len(missing)} image(s), scored 0: "
                        f"{', '.join(missing[:10])}{more}")
    return [], problems, warnings


def submission_errors(sub_dir, sizes):
    """(errors, warnings) for outputs/predictions.json plus the edited images it points to."""
    errors, problems, warnings = submission_problems(sub_dir, sizes)
    return errors + [m for ms in problems.values() for m in ms], warnings
