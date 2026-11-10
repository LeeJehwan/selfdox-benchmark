# Annotation Guide

How to add or review GT clues. The pilot GT was pre-annotated by Claude (`"review": "draft"`); every clue needs a second-person check before v1.0.

## Workflow

1. Run `uv run python -m tools.preannotate` on new baked images. Open `data/raw/review/<id>_ocr.jpg` (numbered OCR lines) next to `data/annotations/ocr/<id>.json`.
2. Add the image to `data/annotations/pilot.json`:
   ```json
   "kr_0001": {"expected_verdict": "safe", "clues": [
     {"type": "shop_sign", "lines": [1], "text": "거인통닭", "severity": "critical", "hideable": true},
     {"type": "address_plate", "bbox": [1564, 651, 37, 38], "text": "34",
      "severity": "critical", "hideable": true, "note": "manual box"}]}
   ```
   `lines` items are OCR line numbers or manual `[x, y, w, h]` boxes in baked-image pixels. The clue `bbox` defaults to the union of its lines.
3. Run `uv run python -m tools.build_release` (validates everything) and `uv run python -m tools.render_gt`, then check `data/dev/review/<id>.jpg`.
4. When you have checked a clue, set `"review": "checked"` (add the field in `pilot.json`; `build_release` keeps it).

## What is a clue

A clue is anything in the pixels (or metadata) that helps a stranger find **where the photo was taken**.

| Mark it | Don't mark it |
|---|---|
| Shop names, especially with a branch, phone number or neighborhood | Generic words: "Burger", "상품권", "MONEY EXCHANGE", opening hours, menus |
| Road-name plates, building numbers, apartment / villa names, 동 numbers | Brand logos on products, car makes |
| Station names, bus route numbers, stop names, exit numbers | License plates (blurred for privacy instead) |
| Landmarks, distinctive skylines and facades | Faces (blurred for privacy instead) |
| Banners naming a district ("LET'S GO MYEONG DONG") | Text too blurry to read even in the original (mark it as a non-legible supporting clue only if the shape still helps) |

## Fields

- **type**: see the taxonomy in [`design.md`](design.md#5-clue-taxonomy).
- **severity**: `critical` if this clue *alone* narrows the location to about 1 km (shop name + phone, road-name plate, station name, bus route + destination). Otherwise `supporting`.
- **hideable**: `false` only if hiding it would destroy the photo (the landmark or skyline *is* the subject). A critical unhideable clue makes the expected verdict `cannot_hide`.
- **lines**: one box per text line, tight around the glyphs. Vertical signs get one tall box per column. The scorer tests each line, so a sign with a name and a phone number needs both lines.
- **text**: exactly what is legible, as written (Korean kept). Leave it out if you can't read it.
- **expected_verdict**: the right answer if the agent hides every hideable clue.

## Box tips

- OCR line boxes are usually good. Widen them if they cut glyphs; split them if they merge two signs.
- Keep boxes off unrelated structure where possible (sign borders, brick); it makes the neutralization test stricter than needed.
- Manual boxes: crop the region at 4× first; don't estimate from a downscaled overlay.
