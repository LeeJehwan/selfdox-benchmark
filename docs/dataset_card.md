# SelfDox dataset card

> **Draft.** These numbers come from the draft answer key (Claude's first pass over model drafts, before the human review). Rerun `tools.dataset_card` on the final key.

## What it is

Street, shop, transit and food photos from Seoul and Busan with an answer key of every
**location clue** in them: what would let a stranger find where the photo was taken. It is
the evaluation data of the SelfDox benchmark: an agent must hide those clues, keep the
rest of the photo, and say honestly whether the photo is now safe. Scoring: `design.md`.

## Sources and licenses

- **Commons photos:** Wikimedia Commons, found by geosearch around 47 places in Seoul and
  Busan and by landmark categories (`data_sources.md`). Only open licenses (CC BY, CC
  BY-SA, CC0, public domain). Authors and links per photo in `data/*/ATTRIBUTION.md`.
- **Hidden photos:** about 30 phone photos taken by our team, never published; only
  the organizers run agents on them (`participants.md`).
- **Changes to every photo:** EXIF orientation applied, converted to sRGB, long side
  2048 px, all metadata removed, faces (YuNet detector) and license plates blurred. GPS
  is written back into 22% of the photos, because removing it is part of the task.

## How it was made

1. **Candidates:** about 6,600 Commons photos crawled; 991 kept for labelling after
   dropping conferences, ceremonies, protests, concerts, car-model shots and interiors
   by title and category, newest first, at most 5 per author.
2. **Model drafts:** Qwen3-VL-8B on the lab cluster boxed every clue it saw (type,
   text, severity, hideable). Repetition loops were retried; drafts were cleaned of
   duplicate and degenerate boxes and of clues the model itself called generic.
3. **Selection:** 280 photos by category quota (at most 3 per author) plus the 20
   hand-picked pilot photos.
4. **OCR lines:** Apple Vision read every photo; lines inside a text clue become the units
   the scorer tests. For clues the full-photo pass missed, a second pass on an upscaled
   crop recovered lines for 190 of 360.
5. **Claude first pass:** seven Claude reviewers went through the 280 Commons photos on
   review sheets (numbered boxes, zoomed crops, OCR candidates) with the criteria in
   `prereview_instructions.md`: keep or drop each draft clue, fix type, severity and
   hideable, flag bad boxes, describe clues nobody boxed, and suggest category and
   expected verdict. 20 photos were checked by hand.
6. **Boxes for described clues:** the first pass described 438 clues it saw but nobody had
   boxed. Qwen3-VL-8B boxed 437 of them on the lab cluster (`cluster/ground_missed.py`).
   415 entered the draft key. 19 repeated a box already there, and 3 were large boxes
   around a group of signs that a reviewer has to split. A landmark, skyline or facade
   box covering a quarter of the photo or more is marked not hideable.
7. **Human review:** on a private review page (one record per photo, Claude's suggestion
   filled in and kept for comparison), a person accepts, fixes or adds every clue and
   marks the photo done or excluded.
8. **Answer key and split:** `tools.build_gt` keeps the accepted clues; `tools.split`
   splits by author (one author never spans two splits), stratified by category and
   GPS, with 2025-26 photos sent to test first.

## Numbers

|  |  |
|---|---|
| photos | 300 |
| clues | 1059 |
| clues per photo (mean) | 3.5 |
| text lines tested by the scorer | 715 |
| text clues with line boxes | 415 / 678 |
| photos with GPS in the input | 66 (22%) |
| expected verdict safe / cannot_hide | 215 / 85 |
| clues checked by a person | 0 (0%) |
| licenses | CC BY-SA 4.0 162, CC BY 4.0 36, CC0 33, CC BY-SA 2.0 17, CC BY 2.0 17, CC BY-SA 3.0 12, Public domain 9, CC BY 3.0 8, CC BY 2.0 kr 4, CC BY-SA 2.0 kr 2 |

### Categories

| category | photos | share | dev | test | spare |
|---|---|---|---|---|---|
| easy | 63 | 21% | 21 | 36 | 6 |
| medium | 51 | 17% | 17 | 29 | 5 |
| hard | 44 | 15% | 14 | 26 | 4 |
| no_clue | 28 | 9% | 10 | 17 | 1 |
| cannot_hide | 84 | 28% | 28 | 46 | 10 |
| exif_only | 30 | 10% | 10 | 16 | 4 |

### Clue types

| clue type | clues | critical | not hideable |
|---|---|---|---|
| shop_sign | 313 | 161 | 0 |
| address_plate | 24 | 9 | 0 |
| apt_name | 10 | 4 | 0 |
| apt_block | 8 | 0 | 0 |
| street_sign | 36 | 20 | 0 |
| transit | 78 | 36 | 0 |
| phone_number | 39 | 25 | 0 |
| utility_mark | 6 | 1 | 0 |
| school_name | 13 | 12 | 0 |
| landmark | 206 | 157 | 102 |
| skyline_view | 62 | 12 | 15 |
| facade | 47 | 2 | 23 |
| reflection | 5 | 2 | 1 |
| exif_gps | 66 | 66 | 0 |
| other | 146 | 36 | 0 |

### Answer-key warnings

Checks from `selfdox.schema.gt_warnings` (also printed by `tools.build_gt`). They are not format errors, but each one changes scores until it is fixed.

| check | photos | examples |
|---|---|---|
| expected verdict cannot_hide, but no boxed clue is critical and not hideable: the key should box what can't be hidden (scorer 0.3 follows the expected verdict, but an agent that reports the landmark gets no R credit for it) | 0 |  |
| a clue category, but no clue has a box, so the photo scores like a no-clue photo | 0 |  |
| expected verdict safe, but a boxed clue is critical and not hideable: either the place is the subject (expected cannot_hide) or the clue can be hidden | 0 |  |
| clues not checked by a person yet | 1059 / 1059 |  |

## Claude suggestion vs. human review

Not available yet: the human review has not finished. The page keeps Claude's first-pass suggestion next to each final answer.

## Known limits

- **Commons bias:** photos are what Commons contributors upload: landmarks, tourist
  streets and events are over-represented; ordinary residential streets are rare. Only
  Seoul and Busan.
- **Missed clues:** the key lists what the model, OCR, Claude and a reviewer saw. A clue
  all of them missed costs an agent that finds it a little precision in R and nothing
  else. Disputes on the dev key go into the changelog.
- **Draft bias:** reviewers start from Qwen3-VL-8B drafts and Claude's suggestions, so
  clues those models handle badly (tiny text, vertical signs, partly hidden plates) are
  the most likely to be missed or mis-boxed.
- **Judgement calls:** whether a landmark is the subject (`cannot_hide`) or a hideable
  part of the scene, and whether a chain store counts as critical, are judgement calls;
  the review page shows Claude's reason for each so reviewers can be consistent.
- **Public photos:** Commons photos may be in a model's training data. The task is to
  hide clues rather than guess the place, and test favours 2025-26 photos, but a model
  can still recognise famous places; the hidden phone photos are the control.
