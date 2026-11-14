# Data Sources and Collection Pipeline

> Written for v0.2 (the 20-photo pilot). v1.0 has 300 Commons photos (dev 100, test 170, spare 30) and about 30 hidden phone photos; see `dataset_card.md`. The numbers below are the original plan.

Goal: about 150–200 photos that **look Korean** and fit the SelfDox story (street, shop, transit, food and landmark photos with readable location clues), each with GT clue boxes. The 20-image pilot is now part of dev (kr_0001–kr_0020).

## 1. Decision: Wikimedia Commons as the main pool

Verified on 2026-10-29 (scripts in `tools/`):

| Check | Result |
|---|---|
| Access | Public API, no key or account (`tools/commons.py`) |
| Volume | 16,162 geotagged files on a 3 km grid over Seoul and Busan (lower bound: many grid points hit the 500-result cap). The pilot crawl of 14 neighborhoods + 8 landmark categories kept 2,223 candidates from 250 authors |
| Filters | In a 1,030-file sample: 99.5% open license, 95% short side ≥ 1000 px, 70% taken 2016 or later |
| License | All CC BY / CC BY-SA / CC0 / public domain. Redistribution and modification allowed (BY-SA: edited outputs share the license) |
| GPS truth | 18 of 18 downloaded originals carried real phone/camera EXIF GPS, identical to the Commons coordinates (0 m difference) |
| Clues | In 48 sampled photos, about half had readable location clues (shop signs with phone numbers, road-name plates, station names, bus routes); food and interior shots serve as `no_clue` |

### Rejected or secondary sources

| Source | Finding | Use |
|---|---|---|
| Flickr API | Non-commercial key is free (no Pro needed), but needs an account; ~30% of Commons candidates are already Flickr imports | Not used |
| Openverse | Anonymous API throttles after 2 requests, max 240 results, no GPS | Not used |
| OSV-5M | CC BY-SA 4.0, open; only 616 Korean images in test, mostly rural dash-cam views | Not used |
| KoreaGEO Bench | On HF as `wangxiaonan-hf/KoreaGEO`, gated (manual approval), license "other" | Ask the authors; taxonomy reference |
| VLM-GeoPrivacyBench | 10 of 1,200 images in Korea; CC BY-NC | Not used |
| DoxBench | Gated; California | Category ideas only |
| AI Hub | Can't be redistributed to non-Korean teams | Not used |

## 2. Pipeline

```
crawl_commons → contact_sheet (pick) → bake → preannotate (OCR) → annotate → build_release → render_gt (review)
```

| Step | Tool | What it does |
|---|---|---|
| Crawl | `tools/crawl_commons.py` | Geosearch (1 km around 14 neighborhoods) + landmark categories → `data/raw/candidates.json`. Keeps open licenses, JPEG/PNG, short side ≥ 1000 px, taken 2016+ |
| Pick | `tools/contact_sheet.py` | Numbered contact sheets with an author cap and pHash dedupe; picks go to `data/annotations/selection.json` with category and `gps_in_input` |
| Bake | `tools/bake.py` | Download original → apply EXIF orientation → sRGB → long side 2048 → blur faces (OpenCV YuNet) and manual privacy boxes (`data/annotations/privacy.json`) → save with **no metadata**. Records the original GPS |
| Pre-annotate | `tools/preannotate.py` | Apple Vision OCR (via `ocrmac`, macOS) → numbered text lines in `data/annotations/ocr/<id>.json` and overlays. A different OCR from agent D's EasyOCR, so the GT is not biased toward D |
| Annotate | by hand | `data/annotations/pilot.json`: clues refer to OCR line numbers or manual boxes; see [`annotation_guide.md`](annotation_guide.md) |
| Release | `tools/build_release.py` | Writes `data/dev/images`, re-inserts GPS (piexif) only where `gps_in_input`, adds the `exif_gps` clue, validates `gt.json`, writes `ATTRIBUTION.md` |
| Review | `tools/render_gt.py` | Overlays in `data/dev/review/` (gitignored) for the second-person check |

### Pilot quotas (20) and the target (~160)

| Category | Pilot | Target | Source |
|---|---|---|---|
| `easy` | 4 | 35 | street / station photos |
| `medium` | 5 | 40 | street photos with plates, buses, small signs |
| `hard` | 3 | 30 | views, reflections, distant landmarks |
| `no_clue` | 3 | 25 | food / interior photos |
| `cannot_hide` | 2 | 15 | landmark categories |
| `exif_only` | 3 | 15 | food photos with their real GPS kept |

## 3. Rules

- No Instagram or Naver crawling (terms of service and privacy).
- Keep title, author, license and page for every image (`ATTRIBUTION.md`).
- Faces and license plates are blurred before release; event, protest and portrait photos are skipped.
- At most 3 images per author in the pilot (raise to ~5 at full scale) to avoid one photographer's style dominating.
- Prefer 2025–2026 photos for the hidden test split: older Commons photos may be in VLM training data.
- Self-shot photos (for selfie / "sharenting" scenes, which Commons lacks): no real faces, nothing near our dorm or campus.
