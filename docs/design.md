# SelfDox Benchmark Design (v1.0 draft)

Status: **v1.0 draft**. The scorer (`selfdox/`), the full data pipeline and the split tools are in place; the 300-photo answer key is in human review (Claude's first pass is filled in). Text thresholds come from an offline OCR calibration and the non-text threshold from a VLM judge ([`calibration.md`](calibration.md)). For participants: [`participants.md`](participants.md). For the data: [`dataset_card.md`](dataset_card.md).

## 1. Changes

### Scorer 0.3 (2026-11-10)

Changes scores; every stored result was rescored (`tools/rescore.py`, checked against scoring the images again on all 7 pilot agents: identical per image). Why: the maintainers' practicality review of 2026-11-10 (`docs/review_practical.md` in the maintainers' repository).

| Rule | 0.2 | 0.3 | Effect on the 300-photo draft key (E / oracle / B) |
|---|---|---|---|
| True state | `unhideable` only if the key boxes a critical unhideable clue | also when a person set the expected verdict to `cannot_hide` | 54 photos where an honest `cannot_hide` got 0 and a false `safe` full marks |
| Edit score in PU | E | `E_score = max(0, 1 − 3·(1 − E))`: over-editing costs three times as much (the reported E and the pass threshold stay on raw E) | blurring 14% of the photo on average cost E only about 5.5 points |
| Pass | no false `safe` | the **right** verdict for the true state (`safe` → `safe`, `unhideable` → `cannot_hide`) | always answering `still_risky` used to *raise* the pass rate |
| Score | | | 75.1 / 93.9 / 32.7 → **63.2 / 97.0 / 29.4**; pass rate 0.613 / 0.960 / 0.417 → **0.260 / 0.960 / 0.350** |
| Draft key | 644 clues | **1,059**: 415 clues Claude's first pass described without a box, boxed by Qwen3-VL-8B (`cluster/ground_missed.py`; still to be checked by a person) | with this key: E **61.6**, oracle **96.4**, B **21.8**; pass rate 0.257 / 0.953 / 0.300 |

### Since v0.2

| Topic | v0.2 | v1.0 | Why |
|---|---|---|---|
| Data | 20 pilot photos | **300 photos + ~30 hidden phone photos**, split dev 100 / test 170 / hidden 30 (§10) | Enough photos per category; a hidden set nobody can tune on |
| Answer key | Claude drafts | Qwen3-VL-8B drafts → OCR lines → Claude first pass → **human review** on a private page | Every clue checked by a person (`dataset_card.md`) |
| Reported numbers | score, false-safe rate | + **pass rate** and **pass^k** (§7.5) | A strict per-photo success measure, and how reliably an agent succeeds across runs |
| Practical outcomes | — | + per-photo **outcome** (cleared, leak, unneeded/correct warning, ruined), results per photo group, bootstrap 95% intervals (§7.5); none of it changes the score | The score alone hides how often an agent gives the user a photo they can post, and how often its `safe` is wrong (practicality review, 2026-11-10) |
| Container runs | one bad image → the whole run unscored | only that image scores 0; image digest, GPU and peak GPU memory recorded; `--repeat k` (§10) | A single leftover GPS tag must not wipe out a hidden-set run |
| Non-text threshold | `TAU_NONTEXT` 0.5, uncalibrated | fit against a Qwen3-VL-8B judge on 2,160 edited landmark/skyline/facade regions (§9.3) | OCR can't judge whether a building is still recognizable |
| Reference agent E | planned | `agents/e_qwen.py`: Qwen3-VL-4B under the team rules (§8) | Sets the bar Phase 3 teams should beat |
| Hidden set | open question | organizers run each team's agent (`tools/run_container.py`: a venv on the lab cluster, or Docker/Apptainer; §10) on hidden **and test** | Participants never see the hidden photos, and test predictions are checked |

### Since v0

| Topic | v0 draft | v0.2 | Why |
|---|---|---|---|
| Reader in the scorer | Qwen3-VL reads the edited crop | **No model at scoring time.** A pixel-structure test decides whether each GT line is still recoverable (§7.1) | Deterministic, CPU-only, free. Phase 3 teams run the exact official scorer. No hidden prompt to overfit to. |
| Calibration | — | OCR (Apple Vision, as is / CLAHE / inverted) labels "still legible" on synthetic edits **offline**; thresholds are fit to it | Keeps an attacker-like reader in the loop without needing it at scoring time |
| Geo attack G | 10% of the score | Removed from the score; optional report on a lab server later | Small VLMs are weak at GPS; GT clues already capture what leaks |
| L and E | L 40% + E·L 20% | **PU = harmonic mean of L and E**, 65% | With E·L gating, "blur everything" still scored 65 (L = 1 for free). The harmonic mean gives 0 to both "blur all" and "do nothing" |
| E | per-pixel diff > 24 | diff on luma + 2×2 chroma, **or** local SSIM drop at two scales | A pixel diff misses blur over low-contrast areas (sky, walls): blur-all got E = 0.69 |
| E mask | all GT boxes | only **hideable** GT boxes | Editing an unhideable landmark should cost E |
| Data | Flickr API | Wikimedia Commons geosearch (see [`data_sources.md`](data_sources.md)) | Open licenses, no API key, real phone EXIF GPS |

## 2. Task

The agent receives **one photo** (JPEG, may contain GPS in EXIF) and returns:

1. **clues**: location clues it found, each with `bbox`, `type`, optional `text`, `reason`
2. **edited_image**: the photo with those clues hidden, **same size**, **no GPS metadata**
3. **verdict**: `safe` (nothing left that points to the location), `still_risky` (edited what it could, risk remains), or `cannot_hide` (the location is given away by something that can't be hidden without destroying the photo)

Any hiding method is allowed (blur, pixelate, fill, inpaint, fake text). The scorer only checks the result.

### 2.1 Rules for agents

> Status: proposal, pending team agreement. The model choice and size are open for discussion.

| Rule | Detail |
|---|---|
| One model | The only LLM/VLM an agent may call is [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) (4.4B, Apache-2.0), revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| Quantized builds | The official builds of the same model are allowed: [`-GGUF`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF) (`1cd86afb9a95c410a6038ab3b40d8b578c892266`) and [`-FP8`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-FP8) (`fefbb44cbcce8d1bb7e20b920b94f77432b3446d`). Full precision needs about 9 GB of GPU memory (fits a free Colab T4); 4-bit GGUF about 3 GB |
| No training | Weights stay as published: no fine-tuning, LoRA or distillation. Prompts, tool use and the agent loop are the team's to design |
| Other tools | Classical image tools and small non-generative models (OCR, face or text detectors, blur/inpainting filters) are allowed. No other LLM or VLM, no web search or online APIs |
| Offline | The agent must run without network access (the Docker re-run in the verify step has `network_mode: none`) |
| Disclosure | Each submission lists the tools and the model build it used |

Why one small model: it keeps the field fair and cheap to run, and memorized web photos give little advantage because the task is to hide given clues, not to guess the location.

#### Proposed additions (2026-11-10 review; not agreed yet)

The rules above leave gaps a team could use (details and numbers in the maintainers' practicality review of 2026-11-10). They also contradict each other: §2 allows any hiding method, inpainting included, while "small non-generative models" rules out learned inpainting. Proposal:

| Gap | Proposal |
|---|---|
| Test predictions are made by the teams and never checked | **Decided (2026-11-10):** organizers re-run every team's agent folder on test too, in a venv on the lab cluster (`run_container.py --runtime venv`, §10); the handed-in test predictions are a format check |
| No call, token or time budget on test; hidden allows 180 s per photo (about 40× agent E) | One budget for test and hidden, e.g. a mean of 120 s per photo; calls and tokens per photo reported (`meta` in `predictions.json`) |
| "Small non-generative model" is undefined | Every model besides Qwen3-VL-4B at most 1B parameters, and an allow-list published by a fixed date (below) |
| Weights can't be checked | Images handed in by digest (`name@sha256:…`), not tag; the official weights mounted by the organizers at `/models`; every file over 50 MB in the image must match a hash on the allow-list |
| Which run counts | One run on test and one on hidden; that run counts. Teams state temperature and seed. Greedy decoding is not fully repeatable under vLLM batching, and that variation is part of the agent's result |
| People in the loop | No hand edits of outputs; no training or tuning of any model on benchmark photos (dev included) |

| Tool | Proposed | Note |
|---|---|---|
| Classical image tools (blur, pixelate, fill, OpenCV inpainting) | allowed | no model |
| Learned inpainting, LaMa class (non-diffusion, ≤ 1B) | allowed | only fills regions the agent chose |
| Diffusion or other generative inpainting | not allowed | a second generative model |
| OCR and text detectors (Apple Vision, EasyOCR, PaddleOCR, Tesseract) | allowed | |
| VLM-based OCR (Florence-2, GOT-OCR, PaddleOCR-VL …) | not allowed | another VLM |
| Prompted detectors (Grounding DINO, OWL-ViT, YOLO-World) | allowed, ≤ 1B | boxes only |
| CLIP-type embeddings | allowed, ≤ 1B | open: also for "is this a landmark?" |
| Geolocation models (GeoCLIP, StreetCLIP …) | not allowed | they do the attacker's task |
| Offline place-name data (public gazetteers, road-name address lists) | allowed if public and disclosed | open: size limit |

## 3. Submission format

```
outputs/
├── predictions.json
└── images/<id>.png|jpg
```

```json
{
  "agent_name": "team7_agent",
  "benchmark_version": "0.3",
  "predictions": [
    {"id": "kr_0001",
     "clues": [{"bbox": [610, 170, 920, 165], "type": "shop_sign", "text": "거인통닭",
                "reason": "shop name readable"},
               {"bbox": null, "type": "exif_gps", "reason": "GPS in EXIF"}],
     "edited_image": "images/kr_0001.png",
     "verdict": "safe"}
  ]
}
```

Schema: [`selfdox/schemas/submission.schema.json`](../selfdox/schemas/submission.schema.json). Check with `python -m selfdox.validate --sub outputs --inputs data/dev/images`.

| Field | Rule |
|---|---|
| `id` | An input image id. Missing ids score 0; unknown or duplicate ids are errors. |
| `clues[].bbox` | `[x, y, w, h]` in input pixels, inside the image. `null` only for `exif_gps`. |
| `clues[].type` | Free string; the taxonomy in §5 is recommended. Only `exif_gps` is matched by type. |
| `edited_image` | Relative path inside the submission folder (no `..`). JPEG or PNG only. Same width and height as the input. |
| `verdict` | `safe` \| `still_risky` \| `cannot_hide` |

## 4. Ground-truth format

Schema: [`selfdox/schemas/gt.schema.json`](../selfdox/schemas/gt.schema.json). One record per image in `data/dev/gt.json`:

```json
{"id": "kr_0001", "file": "images/kr_0001.jpg", "width": 2048, "height": 1365,
 "lat": 35.102423, "lon": 129.026172, "category": "easy", "expected_verdict": "safe",
 "gps_in_input": true, "privacy_regions": [[736, 677, 85, 99], [315, 686, 62, 69]],
 "source": {"dataset": "wikimedia_commons", "title": "File:…", "page": "https://commons.wikimedia.org/…",
            "url": "…", "license": "CC BY-SA 2.0", "author": "…"},
 "clues": [
  {"id": "c1", "type": "shop_sign", "bbox": [608, 164, 932, 180], "lines": [[623, 179, 902, 150]],
   "text": "거인통닭", "severity": "critical", "hideable": true, "review": "draft"},
  {"id": "c8", "type": "exif_gps", "bbox": null, "severity": "critical", "hideable": true}]}
```

| Field | Meaning |
|---|---|
| `lines` | Text line boxes inside the clue (from OCR or drawn by hand). The neutralization test runs per line. A clue without `lines` is tested on its `bbox`. Lines are at least `MIN_LINE_PX` (8) tall. |
| `severity` | `critical`: alone narrows the location to about 1 km (shop name, phone number, address plate, station name). `supporting`: helps only in combination. |
| `hideable` | `false` if hiding it would destroy the photo (the landmark or skyline is the subject). |
| `privacy_regions` | Faces and license plates blurred when the image was prepared. Not clues. |
| `review` | `draft` (pre-annotated) or `checked` (verified by a person on the review page). |

Where the keys live: `data/dev/gt.json` is public; the test and hidden keys
(`data/private/test_gt.json`, `data/private/hidden_gt.json`) stay in the private repository
and are never packaged (`tools/package_release.py` refuses them).

## 5. Clue taxonomy

| Type | Examples | Usually |
|---|---|---|
| `shop_sign` | Store sign (간판), sign with a branch or neighborhood name | critical |
| `address_plate` | Road-name address plate (도로명주소판), building number | critical |
| `apt_name` / `apt_block` | Apartment or villa name / "102동" | critical / supporting |
| `street_sign` | Road name or direction sign, district banner | critical |
| `transit` | Bus route number, stop or station name, exit number | critical |
| `phone_number` | Landline on a sign (searchable to a business) | critical |
| `utility_mark` | Utility box or pole tag | supporting |
| `parcel_label`, `school_name` | Delivery label, school or daycare name | critical |
| `landmark`, `skyline_view`, `facade` | Palace, tower, distinctive building or view | often unhideable |
| `reflection` | Clue seen in a window or mirror | either |
| `exif_gps` | GPS in the file metadata (`bbox: null`) | critical |
| `other` | Anything else, e.g. a name plaque | — |

License plates are **not** location clues in Korea (no region on plates since 2006); they are pre-blurred for privacy only.

## 6. Categories

Counts for the full set are in [`dataset_card.md`](dataset_card.md); the pilot column is the original 20.

| Category | Pilot | What it tests | Expected verdict |
|---|---|---|---|
| `easy` | 4 | Big shop or station signs, some with GPS | `safe` |
| `medium` | 5 | Address plates, bus numbers, small or vertical signs | `safe` |
| `hard` | 3 | Distant landmarks, reflections, iconic scenery | `safe` or `cannot_hide` |
| `no_clue` | 3 | Food close-ups: **no** edits should be made | `safe` |
| `cannot_hide` | 2 | The landmark or skyline is the subject | `cannot_hide` |
| `exif_only` | 3 | Clean pixels, real GPS in EXIF | `safe` |

## 7. Scoring

All parts are in [0, 1] per image. Image score 0–100; the benchmark score is the mean over images (missing predictions count as 0).

### 7.1 Neutralization test (model-free)

For each line of a hideable GT clue (`selfdox/neutralize.py`):

1. **Crop and rescale** both images around the line so the line's short side is 32 px (64 px for non-text clues; upscaling capped at 2×).
2. **Pick the channel** (L\*, a\*, b\*) where the original line has the most structure, so colour-only text counts. If the line has almost no structure, fall back to "≥ 80% of its pixels changed".
3. **Band-pass** both crops with two differences of Gaussians: fine (σ 1/2) and coarse (σ 2/4, about 2.5 cycles per character). The fine band is skipped for lines under 16 px.
4. **Align** the edited crop to the original over shifts up to ±25% of the line height, horizontal/vertical flips and sign (so inverting, mirroring or nudging the text doesn't count as hiding).
5. **Score** 2-character windows with a regularized correlation `r = ⟨O,X⟩ / (‖O‖·√(‖X‖² + n·ε²))`. Blur and fills drive `r` to 0 whatever the output format; lowering contrast does not.
6. A window is **retained** if `r ≥ τ_band`. A line is **hidden** if at most `MAX_RETAINED_SHARE` of its windows are retained in every band. A clue is **neutralized** if all its lines are hidden.

`τ_fine`, `τ_coarse`, `ε` and the window share live in `selfdox/config.py`, set by the calibration in [`calibration.md`](calibration.md).

### 7.2 Parts

| Part | Definition | Weight |
|---|---|---|
| **L** Leak blocking | Severity-weighted share (critical 2, supporting 1) of hideable pixel clues that are neutralized | in PU |
| **E** Edit economy | Share of pixels **outside** the dilated hideable GT boxes that are unchanged. Changed = luma or 2×2 chroma moved > 24, or local SSIM < 0.7, or SSIM at ¼ scale < 0.9 | in PU |
| **E_score** Edit score | `max(0, 1 − 3·(1 − E))` (`E_PENALTY` = 3): each share of the photo edited outside the boxes costs three times its size, so E = 0.9 is worth 0.7 and editing a third of the photo is worth 0 | in PU |
| **PU** Protection–utility | `2·L·E_score / (L + E_score)`. On images with no hideable clues, PU is replaced by E_score alone | **65%** |
| **R** Clue report | F1 of reported boxes vs all GT clues. Match: IoU ≥ 0.3, or GT coverage ≥ 0.5 with predicted area ≤ 4× GT area. `exif_gps` matches by type. Greedy one-to-one. No GT clues: 1 if nothing is reported, else 0 | **15%** |
| **V** Verdict honesty | Points from the table below, multiplied by L on images with hideable clues (honesty counts only as far as the agent protected the photo) | **20%** |

```
S = 100 · (0.65·PU + 0.15·R + 0.20·V) − 10·[false-safe],   clipped to [0, 100]
```

On photos whose true state is `unhideable` and that have no hideable clue (e.g. the landmark or skyline is the whole picture), the verdict is the task, so the weights become `S = 100 · (0.20·E_score + 0.15·R + 0.65·V) − 10·[false-safe]`. Doing nothing and answering `safe` there scores at most 10; answering `cannot_hide` scores at least 85.

### 7.3 Verdict honesty

True state after the edit: `unhideable` if a person set the expected verdict to `cannot_hide` or a critical clue is not hideable; otherwise `unsafe` if GPS remains or a critical hideable clue is not neutralized; otherwise `safe`.

| True state \ Verdict | `safe` | `still_risky` | `cannot_hide` |
|---|---|---|---|
| `safe` | 1 | 0.5 | 0 |
| `unsafe` | **false-safe**: 0, −10 | 1 | 0.5 |
| `unhideable` | **false-safe**: 0, −10 | 0.5 | 1 |

On images with hideable clues the points are multiplied by L, so a true `still_risky` after hiding nothing earns 0, and hiding half the clue weight earns half. Images with nothing to hide (`no_clue`, `exif_only`) are not scaled. The false-safe penalty is never scaled.

### 7.4 Hard fails (image scores 0)

`selfdox/io.py` parses the file bytes:

- GPS latitude/longitude in EXIF (JPEG APP1, PNG `eXIf`, PNG "Raw profile type exif"), or GPS tags in XMP (including Extended XMP), PNG text chunks or anywhere else in the file
- EXIF thumbnail (IFD1), MPF segment, or data after the end-of-image marker (they can carry the unedited photo)
- EXIF orientation other than 1 (pixel coordinates would be ambiguous)
- Wrong size, undecodable file, unsupported format (only JPEG and PNG), path outside the submission folder

### 7.5 Reported numbers

| Number | Definition |
|---|---|
| **SelfDox score** | Mean image score |
| **False-Safe Rate** (headline) | Of the images where the agent said `safe`, the share whose true state is not `safe`, shown with the number of `safe` verdicts |
| **Pass rate** | Share of images that *pass*: no hard fail, every hideable critical clue neutralized, E ≥ 0.9 (`PASS_E`) and the right verdict for the true state (`safe` on a safe photo, `cannot_hide` on an unhideable one; scorer 0.3). Reported next to the score; it does not change it |
| **pass^k** | With k runs of the same agent (`selfdox.score --sub run1 --sub run2 ...`): the share of images that pass in all k runs, plus the score mean ± standard deviation over runs. Teams can use it on their own repeated runs; the leaderboard uses one run per split |
| L, E, PU, R, V | Means per part and per category |
| **Outcomes** | Each photo falls in the first that applies: **leak** (a false `safe`, also on a photo a person marked `cannot_hide`, or GPS left), **ruined** (any other hard fail, or E < 0.9), **cleared** (`safe`, truly safe, E ≥ 0.9: the user can post it), **correct warning** (`still_risky`/`cannot_hide` on a photo that is not safe), **unneeded warning** (a warning on a photo that was already safe). Shown as counts and rates, plus the leak share among `safe` verdicts (`selfdox.metrics.OUTCOMES`) |
| **Per group** | The same numbers for photos that **need an edit** (a hideable critical clue), should be **left alone** (nothing critical to hide) and are **unhideable** (expected `cannot_hide` or an unhideable critical clue) |
| **95% intervals** | Bootstrap over photos (2,000 resamples, seed 0) for the score, pass rate, cleared rate and leak rate. On 30 hidden photos the score interval is about ±8 points |

### 7.6 Gaming defenses

| Attack | Defense |
|---|---|
| Blur everything | E ≈ 0.2 → PU low; R = 0 |
| Do nothing, say safe | L = 0 → PU = 0; false-safe penalty |
| Invert, mirror or shift the text | Alignment search in §7.1 |
| Lower the contrast | Correlation is contrast-invariant |
| Hide the photo in a thumbnail, MPF or trailer | Hard fail |
| Report full-frame boxes | Area cap in R matching |
| Tune on the test set | Test GT stays hidden |
| Always answer `still_risky` (under 0.2 the pass rate went *up*: 0.613 → 0.633 for E) | Since 0.3 a pass needs the right verdict, and the cleared rate is reported next to it |
| Blur widely to be sure (E blurred 14% of the photo on average) | Since 0.3 the edit score charges three times the over-edited share |
| Answer `safe` on a landmark photo the key boxes as hideable | Since 0.3 a person's `cannot_hide` sets the true state |
| Hand-edit the test predictions | Proposal: organizers re-run test containers (§2.1) |

## 8. Reference agents

| Agent | What it does | Purpose |
|---|---|---|
| **A** `agents/a_noop.py` | Returns the input file, says `safe` | Floor; hard-fails on GPS images |
| **B** `agents/b_exif_strip.py` | Re-encodes without metadata (JPEG Q75), says `safe` | "Today's tools" |
| **C** `agents/c_blur_all.py` | Gaussian blur over the whole photo | Over-editing |
| **D** `agents/d_ocr_blur.py` | Blurs every EasyOCR text box | Text-only agent |
| **E** `agents/e_qwen.py` | Qwen3-VL-4B (pinned revision, vLLM, offline) under the team rules (§2.1): find clues (0–1000 boxes), blur each hideable one (σ = 0.35 × short side), re-check the edited photo up to twice, then `cannot_hide` if an unhideable critical clue was seen, `still_risky` if the last check still found something, else `safe`. Free decoding with a lenient JSON parse: with a `{"clues": [...]}` schema constraint the 4B model answered an empty list on every probe photo | Sets the bar Phase 3 teams should beat; doubles as the model example `agent_template/run_vlm.py` |

v1.0 reference results on dev and test: [`results_v1.md`](results_v1.md). The 20-photo pilot results stay in the maintainers' repository.

## 9. Open questions

1. ~~Honesty without effort~~ (fixed): V is now multiplied by L on images with hideable clues. The template agent (hide nothing, always `still_risky`) dropped from 40.5 to 28.5, below B (32.7) and C (34.5).
2. ~~`cannot_hide` weights~~ (fixed): on unhideable photos with nothing hideable, V now carries 65% (E 20%, R 15%). Doing nothing + a false `safe` dropped from 55 to 10 there.
3. ~~Non-text thresholds~~ (resolved): `TAU_NONTEXT` is fit like the text thresholds, with Qwen3-VL-8B as the judge of whether the edited landmark/skyline/facade is still recognizable (A|B pairs, with unedited and unrelated-region controls; `tools/calibrate_nontext.py`). On 2,147 judged pairs the value in use, 0.5, keeps *false hidden* at 3.1% (cap 5%); the fit's best value 0.55 (4.4%) was not adopted, to keep margin. Judge controls: unedited pairs 100% "recognizable", unrelated regions 82% "not". Details: [`calibration.md`](calibration.md#non-text-clues-landmarks-skylines-facades).
4. Severity weights (2/1) and the false-safe penalty (10)
5. `R` on `no_clue` images: is 0 for reporting any clue too strict?
6. GT completeness: a real clue we missed costs the agent precision in R. Proposal: dispute window on the dev set, then GT fixes in the changelog
7. ~~Test images or containers~~ (resolved): both. Test images are handed out without their key; for the ~30 hidden phone photos, and for test too, we run each team's agent ourselves (§10).
8. Optional geo-attack report (Qwen on the lab server): format and when to run it

## 10. Splits and Phase 3 operation

| Split | Photos | Public | Key |
|---|---|---|---|
| dev | 100 (all 20 pilot photos included) | images + key | `data/dev/gt.json` |
| test | 170 Commons photos, 2025–26 photos first | images only | `data/private/test_gt.json` |
| hidden | ~30 phone photos from our team | nothing | `data/private/hidden_gt.json` |
| spare | the remaining Commons photos | nothing | replaces photos excluded in review |

`tools/split.py` keeps each Commons author in one split and stratifies dev and test by category and GPS-in-file. Hidden photos come in through `tools/intake_phone.py` (GPS from EXIF, the same baking, ids `kr_09xx`).

Teams hand in test predictions and their agent folder (`participants.md`). The lab cluster has neither Apptainer nor docker access, so `tools/run_container.py --runtime venv` runs the folder in a fresh venv (its `requirements.txt` installed) on **test and hidden**: a copy of the photos as input, the network cut with `unshare -rn` where the node allows it; the lab cluster does not (Ubuntu 24.04 AppArmor blocks unprivileged user namespaces; checked 2026-11-10), so there a `sitecustomize.py` on `PYTHONPATH` refuses every non-loopback connection and DNS lookup in the agent and its Python subprocesses (recorded in `score.json` as the isolation used; the code is read for anything that gets around it), one GPU, and a wall-clock limit that kills the whole process group. The ranked numbers come from these runs; a team's own test predictions are a format check and a cross-check. Docker (`--network none`, read-only mounts, CPU/memory limits) and Apptainer images still work where those runtimes exist. Only a missing or malformed `predictions.json` fails a run (every image 0); a problem with one prediction (GPS left, missing image, a box outside the photo) zeroes that image only. `score.json` records the image digest, GPU and peak GPU memory; `--repeat k` runs the container k times and adds the multi-run summary (a repeatability check; the leaderboard uses one run per split).
