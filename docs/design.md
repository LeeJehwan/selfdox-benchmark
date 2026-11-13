# SelfDox Benchmark Design (v1.1 draft)

Status: **v1.1 draft** (scorer 0.4). The scorer (`selfdox/`), the full data pipeline and the split tools are in place; the 300-photo answer key is in human review (Claude's first pass is filled in). Text thresholds come from an offline OCR calibration and the non-text threshold from a VLM judge ([`calibration.md`](calibration.md)). For participants: [`participants.md`](participants.md). For the data: [`dataset_card.md`](dataset_card.md).

## 1. Changes

### Scorer 0.4, the 0–100 board and the submission bot (v1.1)

Every stored result was rescored (`tools/rescore.py --recompute-r`: R recomputed from each submission's `predictions.json`, the clue rows and E kept; spot-checked against scoring the images again: identical on every photo checked). Why: the team's review of the talk deck (2026-11-13): count-based R was hard to explain and its thresholds had no clear basis.

| Rule | v1.0 (scorer 0.3) | v1.1 (scorer 0.4) | Effect (test, 170 photos) |
|---|---|---|---|
| Clue report R | F1 of **counted** matches: IoU ≥ 0.3, or GT coverage ≥ 0.5 with predicted area ≤ 4× GT, greedy one-to-one | F1 **by area** (§7.2): recall = how much of each key clue the reported boxes cover, precision = how much of the reported area lies on a clue (key boxes dilated like E). No thresholds | E's mean R 0.52 → 0.62, score 63.6 → **65.0**; A 11.7, oracle 97.5 unchanged; B 22.2 → 22.3. A false box on an exif-only photo now costs all its precision (E on `exif_only` 96.3 → 94.1) |
| Leaderboard score | mean image score | the same mean **rescaled to 0–100**: 0 = agent A, 100 = the oracle, on the board's own photos (§7.5); per-photo scores unchanged | E on the interim board: **62.2** (56.8–67.4); B 12.4 |
| Submissions | one per team per day (Korean time) | **at most 5 per team in total**; only scored ones count | |
| Interim board | our run of each submission on test | the **submission bot** scores the team's uploaded test outputs (§10); totals only | results in minutes, no cluster time per submission |
| Final board | our runs of the last submission on test + hidden | unchanged | |
| Models | only Qwen3-VL-4B (proposal: other models ≤ 1B from an allow-list) | **any open-weight model(s), each ≤ 4B parameters**, declared as `repo@revision` (§2.1) | |

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

Decided 2026-11-13. The model list may later narrow to specific models (announced in the pinned issue).

| Rule | Detail |
|---|---|
| Models | Any open-weight model or models, **each at most 4B parameters** (the size the model is published as; [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), agent E, is allowed). Each one is declared as Hugging Face `repo@revision` (`selfdox.submit --model`, repeatable); the organizers download exactly those into `/models` for the final runs |
| No training | Weights stay as published: no fine-tuning, LoRA or distillation. Prompts, tool use, thresholds and the agent loop are the team's to design and may be tuned on dev |
| Other tools | Classical image tools and libraries (OCR, detectors, blur and inpainting filters) are allowed; weight files over 50 MB that are not on Hugging Face are declared as `<https url>#sha256=<hex>` (`selfdox.submit --weights`) and downloaded by the organizers into `{models}` |
| Offline | The agent runs without network access: no web search, no online API (§10) |
| No hand edits | Outputs come from the agent |
| Pinned versions | `requirements.txt` pins every package version; the official runs install it into a fresh venv |
| Limits | One 24 GB GPU; 90 minutes for the 30 hidden photos, the same time per photo on test; at most 5 submissions per team in total |
| Disclosure | Each submission states its temperature and seed and lists its tools |

Why a size cap rather than one model: teams can pick the model that suits their pipeline, while the field stays fair and cheap to run on one GPU. Memorized web photos give little advantage because the task is to hide given clues, not to guess the location. This rule replaces the 2026-11-10 proposal (Qwen3-VL-4B as the only VLM, every other model ≤ 1B from an allow-list).

| Gap found in the 2026-11-10 review | How it is handled |
|---|---|
| Test predictions are made by the teams | The interim board scores the teams' own test outputs (the bot, §10); the final board uses only the organizers' runs of each team's last agent on test **and** hidden. Hand-edited outputs move only the interim board |
| Which run counts | Every official run is a single run (2026-11-12, 2026-11-13). A run that fails on our side (node or GPU error) is run again; a crash or timeout in the agent counts. Greedy decoding is not fully repeatable under vLLM batching, and that variation is part of the agent's result |
| Weights can't be checked | No file over 50 MB in the team's folder; bigger weights come from `{models}`: the organizers download the declared `repo@revision` models and the declared weight files, checked against their sha256 |
| No budget on test | The same time per photo on test and hidden; calls, tokens and seconds per photo may be reported in `meta` |

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
| **R** Clue report | F1 **by area** (scorer 0.4) of the reported boxes against all GT clues, hideable or not; formula below. No GT clues: 1 if nothing is reported, else 0 | **15%** |
| **V** Verdict honesty | Points from the table below, multiplied by L on images with hideable clues (honesty counts only as far as the agent protected the photo) | **20%** |

```
S = 100 · (0.65·PU + 0.15·R + 0.20·V) − 10·[false-safe],   clipped to [0, 100]
```

The clue report, with U = the union of the reported boxes (pixels), w_g = the severity weight of GT clue g (critical 2, supporting 1) and D = the union of the GT boxes dilated like E (10% of each side, at least 8 px):

```
recall    = Σ_g w_g · |g ∩ U| / |g|  /  Σ_g w_g        (exif_gps: 1 if a clue of that type is reported, else 0)
precision = |U ∩ D| / |U|                              (1 when no box is reported)
R         = 2 · precision · recall / (precision + recall)
```

So several small boxes over one sign score like one tight box, duplicates cost nothing, a margin inside the dilation is free, and a full-photo box gets a precision equal to the clue's share of the photo. No other constants (`selfdox/metrics.py` `report_area`).

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
| **SelfDox score** | Mean image score ("Raw"). The leaderboard shows it on a 0–100 scale between two reference agents on the board's own photos: `100 · (Raw − S_A) / (S_O − S_A)`, clipped to [0, 100], where S_A is agent A (do nothing) and S_O the oracle (knows the key). Interim board: test, S_A = 11.7, S_O = 97.5 (scorer 0.4); final board: test + hidden. Linear, so the ranking is the raw ranking; intervals and the paired difference are rescaled the same way (`tools/leaderboard.py`) |
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
| Report full-frame boxes | R precision is by area: a full-frame box is mostly not clue |
| Report many tiny or duplicate boxes | R works on the union of the boxes, so counts don't matter |
| Tune on the test set | Test GT stays hidden |
| Tune on the test set through the interim board | At most 5 scored submissions per team in total, and the bot reports totals only (no per-photo numbers). The hidden photos, which no board shows before the deadline, show whether a gain carries over |
| Read the test key from the bot | The key is stored encrypted (`data/test/gt.json.enc`); the passphrase is a repository secret only the scoring workflow reads; the workflow never runs submitted code and comments totals only |
| Always answer `still_risky` (under 0.2 the pass rate went *up*: 0.613 → 0.633 for E) | Since 0.3 a pass needs the right verdict, and the cleared rate is reported next to it |
| Blur widely to be sure (E blurred 14% of the photo on average) | Since 0.3 the edit score charges three times the over-edited share |
| Answer `safe` on a landmark photo the key boxes as hideable | Since 0.3 a person's `cannot_hide` sets the true state |
| Hand-edit the test outputs | Against the rules, and it moves only the interim board: the final board is the organizers' runs of each team's last agent on test and hidden (§2.1) |
| Resubmit the same agent to re-roll a run | **Decided (2026-11-13):** the final board uses one test run and one hidden run of each team's last submission, both made after the deadline |

## 8. Reference agents

| Agent | What it does | Purpose |
|---|---|---|
| **A** `agents/a_noop.py` | Returns the input file, says `safe` | Floor; hard-fails on GPS images |
| **B** `agents/b_exif_strip.py` | Re-encodes without metadata (JPEG Q75), says `safe` | "Today's tools" |
| **C** `agents/c_blur_all.py` | Gaussian blur over the whole photo | Over-editing |
| **D** `agents/d_ocr_blur.py` | Blurs every EasyOCR text box | Text-only agent |
| **E** `agents/e_qwen.py` | Qwen3-VL-4B (pinned revision, vLLM, offline) under the team rules (§2.1): find clues (0–1000 boxes), blur each hideable one (σ = 0.35 × short side), re-check the edited photo up to twice, then `cannot_hide` if an unhideable critical clue was seen, `still_risky` if the last check still found something, else `safe`. Free decoding with a lenient JSON parse: with a `{"clues": [...]}` schema constraint the 4B model answered an empty list on every probe photo | Sets the bar Phase 3 teams should beat; doubles as the model example `agent_template/run_vlm.py` |

Reference results on dev and test (scorer 0.4): [`results_v1.md`](results_v1.md). The 20-photo pilot results stay in the maintainers' repository.

## 9. Open questions

1. ~~Honesty without effort~~ (fixed): V is now multiplied by L on images with hideable clues. The template agent (hide nothing, always `still_risky`) dropped from 40.5 to 28.5, below B (32.7) and C (34.5).
2. ~~`cannot_hide` weights~~ (fixed): on unhideable photos with nothing hideable, V now carries 65% (E 20%, R 15%). Doing nothing + a false `safe` dropped from 55 to 10 there.
3. ~~Non-text thresholds~~ (resolved): `TAU_NONTEXT` is fit like the text thresholds, with Qwen3-VL-8B as the judge of whether the edited landmark/skyline/facade is still recognizable (A|B pairs, with unedited and unrelated-region controls; `tools/calibrate_nontext.py`). On 2,147 judged pairs the value in use, 0.5, keeps *false hidden* at 3.1% (cap 5%); the fit's best value 0.55 (4.4%) was not adopted, to keep margin. Judge controls: unedited pairs 100% "recognizable", unrelated regions 82% "not". Details: [`calibration.md`](calibration.md#non-text-clues-landmarks-skylines-facades).
4. Severity weights (2/1) and the false-safe penalty (10)
5. `R` on `no_clue` images: is 0 for reporting any clue too strict?
6. GT completeness: a real clue we missed costs the agent precision in R. Proposal: dispute window on the dev set, then GT fixes in the changelog
7. ~~Test images or containers~~ (resolved): both. Test images are handed out without their key, and the bot scores the teams' test outputs for the interim board; for the final board we run each team's last agent ourselves on test and the ~30 hidden phone photos (§10).
8. Optional geo-attack report (Qwen on the lab server): format and when to run it

## 10. Splits and Phase 3 operation

| Split | Photos | Public | Key |
|---|---|---|---|
| dev | 100 (all 20 pilot photos included) | images + key | `data/dev/gt.json` |
| test | 170 Commons photos, 2025–26 photos first | images only | `data/private/test_gt.json` |
| hidden | ~30 phone photos from our team | nothing | `data/private/hidden_gt.json` |
| spare | the remaining Commons photos | nothing | replaces photos excluded in review |

`tools/split.py` keeps each Commons author in one split and stratifies dev and test by category and GPS-in-file. Hidden photos come in through `tools/intake_phone.py` (GPS from EXIF, the same baking, ids `kr_09xx`).

Teams hand in their agent folder and their own test outputs in one Submission issue (`participants.md` §6), at most 5 times per team.

**Interim board: the submission bot.** `.github/workflows/score-submission.yml` in the public repository runs `tools/bot_score.py` when a Submission issue is opened or edited (or by hand, `workflow_dispatch`, for dry runs). It checks the rules first (`leaderboard/settings.json`: the deadline, and which GitHub accounts may submit for a team name; otherwise the first account that submits for a name owns it, and one account submits for one team; at most 5 scored submissions per team). It downloads the agent zip and the test-outputs zip (https from GitHub only; storage hosts only as redirect targets; size and time caps), recomputes the agent folder's sha256 from the zip's members and refuses a mismatch with `metadata.json`, unzips the outputs safely (no absolute paths or `..`, only `predictions.json` and `images/`, a total size cap), validates the outputs against the 170 test photos, decrypts the test key (`data/test/gt.json.enc`, AES-256-CBC with PBKDF2 via `openssl`; the passphrase is the repository secret `TEST_GT_KEY`) into the runner's temporary folder, scores, and comments with totals only: the 0–100 score with its 95% interval, the raw score, the submission count n/5, the agent's sha256 (from its `metadata.json`) and the scorer version. Scoring writes nothing; a separate publish step fetches the latest board, checks the rules again on it, appends the entry to `leaderboard/interim.json`, rebuilds `LEADERBOARD.md` (`tools/leaderboard.py`; the 0/100 anchors and reference rows come from `leaderboard/refs.json`, aggregates only), pushes (retrying from a fresh fetch if the push is refused), and only then posts the comment, so a shown score is always a counted one. Runs for different issues go in parallel. Text from the submission is quoted in code spans in the comment. It never runs submitted code and never prints the key.

**Final board: our runs.** After the deadline each team's last submission's agent folder (checked by its sha256) runs once on test and once on hidden, with the declared models downloaded at their revision into `/models`. The lab cluster has neither Apptainer nor docker access, so `tools/run_container.py --runtime venv` runs the folder in a fresh venv (its pinned `requirements.txt` installed): a copy of the photos as input, the network cut with `unshare -rn` where the node allows it; the lab cluster does not (Ubuntu 24.04 AppArmor blocks unprivileged user namespaces; checked 2026-11-10), so there a `sitecustomize.py` on `PYTHONPATH` refuses every non-loopback connection and DNS lookup in the agent and its Python subprocesses (recorded in `score.json` as the isolation used; the code is read for anything that gets around it), one GPU, and a wall-clock limit that kills the whole process group. The final board comes from these runs only; a team's uploaded test outputs count only on the interim board. Docker (`--network none`, read-only mounts, CPU/memory limits) and Apptainer images still work where those runtimes exist. Only a missing or malformed `predictions.json` fails a run (every image 0); a problem with one prediction (GPS left, missing image, a box outside the photo) zeroes that image only. `score.json` records the image digest, GPU and peak GPU memory; `--repeat k` runs the container k times and adds the multi-run summary (a repeatability check for your own runs; official runs are single runs, and the leaderboard refuses repeated runs).
