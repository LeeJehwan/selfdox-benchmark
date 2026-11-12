# SelfDox for Phase 3 teams

Build an agent that takes one street photo from Korea, finds what would tell a stranger
**where it was taken**, hides only that, and says honestly whether the photo is now safe
to post.

## 1. The task

| | |
|---|---|
| **Input** | One JPEG, long side 2048 px. Some files still carry GPS in their EXIF. Faces and license plates are already blurred. |
| **Output 1: clues** | Every location clue you found: `bbox` `[x, y, w, h]` in input pixels, `type`, optional `text` and `reason`. GPS in the file is reported as `{"bbox": null, "type": "exif_gps"}`. |
| **Output 2: edited image** | The same photo, same width and height, with your clues hidden by any method (blur, pixelate, fill, inpaint, fake text) and **no GPS or other location metadata**. JPEG or PNG. |
| **Output 3: verdict** | `safe` (nothing left points to the place), `still_risky` (you hid what you could, but risk remains) or `cannot_hide` (the place itself is the subject, e.g. a palace or skyline, and can't be hidden without destroying the photo). |

What counts as a clue: shop names (especially with a branch or phone number), road-name
address plates and building numbers, apartment names, station and bus stop names, bus
routes, phone numbers, school names, landmarks, distinctive skylines and facades, and GPS
in the file. Not clues: generic words, menus and prices, product logos, car makes, license
plates and faces. The full list with examples is in
[`annotation_guide.md`](annotation_guide.md).

## 2. Output format

```
outputs/
├── predictions.json
└── images/<id>.png | <id>.jpg
```

```json
{"agent_name": "team7_agent", "benchmark_version": "0.3",
 "predictions": [
  {"id": "kr_0001",
   "clues": [{"bbox": [610, 170, 920, 165], "type": "shop_sign", "text": "거인통닭",
              "reason": "shop name"},
             {"bbox": null, "type": "exif_gps", "reason": "GPS in EXIF"}],
   "edited_image": "images/kr_0001.png",
   "verdict": "safe"}]}
```

Schema: [`selfdox/schemas/submission.schema.json`](../selfdox/schemas/submission.schema.json).
Check a folder before you hand it in:

```bash
uv run python -m selfdox.validate --sub outputs --inputs data/test/images
```

## 3. Rules

> Status: **proposal, pending agreement in our team**. The model list below
> is an example of what we propose; the final list will be announced with v1.0.

| Rule | Detail |
|---|---|
| Models | Only the allowed open models may be called, for example [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) at revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` (or its official GGUF/FP8 builds). Full precision needs about 9 GB of GPU memory. |
| No training | Weights stay as published: no fine-tuning, LoRA or distillation. Prompts, tools and the agent loop are yours. |
| Other tools | Classical image tools and small non-generative models (OCR, face or text detectors, blur and inpainting filters) are fine. No other LLM or VLM, no web search, no online API. |
| Offline | The agent must run with no network access. |
| Disclosure | List the tools and the model build you used in your write-up. |

**Proposed additions (2026-11-10, not agreed yet).** Today the rules leave open how much
compute an agent may use, which extra models count as "small" and how test predictions
are checked. We propose:

| Rule | Proposal |
|---|---|
| Other models | Every model besides Qwen3-VL-4B at most **1B parameters**, from an allow-list published by a fixed date. Allowed: OCR and text detectors, LaMa-class (non-diffusion) inpainting, prompted detectors (Grounding DINO, OWL-ViT), CLIP-type embeddings. Not allowed: diffusion inpainting, VLM-based OCR (Florence-2, GOT-OCR), geolocation models. Public offline place-name data (gazetteers, road-name address lists) is allowed if you disclose it. |
| Budget | The same per-photo time budget on test and hidden (for example a mean of 120 s per photo). Report the model calls, tokens and seconds per photo in the optional `meta` field of each prediction. |
| Weights | We mount the official weights at `/models`. Any file over 50 MB in your image must match a hash on the allow-list. |
| Decoding | State your temperature and seed. |
| People and training | No hand edits of the outputs. No training or fine-tuning of model weights on benchmark photos, dev included. Tuning prompts, thresholds and code on dev is what dev is for. |

## 4. Data

| Split | Photos | You get | Used for |
|---|---|---|---|
| **dev** | 100 | images **and** the answer key (`data/dev/gt.json`) | building and scoring your agent yourself |
| **test** | 170 | images only | your predictions file |
| **hidden** | 30 | nothing | we run your agent folder on these ourselves (and on test) |

All photos are Wikimedia Commons photos from Seoul and Busan except the hidden ones,
which our team took. Licenses and authors: `data/*/ATTRIBUTION.md`. About one photo in
five has GPS in its EXIF on purpose: removing it is part of the task.

## 5. How you are scored

No model runs at scoring time. For each answer-key clue, a pixel-structure test decides
whether the clue can still be recovered from your edited image (inverting, mirroring,
shifting or lowering the contrast of text does **not** count as hiding it). The details
and the calibration against OCR and a VLM judge are in [`design.md`](design.md#7-scoring)
and [`calibration.md`](calibration.md).

| Part | Question | Weight |
|---|---|---|
| **PU** | Harmonic mean of **L** (share of answer-key clues you hid, critical ones count double) and the **edit score** `max(0, 1 − 3·(1 − E))`, where **E** is the share of the rest of the photo you left untouched: editing 10% too much costs 30% | 65% |
| **R** | Did you report the right clue boxes? (F1) | 15% |
| **V** | Is your verdict honest? A false `safe` also costs 10 points. When we marked a photo `cannot_hide` (the place itself is the subject), `cannot_hide` is the right answer even if you could blur parts of it | 20% |

Three rules that decide most points:

- **"Safe" is about critical clues and GPS.** A photo is safe after your edit when every
  hideable *critical* clue is hidden and no GPS is left. Supporting clues raise L but
  don't make a photo unsafe. A photo we marked `cannot_hide` is never safe.
- **Honesty counts as far as you protected the photo.** On photos with hideable clues,
  V is multiplied by L: answering `cannot_hide` on a landmark photo while leaving its
  shop signs readable earns little.
- **Margins for scenes.** Landmarks, skylines and facades are judged on coarse
  structure that reaches past a tight box: blur them with a margin of about a quarter
  of the box's short side.

An image scores 0 if GPS is still in the file, the file hides another image (EXIF
thumbnail, MPF, data after the end), has an orientation tag, has the wrong size, or its
prediction fails validation (e.g. a box outside the photo).

Score yourself on dev at any time, and look at why each photo scored what it did:

```bash
uv run python -m selfdox.score --gt data/dev/gt.json --sub outputs --out results.json
uv run python -m selfdox.view --gt data/dev/gt.json --sub outputs --results results.json --out view
```

## 6. What to hand in

1. **Test predictions:** your `outputs/` folder for the 170 test images, zipped. We
   check its format and compare it with our own run of your agent on test (item 2);
   the leaderboard uses our run.
2. **Your agent folder**, in the [`agent_template/`](../agent_template/) layout:
   `requirements.txt`, `run.py` and anything they need. We run it ourselves on the lab
   cluster: each submission once on **test** for the interim board, and after the deadline
   your last submission once more on **test** and once on **hidden** for the final board.
   It runs in a fresh Python venv (the cluster has no Docker), with
   `tools/run_container.py --runtime venv`:
   - command: `python run.py --input <photos> --output <folder>`, or the list you put in
     `selfdox_run.json`, e.g.
     `{"command": ["python", "run_vlm.py", "--inputs", "{input}", "--out", "{output}",
     "--model", "{models}/Qwen3-VL-4B-Instruct"]}`
   - the venv gets your `requirements.txt` installed before the run; the run itself has
     **no network**: Python cannot open any connection except to localhost (a
     `sitecustomize.py` we put on `PYTHONPATH`, inherited by Python subprocesses; the
     cluster does not allow network namespaces), Hugging Face is set offline, and we
     read your code for anything that gets around it. The allowed weights are at
     `{models}`
   - your agent reads a copy of the photos and writes only to its output folder
   - one GPU with 24 GB (RTX 3090 class)
   - 90 minutes for the 30 hidden photos (test: the same per-photo time); then the whole
     process group is stopped and whatever `predictions.json` holds is scored (write it
     as you go, replacing the file atomically as `agents/e_qwen.py` does; no file means
     every image scores 0)
   - a problem with one prediction (GPS left, image missing, a box outside the photo)
     scores that image 0; the others are scored as usual
   - we record a hash of your folder, the GPU and your peak GPU memory

   A Docker image (`agent_template/Dockerfile`) still works where Docker is available,
   handed in by digest (`name@sha256:…`).

   Run it yourself exactly like this before you hand it in (the runner is in the
   package; without `--python` it builds a fresh venv from your `requirements.txt`):

   ```bash
   uv run python -m tools.run_container --runtime venv --image my_agent --split dev --name try1
   # -> runs/try1/dev/score.json, container.log, validate.txt
   ```

   The official runs use Python 3.12 on the lab cluster (NVIDIA driver 580 / CUDA 13.0,
   one RTX A6000 or RTX 3090).
3. A short write-up: the tools and model build, and how your agent decides the verdict.

**How:** pack the folder (and, if you like, your own test predictions) with

```bash
uv run python -m selfdox.submit --agent my_agent --team team7 --contact you@example.com \
    --model "Qwen/Qwen3-VL-4B-Instruct@ebb281e (bf16)" --tools "EasyOCR 1.7.2" \
    --temperature 0 --seed 0 --smoke 3 [--test-sub outputs_test]
```

It checks the folder, runs it on 3 dev photos the way we will, and writes
`submission_team7.zip` with a `metadata.json`. Open a **Submission** issue on the
benchmark repository and attach the zip (or a link, over 25 MB). At most one submission
per team per day (Korean time, KST = UTC+9, by when the issue is opened); a second one on
the same day is not run and doesn't count. Your last submission before the deadline is the
one on the final board. Deadlines are announced in class and in the pinned issue.

The template agent and our reference agents show the format:
[`agent_template/run.py`](../agent_template/run.py) (classical tools),
[`agents/e_qwen.py`](../agents/e_qwen.py) (Qwen3-VL-4B with vLLM: find, blur, re-check,
verdict).

## 7. Leaderboard

| Number | Meaning |
|---|---|
| **SelfDox score** | Mean image score (0–100) on test + hidden (the interim board before the deadline: test only); missing images count 0 |
| **False-safe rate** | Of the images you called `safe`, the share that were not; shown with your number of `safe` verdicts |
| **Pass rate** | Share of images with no hard fail, every hideable critical clue hidden, E ≥ 0.9 and the right verdict: `safe` when the photo is safe after your edit, `cannot_hide` when it can't be made safe. Warning on a photo you already made safe does not pass |
| **Outcomes** | Per photo, the first that applies: **leak** (false `safe`, also on a photo we marked `cannot_hide`, or GPS left), **ruined** (other hard fail or E < 0.9), **cleared** (`safe`, truly safe, E ≥ 0.9), **correct warning**, **unneeded warning** (a warning on a photo that was already safe). Shown as counts and rates, also per photo group (needs an edit / leave alone / unhideable) |
| **95% interval** | Bootstrap over photos. On 30 hidden photos it is about ±8 points, so smaller gaps there are noise. The ranking still follows the score; the intervals show how sure the order is |

Before the deadline we run each submission once on test, and the board shows that run
(interim). After the deadline we run your last submission once more on test and once on
hidden, and the final board uses those two runs. If a run fails because of our cluster (a
node or GPU error), we run it again; a crash or timeout in your agent counts. If your agent
answers differently from run to run, it is scored on the run it gives. Ties are broken by
the false-safe rate over the board's photos (test on the interim board, test + hidden on the
final board; lower is better).

*Proposal (not agreed yet):* rank only on runs we made ourselves; show the score with its
interval, the cleared and leak rates, the per-group results, test and hidden separately,
seconds, calls and tokens per photo and the peak GPU memory; and freeze the answer key
before the final runs, after a dispute window on dev, with every change in the changelog.

Where: [`LEADERBOARD.md`](../LEADERBOARD.md), rebuilt after every batch of official runs.
Reference agents A, B, E and the oracle are shown (not ranked) on the test photos. Next to
each team: the paired difference to the team below with its 95% interval. Until the rules
are agreed, the board applies the proposal above and ranks only our runs. Every official
run is a single run (decided 2026-11-12), and the final board uses one test run and one
hidden run of your last submission, made after the deadline (decided 2026-11-13).

## 8. Questions, bugs, answer-key disputes

Open an issue on the benchmark repository and pick a template: **Question**, **Bug in
the benchmark**, **Answer-key dispute** (dev photos) or **Submission**. We answer within
two working days; fixes get a new version in [`CHANGELOG.md`](../CHANGELOG.md) and all
stored results are rescored. Common questions and known issues:
[`FAQ.md`](FAQ.md).

- **Can I see the test answer key?** No. It stays private so that the test measures your
  agent, not tuning. If you think the dev key misses a clue or marks one wrongly, open an
  issue; fixes go into the changelog.
- **What if my agent can't decide?** Say `still_risky`. A false `safe` is the most
  expensive mistake.
- **Do I have to hide landmarks?** Only when they can be hidden without destroying the
  photo. If the landmark is the subject, leave it and answer `cannot_hide`.
