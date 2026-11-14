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

| Rule | Detail |
|---|---|
| Models | Any open-weight model or models, **each at most 4B parameters** (the size the model is published as; [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), our agent E, is allowed). Declare each one as Hugging Face `repo@revision` when you submit; we download exactly those into `{models}` for the official runs. |
| No training | Weights stay as published: no fine-tuning, LoRA or distillation. Prompts, tools, thresholds and the agent loop are yours, and tuning them on dev is what dev is for. |
| Other tools | Classical image tools and libraries (OCR, detectors, blur and inpainting filters) are fine. Their weights come like the models': files under 50 MB may ship in your folder; bigger ones are declared as a public https URL with their sha256 (`--weights "<url>#sha256=<hex>"`), and we download and check them into `{models}`. Point the library at `{models}` and switch its own downloads off (EasyOCR: `model_storage_directory`, `download_enabled=False`). |
| Offline | The agent runs with no network access: no web search, no online API. |
| No hand edits | Your outputs come from your agent. Nobody edits them by hand. |
| Pinned versions | `requirements.txt` pins every package version (`name==x.y.z`), so our run installs what you tested. |
| Limits | One 24 GB GPU; **90 minutes for all 200 photos of the final run together** (test 170 + hidden 30, about 27 s per photo); **at most 5 submissions per team** in total (§6). |
| Disclosure | State your temperature and seed, and list the tools you used. |

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
| **R** | Did you box the clues? By area: how much of each answer-key clue your boxes cover (critical clues count double), and how much of your box area lies on a clue (with a small free margin). F1 of the two; `exif_gps` counts when you report it | 15% |
| **V** | Is your verdict honest? A false `safe` also costs 10 points. When we marked a photo `cannot_hide` (the place itself is the subject), `cannot_hide` is the right answer even if you could blur parts of it | 20% |

Four rules that decide most points:

- **"Safe" is about critical clues and GPS.** A photo is safe after your edit when every
  hideable *critical* clue is hidden and no GPS is left. Supporting clues raise L but
  don't make a photo unsafe. A photo we marked `cannot_hide` is never safe.
- **Honesty counts as far as you protected the photo.** On photos with hideable clues,
  V is multiplied by L: answering `cannot_hide` on a landmark photo while leaving its
  shop signs readable earns little.
- **Margins for scenes.** Landmarks, skylines and facades are judged on coarse
  structure that reaches past a tight box: blur them with a margin of about a quarter
  of the box's short side.
- **Boxes are measured by area (scorer 0.4).** Several small boxes over one sign score
  like one tight box around it; one box over the whole photo covers every clue but is
  mostly not clue, so it scores close to 0. A margin of about 10% (at least 8 px) costs
  nothing. Only boxes count for precision: on a photo whose only key clue is `exif_gps`,
  any box you report is off the clues and R is 0 there.

An image scores 0 if GPS is still in the file, the file hides another image (EXIF
thumbnail, MPF, data after the end), has an orientation tag, has the wrong size, or its
prediction fails validation (e.g. a box outside the photo).

Score yourself on dev at any time, and look at why each photo scored what it did:

```bash
uv run python -m selfdox.score --gt data/dev/gt.json --sub outputs --out results.json
uv run python -m selfdox.view --gt data/dev/gt.json --sub outputs --results results.json --out view
```

## 6. What to hand in

Each submission has two parts, and both go up together on the submission site:

1. **Your agent folder**, in the [`agent_template/`](../agent_template/) layout:
   `requirements.txt` (pinned versions), `run.py` and anything they need. After the
   deadline we run the folder of your **last** submission ourselves (§7).
2. **Your agent's outputs on the 170 test photos** (`outputs/` with `predictions.json` and
   `images/`). The site scores them for the interim board.

**Pack both** with

```bash
uv run python -m selfdox.submit --agent my_agent --team team7 \
    --model "Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17" \
    --tools "EasyOCR 1.7.2" --temperature 0 --seed 0 --smoke 3 --test-sub outputs_test
```

- `--model`: each open-weight model your agent loads, as Hugging Face `repo@revision`, each
  at most 4B parameters. Repeat it for several models; `none` if you use none.
- `--weights`: each weight file over 50 MB that is not on Hugging Face (an OCR detector, for
  instance), as `"<https url>#sha256=<hex>"`. Repeat for several. It lands in `{models}`
  under the URL's file name (the last part of the path).
- `--smoke 3` runs the folder on 3 dev photos the way we will.
- `--test-sub` validates your test outputs.

It writes two files:
- `submission_team7.zip`: the agent folder, a `metadata.json` (team, models, tools,
  decoding, the folder's sha256; `--contact` adds a contact, but the file is public on the
  submissions dataset) and your test predictions;
- `test_outputs_team7.zip`: `predictions.json` and the edited images (JPEG files as they
  are, other formats saved as JPEG quality 92), about 100 MB.

**Then submit on the site**, <https://huggingface.co/spaces/wpghks7/selfdox>:
- sign in with Hugging Face (a free account is enough);
- enter your **team code**, the one the organizers gave your team;
- add `submission_team7.zip` (at most 30 MB) and `test_outputs_team7.zip` (at most
  300 MB) and press submit.

Your upload becomes a pull request on the dataset `wpghks7/selfdox-submissions`. The
submission files are public there, as they were in issues, so do not put secrets in them.

**Team codes.** Any member can submit; no single account owns a team. Each team has a
code; any member can submit from their own Hugging Face account with it. The site shows
the team name the code belongs to. The code itself is never published, only a proof that
your account knows it. If you lose your code, ask in a Question issue.

**The score appears in about 5 to 10 minutes.** A workflow checks new submissions every 5
minutes: it verifies your team, applies the rules, scores, comments on your pull request
with the totals (your score on the 0–100 board scale with its 95% interval, the raw score,
which submission this is (n/5), the agent's sha256 and the scorer version), closes it and
updates the leaderboard on the site. It shows totals only, no per-photo numbers. It never
runs your code; it only reads the two zips, and it checks that the agent files match the
sha256 in `metadata.json`.

**At most 5 submissions per team, in total.** Only a scored submission counts; a rejected
or failed one does not. To submit again, upload again. Your last scored submission before
the deadline is the one on the final board; after the deadline the site refuses new
submissions. If the scoring fails on our side for a submission made before the deadline,
we score it again. Further announcements will be posted in the pinned issue.

**How we run your agent after the deadline.** Your last submission's agent folder (checked
by its sha256) runs **once, offline, on all 200 photos** (test 170 + hidden 30, in one
folder) in our environment, the same for every team, and only this run ranks: `tools/run_container.py --runtime venv` on the lab cluster (no Docker needed; the
cluster can't run it):
- the same node and Python 3.12 for every team (NVIDIA driver 580 / CUDA 13.0, one 24 GB
  GPU, RTX A6000 or RTX 3090 class); a fresh venv with your `requirements.txt` installed;
- command: `python run.py --input <photos> --output <folder>`, or the list you put in
  `selfdox_run.json`, e.g.
  `{"command": ["python", "run_vlm.py", "--inputs", "{input}", "--out", "{output}",
  "--model", "{models}/Qwen3-VL-4B-Instruct"]}`;
- the models you declared, downloaded at their revision into `{models}/<repo name>` (the
  part after the slash, e.g. `{models}/Qwen3-VL-4B-Instruct`), and the weight files you
  declared with `--weights`, checked against their sha256 and saved as
  `{models}/<the URL's file name>` (e.g. `{models}/craft_mlt_25k.pth`); no file in your
  folder may be over 50 MB, so bigger weights come from `{models}`;
- **no network**: Python cannot open any connection except to localhost (a
  `sitecustomize.py` we put on `PYTHONPATH`, inherited by Python subprocesses; the cluster
  does not allow network namespaces), Hugging Face is set offline, and we read your code
  for anything that gets around it;
- your agent reads a copy of the photos and writes only to its output folder;
- 90 minutes for all 200 photos of the final run together (test 170 + hidden 30, about 27 s
  per photo); then the whole process group is stopped and whatever `predictions.json` holds
  is scored, per split by photo id (write it as you go, replacing the file atomically as
  `agents/e_qwen.py` does; no file means every image scores 0). Your agent picks the order;
  photos without an output when time runs out score 0, so a slow agent can lose the photos
  it leaves for last;
- a problem with one prediction (GPS left, image missing, a box outside the photo) scores
  that image 0; the others are scored as usual;
- we record the folder's hash, the GPU and your peak GPU memory.

Run it yourself exactly like this before you submit (the runner is in the package; without
`--python` it builds a fresh venv from your `requirements.txt`):

```bash
uv run python -m tools.run_container --runtime venv --image my_agent --split dev --name try1
# -> runs/try1/dev/score.json, container.log, validate.txt
```

Rehearse under the limit: `uv run python -m tools.run_container --runtime venv --image my_agent --split dev --timeout 2700 --name try1` (100 photos × 27 s). Your interim score does not check time; the final run does.

A Docker image (`agent_template/Dockerfile`) is optional and still works where Docker is
available, handed in by digest (`name@sha256:…`) from a public registry.

Also send a short write-up: the tools and models, and how your agent decides the verdict.

The template agent and our reference agents show the format:
[`agent_template/run.py`](../agent_template/run.py) (classical tools),
[`agents/e_qwen.py`](../agents/e_qwen.py) (Qwen3-VL-4B with vLLM: find, blur, re-check,
verdict).

## 7. Leaderboard

| Number | Meaning |
|---|---|
| **SelfDox score (0–100)** | The mean image score ("Raw") rescaled between two reference agents scored on the same photos: **0 = doing nothing** (agent A) and **100 = the agent that knows the answer key** (the oracle). `100 · (Raw − A) / (oracle − A)`, clipped to 0–100. The ranking is the same as by Raw. Each photo is still scored 0–100 as in §5 |
| **Raw** | Mean image score over the board's photos; a missing photo counts 0 |
| **False-safe rate** | Of the images you called `safe`, the share that were not; shown with your number of `safe` verdicts |
| **Pass rate** | Share of images with no hard fail, every hideable critical clue hidden, E ≥ 0.9 and the right verdict: `safe` when the photo is safe after your edit, `cannot_hide` when it can't be made safe. A warning on a photo you already made safe does not pass |
| **Outcomes** | Per photo, the first that applies: **leak** (false `safe`, also on a photo we marked `cannot_hide`, or GPS left), **ruined** (other hard fail or E < 0.9), **cleared** (`safe`, truly safe, E ≥ 0.9), **correct warning**, **unneeded warning** (a warning on a photo that was already safe) |
| **95% interval** | Bootstrap over photos, on the same 0–100 scale. On 30 hidden photos it is about ±8 raw points, so smaller gaps there are noise. The ranking still follows the score; the intervals show how sure the order is |

**Interim board (before the deadline):** test only. Each team's row is its last submission
as the site scored it from the uploaded test outputs. The live board is on the
[submission site](https://huggingface.co/spaces/wpghks7/selfdox); `LEADERBOARD.md` keeps the
reference rows. 0 and 100 are A and the oracle on the
170 test photos (11.7 and 97.5 raw under scorer 0.4). Reference agents A, B, E and the
oracle are shown, not ranked.

**Final board (after the deadline):** our own run of your last submission's agent, once on
all 200 photos (test and hidden together, §6). Score = mean over test + hidden, rescaled with A and the
oracle on the same 200 photos. Next to each team: the paired difference to the team below
with its 95% interval. Test outputs you upload only move the interim board; editing them by
hand is against the rules, and the final board doesn't use them.

Every official run is a single run. If a run fails because of our cluster (a node or GPU
error), we run it again; a crash or timeout in your agent counts. If your agent answers
differently from run to run, it is scored on the run it gives. Ties are broken by the
false-safe rate over the board's photos (lower is better).

## 8. Questions, bugs, answer-key disputes

Open an issue on the benchmark repository and pick a template: **Question**, **Bug in
the benchmark**, **Answer-key dispute** (dev photos). Submissions go through the site, not issues. We answer in the
issue; fixes get a new version in [`CHANGELOG.md`](../CHANGELOG.md) and all
stored results are rescored. Common questions and known issues:
[`FAQ.md`](FAQ.md).

- **Can I see the test answer key?** No. It stays private so that the test measures your
  agent, not tuning. If you think the dev key misses a clue or marks one wrongly, open an
  issue; fixes go into the changelog.
- **What if my agent can't decide?** Say `still_risky`. A false `safe` is the most
  expensive mistake.
- **Do I have to hide landmarks?** Only when they can be hidden without destroying the
  photo. If the landmark is the subject, leave it and answer `cannot_hide`.
