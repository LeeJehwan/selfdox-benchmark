# SelfDox FAQ

Can't find your answer here? Open a **Question** issue. We answer within two working days and add common questions to this page.

## Data

**Where are the photos?**
- `data/dev/images`: 100 photos, with the answer key `data/dev/gt.json`.
- `data/test/images`: 170 photos, without a key.
- The hidden set: about 30 phone photos our team took. Nobody outside the team sees them, and we run your agent on them ourselves.

**Is the answer key final?**
- No. It is a reviewed draft of 2026-11-10.
- Every clue went through an AI first pass. Boxes Claude described in words were drawn by Qwen3-VL-8B. People are checking every clue now; the count is in `docs/dataset_card.md`.
- If you see a missing clue, a wrong box or a wrong `hideable`/`severity`, open an **Answer-key dispute**.
- We fix the key and bump the version (`CHANGELOG.md`), then rescore every stored result.
- The test and hidden keys are frozen before the final run.

**What will the hidden photos look like?** The same format as the others:
- JPEG, long side 2048 px, sRGB
- no orientation tag
- faces and plates blurred
- some with GPS in the EXIF

## Building your agent

**May I tune prompts, thresholds or code on dev?** Yes. That is what dev is for. Not allowed: training or fine-tuning model weights, and editing outputs by hand (`docs/participants.md` §3).

**Which models and tools may I use?**
- Any open-weight model or models, each at most 4B parameters (Qwen3-VL-4B-Instruct, our agent E, is allowed). No fine-tuning. Offline: no web search, no online API.
- Declare each model as Hugging Face `repo@revision` (`selfdox.submit --model`, once per model). We download exactly those for the final run.
- Classical tools such as OCR, detectors and blur or inpainting filters are fine.
- The list may later narrow to specific models; that would be announced in the pinned issue.
- If you are unsure about a model or tool, ask in a Question issue before you build on it.

**My agent says `still_risky` on a photo it already cleaned. Why did that cost points?**
- `still_risky` on a photo that is safe after your edit is an *unneeded warning*. It gets half the verdict points and no pass.
- Say `safe` when you hid everything you found and a second look finds nothing new.
- Say `cannot_hide` when the place itself is the subject (a palace, a skyline).

**What decides whether a photo is "safe"?**
- Only the **critical** clues and GPS. Supporting clues raise L, but leaving one doesn't make the photo unsafe.
- A photo our annotators marked `cannot_hide` is unhideable however much you blur.

**I hid nothing on a `cannot_hide` photo and answered `cannot_hide`. Why only partial verdict points?**
- On photos that also have hideable clues, the verdict points are multiplied by L, the share of clues you hid.
- So also blur the shop signs and plates; the right verdict for the photo stays `cannot_hide`.

**My landmark blur didn't count as hidden.**
- Landmarks, skylines and facades are judged on coarse structure, which reaches past a tight box.
- Blur with a margin of about a quarter of the box's short side.
- The margin costs a little edit economy; leaving the landmark recognizable costs more.

**How are my clue boxes scored (R)?**
- By area, not by counting boxes (scorer 0.4).
- Recall: how much of each answer-key clue your boxes cover (critical clues count double).
- Precision: how much of your box area lies on a clue. A margin of about 10% around a clue (at least 8 px) is free.
- R is the F1 of the two. Several small boxes over one sign score like one tight box; a box over the whole photo scores close to 0.
- On a photo with no clue in the key, R is 1 only if you report nothing.
- Only boxes count for precision. On a photo whose only key clue is `exif_gps` (no box), any box you report lies off the clues, so R is 0 there; report a box only for something you see. A clue without a box other than `exif_gps` counts neither way.

**Do I have to report `exif_gps`?** Yes, when the input had GPS: report `{"bbox": null, "type": "exif_gps"}`. Saving the photo without metadata removes it. GPS left in the file scores the photo 0.

**Can my agent import `selfdox`?**
- The official venv has only what your `requirements.txt` installs.
- If you want our code, copy the parts you need into your folder (MIT license).

## Running and submitting

**How exactly will you run my agent?**
- After the deadline we run your last submission's agent folder once, offline, on all 200 photos (test 170 + hidden 30) in one folder, within 90 minutes on one 24 GB GPU, with `tools/run_container.py --runtime venv --split final`; you have the same runner. Photos without an output when time runs out score 0; your agent picks the order. Only this run ranks.
- It creates a fresh Python 3.12 venv, installs your `requirements.txt`, copies the photos and runs `python run.py --input <photos> --output <folder>` (or your `selfdox_run.json` command).
- Each run has one GPU, no network and a time limit. When the limit hits, the whole process group is stopped.
- Run it yourself on dev to see what we will see.

**Do I need to send a Docker image?** No. You send your agent folder; we run it in our environment, the same for every team: a fresh Python 3.12 venv with your pinned `requirements.txt`, the models you declared downloaded at their revision, one 24 GB GPU, offline, the same time limit. The lab cluster can't run Docker anyway. A Docker image by digest stays optional.

**How do you cut the network?** The lab cluster doesn't allow containers or network namespaces, so:
- a `sitecustomize.py` refuses every connection that isn't to localhost, in your agent and in every Python process it starts;
- Hugging Face is set offline;
- we read your code.

Load weights from `{models}` or ship small ones in your folder (no file over 50 MB). Bigger weights that are not on Hugging Face (EasyOCR's detector, for instance): declare them with `selfdox.submit --weights "<https url>#sha256=<hex>"`, and we download and check them into `{models}`, under the URL's file name (`…/craft_mlt_25k.pth` → `{models}/craft_mlt_25k.pth`). Hugging Face models land in `{models}/<repo name>`. Turn the library's own downloads off (EasyOCR: `model_storage_directory=<models>`, `download_enabled=False`).

**What if my agent is slow or crashes halfway?**
- Write `predictions.json` as you go, replacing the file atomically (`agent_template/run.py` does this after every photo).
- What the file holds when the run stops is scored; photos it lacks count 0.
- A problem with one photo (GPS left, a missing image, a box outside the photo) scores only that photo 0.

**Time and compute budget?**
- Final run: 90 minutes for all 200 photos together (test 170 + hidden 30) on one 24 GB GPU, about 27 s per photo.
- Rehearse under the limit: `uv run python -m tools.run_container --runtime venv --image my_agent --split dev --timeout 2700 --name try1` (100 photos × 27 s). Your interim score does not check time; the final run does.
- You may report your model calls, tokens and seconds per photo in each prediction's optional `meta` field.

**How do I submit?**
- Run your agent on `data/test/images`, then `uv run python -m selfdox.submit ... --smoke 3 --test-sub <outputs>`. It writes `submission_<team>.zip` (your agent folder) and `test_outputs_<team>.zip` (your test outputs).
- Go to the submission site, <https://huggingface.co/spaces/wpghks7/selfdox>, sign in with Hugging Face (a free account), enter your team code and upload both files: `submission_<team>.zip` (at most 30 MB) and `test_outputs_<team>.zip` (at most 300 MB). The upload becomes a pull request on the dataset `wpghks7/selfdox-submissions`; the files are public there.
- Team codes: the organizers give each team a code. Any member can submit from their own Hugging Face account with it. The code is never published. Lost it? Ask in a Question issue.
- A workflow checks new submissions every 5 minutes, scores your test outputs, comments the total score on the pull request, closes it and updates the leaderboard on the site. Expect a score 5 to 10 minutes after the upload.
- **At most 5 submissions per team, in total.** Only scored submissions count; rejected or failed ones do not. To submit again, upload again.

**What counts on the leaderboard?**
- The board score runs from 0 to 100: **0 = doing nothing** (agent A), **100 = the agent that knows the answer key** (the oracle), both on the board's photos. In between it is linear in the mean image score ("Raw"), so the order is the same as by Raw. Each photo is still scored 0–100 as before.
- Interim board (before the deadline): the site's score of your uploaded test outputs, test only, your last submission.
- Final board (after the deadline): our own run of your last submission's agent, once on all 200 photos (test and hidden together). Score = mean over test + hidden, rescaled with A and the oracle on the same photos; a missing photo counts 0.
- Uploaded test outputs only move the interim board. The final board doesn't use them.
- Even at temperature 0 a model can answer differently from run to run (vLLM batching); that is part of your agent's result.
- If a run fails because of our cluster (a node or GPU error), we run it again. A crash or timeout in your agent counts.

**How does the scoring workflow keep the test answer key secret?**
- The key is in the repository only encrypted (`data/test/gt.json.enc`, AES-256). The passphrase is a repository secret that only the scoring workflow can read.
- The workflow decrypts the key into its own temporary folder, scores, and comments with totals only: no per-photo numbers.
- With 5 submissions per team, totals say very little about single photos. And the final board is our own run of your agent, on test and on 30 hidden photos nobody outside our team has seen.

**How precise are the scores?**
- On 170 test photos the standard error is about 1.7 raw points (about 2 on the 0–100 board scale); on 30 hidden photos about 4.
- The leaderboard shows 95% intervals on the 0–100 scale. The final board also shows the paired difference to the next team; if that interval includes 0, the two are not separable. The ranking still follows the score; the intervals show how sure the order is.

## Known issues

- **The answer key is not finished.** Clues are still being checked by people. In a dev-key spot check, the oracle (blur exactly the key's clues) scored 96.4. The rest came from 9 photos where the reference blur did not hide a key clue and 1 over-edit. Since then we fixed text boxes that did not cover their OCR lines and gave landmarks a wider margin. The remaining cases are on the review list.
- **The attacker check is a lower bound.** Qwen3-VL-8B is a weak geolocator. "The attacker can't find it" is not proof that a photo is safe, and it is not part of the score.
- **Network isolation is Python-level.** A program that isn't Python could get around it. Doing so on purpose is against the rules, and we read every agent's code.
