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
- The test and hidden keys are frozen before the final runs.

**What will the hidden photos look like?** The same format as the others:
- JPEG, long side 2048 px, sRGB
- no orientation tag
- faces and plates blurred
- some with GPS in the EXIF

## Building your agent

**May I tune prompts, thresholds or code on dev?** Yes. That is what dev is for. What we propose to forbid is training or fine-tuning model weights on benchmark photos, and editing outputs by hand (`docs/participants.md` §3).

**Which models and tools may I use?**
- The rule set is still marked "proposal" in `docs/participants.md` §3.
- Today's version: the only LLM/VLM is Qwen3-VL-4B-Instruct (pinned revision, official GGUF/FP8 builds), plus classical tools and small non-generative models such as OCR, detectors and OpenCV inpainting.
- The proposal adds an allow-list with a 1B-parameter cap.
- If you are unsure about a tool, ask in a Question issue before you build on it.

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

**Do I have to report `exif_gps`?** Yes, when the input had GPS: report `{"bbox": null, "type": "exif_gps"}`. Saving the photo without metadata removes it. GPS left in the file scores the photo 0.

**Can my agent import `selfdox`?**
- The official venv has only what your `requirements.txt` installs.
- If you want our code, copy the parts you need into your folder (MIT license).

## Running and submitting

**How exactly will you run my agent?**
- We use `tools/run_container.py --runtime venv`, which you have too.
- It creates a fresh Python 3.12 venv, installs your `requirements.txt`, copies the photos and runs `python run.py --input <photos> --output <folder>` (or your `selfdox_run.json` command).
- Each run has one GPU, no network and a time limit. When the limit hits, the whole process group is stopped.
- Run it yourself on dev to see what we will see.

**How do you cut the network?** The lab cluster doesn't allow containers or network namespaces, so:
- a `sitecustomize.py` refuses every connection that isn't to localhost, in your agent and in every Python process it starts;
- Hugging Face is set offline;
- we read your code.

Load weights from `{models}` or ship small ones in your folder (no file over 50 MB).

**What if my agent is slow or crashes halfway?**
- Write `predictions.json` as you go, replacing the file atomically (`agent_template/run.py` does this after every photo).
- What the file holds when the run stops is scored; photos it lacks count 0.
- A problem with one photo (GPS left, a missing image, a box outside the photo) scores only that photo 0.

**Time and compute budget?**
- Hidden: 90 minutes for about 30 photos on one 24 GB GPU (about 180 s per photo). Test uses the same per-photo time.
- A tighter budget (a mean of 120 s per photo) is proposed.
- Report your model calls, tokens and seconds per photo in each prediction's optional `meta` field.

**How do I submit?** Run `uv run python -m selfdox.submit ... --smoke 3`, open a **Submission** issue and attach the zip. If it's over 25 MB, put a link in the issue instead.

**What counts on the leaderboard?**
- Only our own runs.
- Score = the mean image score over test + hidden photos; a missing photo counts 0.
- Test and hidden are each run once, and that run counts. Even at temperature 0 a model can answer differently from run to run (vLLM batching); that is part of your agent's result.
- Your own test predictions are a format check and a cross-check against our run.

**How precise are the scores?**
- On 170 test photos the standard error is about 1.7 points; on 30 hidden photos about 4.
- The leaderboard shows 95% intervals, plus the paired difference to the next team. If that interval includes 0, the two are not separable.

## Known issues

- **The answer key is not finished.** Clues are still being checked by people. In a dev-key spot check, the oracle (blur exactly the key's clues) scored 96.4. The rest came from 9 photos where the reference blur did not hide a key clue and 1 over-edit. Since then we fixed text boxes that did not cover their OCR lines and gave landmarks a wider margin. The remaining cases are on the review list.
- **The attacker check is a lower bound.** Qwen3-VL-8B is a weak geolocator. "The attacker can't find it" is not proof that a photo is safe, and it is not part of the score.
- **Network isolation is Python-level.** A program that isn't Python could get around it. Doing so on purpose is against the rules, and we read every agent's code.
