# Changelog

Every change that can move a score bumps the scorer version or the answer-key date, and every stored result is rescored (`tools/rescore.py`). Scores from different versions are not comparable.

## v1.2.1 · time limit · 2026-11-14

- **One final run, one clock:** after the deadline we run your last agent once, offline, on all 200 photos (test 170 + hidden 30) in one folder, within 90 minutes on one 24 GB GPU, about 27 s per photo (was: 90 minutes for the 30 hidden photos and the same time per photo on test, in two runs). Your agent picks the order; photos without an output when time runs out score 0. The one output is scored per split by photo id, and only this run ranks.
- **Rehearse under the limit:** `uv run python -m tools.run_container --runtime venv --image my_agent --split dev --timeout 2700 --name try1` (100 photos × 27 s). Your interim score does not check time; the final run does.
- `tools/run_container.py --split final` runs the test and hidden photos together under one `--timeout` (default 5400 s) and writes `test/score.json` and `hidden/score.json` for the final board.
- Reference agents A–D (`agents/common.py`) write `predictions.json` atomically after every photo, as the template does; their outputs and scores are unchanged.
- Scorer, answer key and format are unchanged (scorer 0.4).

## v1.2 · submission site · 2026-11-13

- **Submissions go through a site now:** <https://huggingface.co/spaces/wpghks7/selfdox>. Sign in with Hugging Face (a free account), enter your team code and upload `submission_<team>.zip` (at most 30 MB) and `test_outputs_<team>.zip` (at most 300 MB). The upload becomes a pull request on the dataset `wpghks7/selfdox-submissions`; the files are public there, as they were in issues.
- **Team codes:** each team gets a code from the organizers; any member can submit from their own Hugging Face account with it. The code is never published.
- **Scoring every 5 minutes:** a GitHub workflow checks new submissions, applies the rules, scores, comments the totals on the pull request, closes it and updates the leaderboard on the site. A score appears about 5 to 10 minutes after the upload.
- **Retired:** the Submission issue template, the issue bot (`score-submission.yml`) and the "one GitHub account per team" rule. `LEADERBOARD.md` keeps the reference rows and points to the site.
- Rules, scorer and answer key are unchanged (scorer 0.4): 5 scored submissions per team, any open-weight models up to 4B, offline, no hand edits; final board = our runs of each team's newest scored agent on test + hidden.
- `selfdox.submit`: the zip limit is 30 MB and the last message points to the site.

## v1.1 · scorer 0.4 · 2026-11-13

- **Clue report R is by area, not by counting boxes.** Recall = how much of each answer-key clue the reported boxes cover (critical clues count double; `exif_gps` counts when reported). Precision = how much of the reported box area lies on a clue, with the key boxes dilated like the edit economy (10%, at least 8 px), so a small margin is free. R = their F1. Several small boxes over one clue score like one tight box; a full-photo box scores close to 0. Only boxes count for precision: on a photo whose only key clue is `exif_gps`, any reported box makes R 0. The IoU, coverage and area-ratio thresholds are gone. Weights are unchanged (65 / 15 / 20).
- **Every stored result was rescored.** Test: E 63.6 → 65.0, B 22.2 → 22.3, A 11.7 and the oracle 97.5 unchanged. Dev: E 60.5 → 62.8. Scores from scorer 0.3 are not comparable.
- **Leaderboard on a 0–100 scale:** 0 = agent A (does nothing), 100 = the oracle (knows the key), both on the board's own photos; `100 · (Raw − A) / (oracle − A)`, clipped. The raw mean is shown next to it, and the ranking is the same. Per-photo scores are unchanged. Interim board (test): A = 11.7, oracle = 97.5 raw; E = 62.2.
- **Submission bot:** a Submission issue now carries the agent zip and a link to your test outputs. A GitHub workflow scores the outputs in about 10 minutes, comments with totals only (0–100 score and interval, raw score, n/5, agent sha256, scorer version) and updates `LEADERBOARD.md`. The test key is in the repository only encrypted.
- **`selfdox.submit`:** `--model` can be given several times (one per model, `repo@revision`); `--weights "<url>#sha256=<hex>"` declares a weight file that is not on Hugging Face (we download it into `{models}`); `--contact` is optional; `--test-sub` also writes `test_outputs_<team>.zip` for the bot.
- `predictions.json` keeps `"benchmark_version": "0.3"`: the answer key and the format did not change.

## Rules · 2026-11-13 (v1.1)

- **Submissions:** at most 5 per team in total; only scored submissions count, and one issue is one submission. Your last submission before the deadline is the one on the final board; after the deadline the bot refuses new ones. This replaces "one per team per day".
- **One GitHub account per team:** the first account that submits for a team name owns it, and one account submits for one team; to submit from several accounts, ask in a Question issue. Contact in the Submission issue is optional (the issue is public).
- **Models:** any open-weight model or models, each at most 4B parameters, declared as Hugging Face `repo@revision`; no fine-tuning; offline. The list may later narrow to specific models (pinned issue). `requirements.txt` pins its versions.
- **Interim board:** the bot's score of your uploaded test outputs (was: our run of each submission on test). The final board is unchanged: our own runs of your last submission's agent, once on test and once on hidden, after the deadline.

## Rules · 2026-11-13

- **Final runs and failures:** before the deadline each submission is run once on test (interim board). After the deadline each team's last submission is run once on test and once on hidden, and the final board uses those two runs. A run that fails because of our cluster (a node or GPU error) is run again; a crash or timeout in the agent counts. No score changes.
- **Leaderboard:** the pass rate on hidden is shown as "Hidden pass", ties go to the lower false-safe rate over the board's photos (test on the interim board, test + hidden on the final board), and before the hidden runs the board is test only.
- **Submissions:** at most one per team per day (Korean time); a second one on the same day is not run and doesn't count.

## Rules · 2026-11-12

- **Official runs:** test and hidden are each run once, and that run counts (was: the mean of three hidden runs). No score changes: the scorer and the answer key are unchanged.

## v1.0 · scorer 0.3 · answer key 2026-11-10 (draft)

- **Data:**
  - 300 Commons photos split by author: dev 100 (key public), test 170 (key private), spare 30. About 30 hidden phone photos follow.
  - The 20-photo v0.2 pilot is part of dev.
- **Answer key (draft):**
  - 1,059 clues on 300 photos.
  - 415 clues that Claude's first pass described but nobody had boxed were boxed by Qwen3-VL-8B.
  - A text clue's box now covers all of its OCR lines.
  - People are still checking the clues.
- **Scorer 0.3:**
  - A photo a person marked `cannot_hide` is unhideable in scoring.
  - The edit score is max(0, 1 − 3·(1 − E)).
  - A pass needs the right verdict for the photo's true state.
- **Reporting:** per-photo outcomes (cleared, leak, ruined, correct or unneeded warning), results per photo group, bootstrap 95% intervals, hard-failed image ids.
- **Tools:**
  - `selfdox.view` (HTML report)
  - `selfdox.submit` (check and pack)
  - `tools/run_container.py --runtime venv` (the official runner)
  - `selfdox.score` zeroes a photo whose prediction fails validation, as the official runs do.
- **Template:**
  - the reference verdict rule
  - `exif_gps` reported
  - `predictions.json` rewritten after every photo
  - `--input/--output` accepted by every agent

## v0.2 · scorer 0.2 (pilot, 2026-11)

- **Data:** 20-photo pilot.
- **Scorer:** model-free pixel test, PU/R/V, pass rate and pass^k. See `docs/design.md` §1.
