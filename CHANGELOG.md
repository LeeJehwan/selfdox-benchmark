# Changelog

Every change that can move a score bumps the scorer version or the answer-key date, and every stored result is rescored (`tools/rescore.py`). Scores from different versions are not comparable.

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
