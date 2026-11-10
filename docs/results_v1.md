# Results (v1.0)

> **Draft.** Scored against the draft answer key (before the human review); the numbers will move. Rerun on the final key.

Scorer 0.3. Reproduce: run the agents in `agents/`, score each split with `python -m selfdox.score --out results_<split>_<agent>.json`, then `python -m tools.report --v1 <dir>`.

## dev (100 photos)

| Agent | Score | Pass rate | False-safe rate (# safe) | L | E | PU | R | V | Hard fails |
|---|---|---|---|---|---|---|---|---|---|
| A_noop | **10.3** | 0.14 | 0.86 (100) | 0.00 | 1.00 | 0.00 | 0.10 | 0.10 | metadata=22 |
| B_exif_strip | **21.3** | 0.26 | 0.74 (100) | 0.00 | 1.00 | 0.00 | 0.24 | 0.21 |  |
| C_blur_all | **13.3** | 0.00 | 0.41 (100) | 0.84 | 0.24 | 0.00 | 0.10 | 0.58 |  |
| E_qwen | **60.5** | 0.21 | 0.33 (46) | 0.82 | 0.87 | 0.67 | 0.46 | 0.56 |  |
| template | **21.4** | 0.26 | 0.74 (100) | 0.00 | 1.00 | 0.00 | 0.24 | 0.21 |  |
| oracle | **96.6** | 0.91 | 0.00 (71) | 1.00 | 0.97 | 0.93 | 1.00 | 1.00 |  |

| Agent | easy | medium | hard | no_clue | cannot_hide | exif_only |
|---|---|---|---|---|---|---|
| A_noop | 0.0 | 0.0 | 0.0 | 100.0 | 1.1 | 0.0 |
| B_exif_strip | 0.0 | 0.0 | 7.4 | 99.9 | 1.1 | 99.9 |
| C_blur_all | 14.3 | 17.2 | 12.1 | 35.0 | 0.0 | 22.1 |
| E_qwen | 68.3 | 58.2 | 62.9 | 66.3 | 42.8 | 88.4 |
| template | 0.0 | 0.0 | 7.5 | 100.0 | 1.1 | 100.0 |
| oracle | 93.7 | 96.0 | 96.0 | 100.0 | 97.1 | 100.0 |

| Agent | Cleared | Leaks | Unneeded warnings | Correct warnings | Ruined | Leaks among `safe` | Score 95% CI | Cleared 95% CI | Leaks 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| A_noop | 14 (14%) | 86 (86%) | 0 (0%) | 0 (0%) | 0 (0%) | 0.86 (100) | 5.2–16.3 | 8%–21% | 79%–92% |
| B_exif_strip | 26 (26%) | 74 (74%) | 0 (0%) | 0 (0%) | 0 (0%) | 0.74 (100) | 13.5–29.3 | 17%–35% | 65%–83% |
| C_blur_all | 0 (0%) | 41 (41%) | 0 (0%) | 0 (0%) | 59 (59%) | 0.41 (100) | 11.1–15.6 | 0%–0% | 32%–50% |
| E_qwen | 19 (19%) | 15 (15%) | 18 (18%) | 18 (18%) | 30 (30%) | 0.33 (46) | 55.1–65.9 | 12%–27% | 8%–23% |
| template | 26 (26%) | 74 (74%) | 0 (0%) | 0 (0%) | 0 (0%) | 0.74 (100) | 13.6–29.4 | 17%–35% | 65%–83% |
| oracle | 63 (63%) | 0 (0%) | 0 (0%) | 28 (28%) | 9 (9%) | 0.00 (71) | 95.0–97.9 | 54%–72% | 0%–0% |

| Agent | needs_edit (score · cleared · leaks) | leave_alone | unhideable |
|---|---|---|---|
| A_noop | 0.0 · cleared 0/45 · leaks 45 | 38.5 · cleared 14/26 · leaks 12 | 1.0 · cleared 0/29 · leaks 29 |
| B_exif_strip | 0.0 · cleared 0/45 · leaks 45 | 80.9 · cleared 26/26 · leaks 0 | 1.0 · cleared 0/29 · leaks 29 |
| C_blur_all | 14.5 · cleared 0/45 · leaks 12 | 26.1 · cleared 0/26 · leaks 0 | 0.0 · cleared 0/29 · leaks 29 |
| E_qwen | 63.5 · cleared 5/45 · leaks 0 | 73.1 · cleared 14/26 · leaks 0 | 44.6 · cleared 0/29 · leaks 15 |
| template | 0.0 · cleared 0/45 · leaks 45 | 81.0 · cleared 26/26 · leaks 0 | 1.0 · cleared 0/29 · leaks 29 |
| oracle | 94.8 · cleared 38/45 · leaks 0 | 99.2 · cleared 25/26 · leaks 0 | 97.2 · cleared 0/29 · leaks 0 |

## test (170 photos)

| Agent | Score | Pass rate | False-safe rate (# safe) | L | E | PU | R | V | Hard fails |
|---|---|---|---|---|---|---|---|---|---|
| A_noop | **11.7** | 0.20 | 0.80 (170) | 0.00 | 1.00 | 0.00 | 0.10 | 0.12 | metadata=37 |
| B_exif_strip | **22.2** | 0.33 | 0.67 (170) | 0.00 | 1.00 | 0.00 | 0.25 | 0.22 |  |
| E_qwen | **63.6** | 0.30 | 0.29 (86) | 0.81 | 0.89 | 0.69 | 0.52 | 0.55 |  |
| oracle | **97.5** | 0.97 | 0.02 (124) | 0.98 | 0.98 | 0.96 | 1.00 | 0.98 |  |

| Agent | easy | medium | hard | no_clue | cannot_hide | exif_only |
|---|---|---|---|---|---|---|
| A_noop | 0.0 | 0.0 | 9.8 | 100.0 | 0.7 | 0.0 |
| B_exif_strip | 0.0 | 0.0 | 14.9 | 99.9 | 1.9 | 99.9 |
| E_qwen | 70.0 | 65.6 | 51.4 | 80.4 | 46.5 | 96.3 |
| oracle | 96.6 | 97.5 | 94.6 | 100.0 | 98.0 | 100.0 |

| Agent | Cleared | Leaks | Unneeded warnings | Correct warnings | Ruined | Leaks among `safe` | Score 95% CI | Cleared 95% CI | Leaks 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| A_noop | 34 (20%) | 136 (80%) | 0 (0%) | 0 (0%) | 0 (0%) | 0.80 (170) | 7.4–16.6 | 14%–26% | 74%–86% |
| B_exif_strip | 56 (33%) | 114 (67%) | 0 (0%) | 0 (0%) | 0 (0%) | 0.67 (170) | 16.4–28.7 | 26%–40% | 60%–74% |
| E_qwen | 50 (29%) | 25 (15%) | 39 (23%) | 24 (14%) | 32 (19%) | 0.29 (86) | 58.9–68.2 | 23%–36% | 10%–20% |
| oracle | 121 (71%) | 2 (1%) | 0 (0%) | 45 (26%) | 2 (1%) | 0.02 (124) | 96.5–98.4 | 64%–78% | 0%–3% |

| Agent | needs_edit (score · cleared · leaks) | leave_alone | unhideable |
|---|---|---|---|
| A_noop | 0.0 · cleared 0/68 · leaks 68 | 34.9 · cleared 34/56 · leaks 22 | 0.7 · cleared 0/46 · leaks 46 |
| B_exif_strip | 0.0 · cleared 0/68 · leaks 68 | 65.8 · cleared 56/56 · leaks 0 | 1.9 · cleared 0/46 · leaks 46 |
| E_qwen | 65.9 · cleared 14/68 · leaks 3 | 74.8 · cleared 36/56 · leaks 0 | 46.5 · cleared 0/46 · leaks 22 |
| oracle | 96.0 · cleared 65/68 · leaks 2 | 98.9 · cleared 56/56 · leaks 0 | 98.0 · cleared 0/46 · leaks 0 |

Outcomes per photo, first match wins: **leak** = false `safe` (also on a photo a person marked `cannot_hide`) or GPS left; **ruined** = other hard fail or E < 0.9; **cleared** = `safe`, truly safe, E ≥ 0.9; **correct warning** = `still_risky`/`cannot_hide` on a photo that is not safe; **unneeded warning** = a warning on a photo that was already safe. Groups: **needs_edit** has a hideable critical clue, **unhideable** is expected `cannot_hide` or has a critical clue that can't be hidden, **leave_alone** is the rest. Intervals: bootstrap over photos (2,000 resamples, seed 0).

## Attacker check

Qwen3-VL-8B guesses where each photo was taken (`tools/attack_report.py`).

| run | median error (km) | ≤1 km | ≤25 km | ≤200 km | city | district |
|---|---|---|---|---|---|---|
| original | 5.8 | 13% | 70% | 72% | 74% | 28% |
| gt_blurred | 6.3 | 12% | 67% | 69% | 71% | 25% |
| gt_blurred_k2 | 6.8 | 10% | 65% | 65% | 68% | 23% |
| e_qwen | 11.2 | 7% | 54% | 54% | 55% | 19% |
| e_qwen · passed | 3715.6 | 6% | 30% | 30% | 30% | 15% |
| e_qwen · failed | 8.7 | 8% | 61% | 62% | 63% | 21% |
| e_qwen · cleared | 3715.6 | 5% | 29% | 29% | 29% | 14% |
