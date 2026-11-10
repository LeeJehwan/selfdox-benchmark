# SelfDox leaderboard

Scorer 0.3. Updated 2026-11-10. Only the organizers' own runs are ranked: test (170 photos) and hidden (the phone photos; added when they are in). Score = mean image score over test + hidden, a missing photo counts 0. How to get on it: [`docs/participants.md`](docs/participants.md) §6.

*No team submissions yet. The reference agents below show the range.*

| # | Team | Score (95% CI) | Δ next (95% CI) | Test | Hidden | Cleared | Leaks | False-safe (# safe) | Pass | Hidden pass^k | s/photo | Peak GPU | Network cut | Run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  | A · do nothing (reference) | **11.7** (7.4–16.6) | – | 11.7 | – | 20% | 80% | 0.80 (170) | 20% | – | – | – | – | – |
|  | B · strip GPS (reference) | **22.2** (16.4–28.7) | – | 22.2 | – | 33% | 67% | 0.67 (170) | 33% | – | – | – | – | – |
|  | E · Qwen3-VL-4B (reference) | **63.6** (58.9–68.2) | – | 63.6 | – | 29% | 15% | 0.29 (86) | 30% | – | – | – | – | – |
|  | Oracle · knows the key (reference) | **97.5** (96.5–98.4) | – | 97.5 | – | 71% | 1% | 0.02 (124) | 97% | – | – | – | – | – |

Reference rows are scored on the test photos only (no hidden runs) and are not ranked. Cleared = `safe`, truly safe and E ≥ 0.9; leak = a false `safe` or GPS left (`docs/design.md` §7.5).
