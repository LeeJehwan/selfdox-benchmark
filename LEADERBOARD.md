# SelfDox leaderboard

Scorer 0.4. Updated 2026-11-13. **Interim board:** test only (170 photos). Each row is the team's last submission, scored by the submission bot on the team's own test outputs. After the deadline we run each team's last agent once on test and once on hidden for the final board. A missing photo counts 0. How to get on it: [`docs/participants.md`](docs/participants.md) §6.

Score = the mean image score (“Raw”, each photo 0–100) rescaled so that **0 = doing nothing** (reference agent A) and **100 = the agent that knows the answer key** (the oracle), both on the same photos; clipped to 0–100. The ranking is the same as by Raw.

*No team submissions yet. The reference agents below show the range.*

| # | Team | Score 0–100 (95% CI) | Raw | Cleared | Leaks | False-safe (# safe) | Pass | Submissions | Scored |
|---|---|---|---|---|---|---|---|---|---|
|  | A · do nothing (reference) | **0.0** (0.0–5.7) | 11.7 | 20% | 80% | 0.80 (170) | 20% | – | – |
|  | B · strip GPS (reference) | **12.4** (5.6–20.0) | 22.3 | 33% | 67% | 0.67 (170) | 33% | – | – |
|  | E · Qwen3-VL-4B (reference) | **62.2** (56.8–67.4) | 65.0 | 29% | 15% | 0.29 (86) | 30% | – | – |
|  | Oracle · knows the key (reference) | **100.0** (98.8–100.0) | 97.5 | 71% | 1% | 0.02 (124) | 97% | – | – |

0 = A’s mean on these photos (11.7 raw), 100 = the oracle’s (97.5 raw). Reference rows are not ranked. Cleared = `safe`, truly safe and E ≥ 0.9; leak = a false `safe` or GPS left (`docs/design.md` §7.5). Ties go to the lower false-safe rate.
