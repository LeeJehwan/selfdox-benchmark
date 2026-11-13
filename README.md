# SelfDox

**Can your agent find what gives away where a photo was taken, hide only that, and say honestly whether the photo is now safe to post?**

Each photo is a street, shop, transit or landmark photo from Seoul or Busan. Your agent returns three things:

- the location clues it found: shop signs, phone numbers, road-name plates, station names, landmarks, GPS in the EXIF
- the photo with those clues hidden
- a verdict: `safe`, `still_risky` or `cannot_hide`

No model runs at scoring time. A pixel test checks each answer-key clue against your edited photo.

## Start in 5 minutes

Needs [uv](https://docs.astral.sh/uv/) (Python 3.12+). Everything runs on a laptop CPU.

```bash
git clone https://github.com/LeeJehwan/selfdox-benchmark && cd selfdox-benchmark
uv sync

# 1. run the template agent on the 100 dev photos
uv run python agent_template/run.py --input data/dev/images --output outputs/dev
# 2. check the output format
uv run python -m selfdox.validate --sub outputs/dev --inputs data/dev/images
# 3. score it, then see why each photo scored what it did
uv run python -m selfdox.score --gt data/dev/gt.json --sub outputs/dev --out results.json
uv run python -m selfdox.view --gt data/dev/gt.json --sub outputs/dev --results results.json --out view
#    open view/index.html: green = hidden, red = still recoverable, grey = leave it
```

Then replace `find_clues()` in `agent_template/run.py` with your detector. Or start from `agent_template/run_vlm.py`, our Qwen3-VL-4B agent E (vLLM, one GPU).

## What's here

| Path | What |
|---|---|
| `data/dev/` | 100 photos and their answer key (`gt.json`). Build and tune on these. |
| `data/test/` | 170 photos, no answer key |
| `agent_template/` | Start here: `run.py` (classical tools) and `run_vlm.py` (Qwen3-VL-4B) |
| `agents/` | Reference agents: A (do nothing), B (strip GPS), C (blur all), D (OCR + blur), E (Qwen3-VL-4B), oracle (knows the key) |
| `selfdox/` | `validate`, `score`, `view` (HTML report), `submit` (check and pack) |
| `tools/run_container.py` | The exact runner we use for the official runs, so you can run it yourself |
| `docs/participants.md` | **The guide:** task, format, scoring, rules, what to hand in |
| `docs/FAQ.md` | Common questions and known issues |
| `LEADERBOARD.md` · `CHANGELOG.md` | Rankings and every change to the scorer or answer key |
| `leaderboard/` · `.github/workflows/` | The 0/100 anchors (`refs.json`) and the workflow that scores submissions from the site |

## How you are scored, in short

| Part | Question | Weight |
|---|---|---|
| **PU** | Did you hide the clues (L), and leave the rest of the photo alone (edit score from E)? Over-editing costs three times its share. | 65% |
| **R** | Did you box the clues? By area: how much of each clue your boxes cover, and how much of your boxes is clue. | 15% |
| **V** | Is your verdict right? A false `safe` costs another 10 points. | 20% |

These score 0 for a photo:
- GPS left in the file
- a hidden copy of the photo inside the file
- the wrong image size
- a box outside the photo

Next to the score you get the **pass rate** and what each photo means to a user:
- **cleared:** the user can post it
- **leak**
- **ruined:** too much blurred
- **correct warning** / **unneeded warning**

Details: [`docs/participants.md`](docs/participants.md) §5.

On the leaderboard the mean over photos is shown on a **0–100 scale: 0 = doing nothing, 100 = the agent that knows the answer key** (reference agents A and the oracle on the same photos). The order is the same as by the raw mean.

## Get on the leaderboard

Rules: any open-weight model(s), each at most 4B parameters; no fine-tuning; offline; no hand edits; pinned `requirements.txt`; 90 minutes for the 30 hidden photos on one 24 GB GPU; **at most 5 submissions per team**. Full list: [`docs/participants.md`](docs/participants.md) §3.

```bash
# run your agent exactly as we will: a fresh venv, offline, on a copy of the photos
uv run python -m tools.run_container --runtime venv --image my_agent --split dev --name try1
# run it on the test photos, then pack: checks the folder, runs it on 3 dev photos,
# writes submission_<team>.zip (agent + metadata) and test_outputs_<team>.zip
uv run python -m selfdox.submit --agent my_agent --team <team> \
    --model "<hf repo>@<revision>" --temperature 0 --seed 0 --smoke 3 --test-sub <test outputs>
```

1. Go to the [submission site](https://huggingface.co/spaces/wpghks7/selfdox), sign in with Hugging Face (a free account), enter your team code and upload `submission_<team>.zip` (at most 30 MB) and `test_outputs_<team>.zip` (at most 300 MB). The upload becomes a pull request on the dataset `wpghks7/selfdox-submissions`; the files are public there.
2. A workflow checks new submissions every 5 minutes, scores your test outputs, comments with your score (totals only) on the pull request and updates the leaderboard on the site: the interim board. Expect it 5 to 10 minutes after the upload.
3. After the deadline we run your last submission's agent ourselves, once on test and once on the 30 hidden photos: the final board.

Only scored submissions count; at most 5 per team. Each team has a code from the organizers, and any member can submit with it from their own Hugging Face account. Deadlines are announced in class and in the pinned issue.

## Questions, bugs, answer-key disputes

Open an issue and pick a template: **Question**, **Bug in the benchmark**, or **Answer-key dispute**. Submissions go through the site, not issues.
- We answer within two working days.
- Fixes go into [`CHANGELOG.md`](CHANGELOG.md) with a new version, and stored results are rescored.
- The dev answer key is a reviewed draft. If a clue is missing, boxed wrongly or marked wrongly, a dispute helps everyone.

## Versions

| | |
|---|---|
| Release | v1.1 (scorer 0.4; data unchanged since v1.0) |
| Scorer | 0.4. `predictions.json` still says `"benchmark_version": "0.3"` (the answer-key format version). |
| Answer key | dev draft of 2026-11-10: 377 clues; people are still checking them (see `docs/dataset_card.md`) |
| Official runs | Python 3.12 venv on a lab GPU cluster: NVIDIA driver 580 / CUDA 13.0, one RTX A6000 or RTX 3090, no network |

## License

- Code: MIT ([`LICENSE`](LICENSE)).
- Answer key: CC BY-SA 4.0.
- Photos: their own Wikimedia Commons licenses. Authors and links are in `data/*/ATTRIBUTION.md`.
