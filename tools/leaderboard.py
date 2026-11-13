"""Build the leaderboard (Phase 3).

    uv run python -m tools.leaderboard                 # interim board, from the scoring bot
    uv run python -m tools.leaderboard --write-refs    # (maintainers) refresh refs.json
    uv run python -m tools.leaderboard --final --runs runs_final    # after the deadline

Every board shows the SelfDox score on a 0-100 scale between two reference agents scored on
the board's own photos (normalize()): 0 = agent A, which does nothing, and 100 = the oracle,
which knows the answer key. Per-photo scoring is unchanged (0-100 per photo); the board
rescales the mean image score ("Raw"), so the ranking is the raw ranking.

Interim board (default): test only. tools/bot_score.py scores each team's uploaded test
outputs and appends an entry to leaderboard/interim.json; the board shows each team's last
entry, with the reference rows and the 0/100 anchors from leaderboard/refs.json (aggregates
only, so it can be rebuilt in the public repository). No hidden numbers go on it, and it
refuses to overwrite a final board.

Final board (--final): after the deadline each team's last submission is run by us ONCE on
the test and hidden photos together, in one folder under one 90-minute clock
(tools/run_container.py --split final), which scores that one output per split into --runs:
runs/<team>/test/score.json and runs/<team>/hidden/score.json. A split with run1..runk/
(run_container --repeat k) is refused: official results are single runs
(docs/participants.md §7); <team> is team_slug() of the team's name. Every team in
interim.json must have both runs there
(run_container writes score.json even when the agent crashes, so a missing one means the
run was not made), unless --allow-missing. The reference agents come from selfdox.score
result files in --refs (results_draft_<agent>.json cut to the test photos, plus
results_hidden_<agent>.json), and the 0/100 anchors are A and the oracle on test + hidden.

Ranked by the mean image score over the board's photos (test, or test + hidden; a missing
photo counts 0); ties go to the lower false-safe rate over the same photos. The final board
also shows test and hidden separately with bootstrap 95% intervals, seconds per photo (the run's
seconds over its photos_in_run, or over the test photos for a run of test alone), peak
GPU memory, how the network was cut and the run date; "Δ next" is the paired bootstrap
difference to the team ranked just below, over the photos both were scored on (an interval
that includes 0 means the two are not separable), on the same 0-100 scale.
"""
import argparse
import datetime
import json
import re
import sys
from pathlib import Path

import numpy as np

from selfdox import config as C
from selfdox.metrics import BOOTSTRAP_N, BOOTSTRAP_SEED, aggregate, bootstrap_ci, from_dict

REF_NAMES = {"A_noop": "A · do nothing (reference)", "B_exif_strip": "B · strip GPS (reference)",
             "E_qwen": "E · Qwen3-VL-4B (reference)", "oracle": "Oracle · knows the key (reference)"}
ZERO, HUNDRED = "A_noop", "oracle"           # the agents that set 0 and 100
INTERIM = "leaderboard/interim.json"
SETTINGS = "leaderboard/settings.json"
MAX_SUBMISSIONS = 5                    # per team, in total (docs/participants.md §6)
REFS = "leaderboard/refs.json"


def normalize(s, s_a, s_o):
    """A mean image score on the board's 0-100 scale: 0 = agent A (does nothing), 100 = the
    oracle (knows the key), both on the same photos; clipped to [0, 100]."""
    if s is None:
        return None
    return float(min(100.0, max(0.0, 100.0 * (s - s_a) / (s_o - s_a))))


def scale(d, s_a, s_o):
    """A score difference on the 0-100 scale (no shift, no clip)."""
    return 100.0 * d / (s_o - s_a)


def _rows(doc):
    return doc["runs"][0]["images"] if "multi_run" in doc else doc["images"]


def load_team(team_dir):
    """{"test": {id: score row}, "hidden": {id: score row}, meta}: one official run per split."""
    team_dir = Path(team_dir)
    out = {"name": team_dir.name, "test": {}, "hidden": {}, "meta": {}}
    for split in ("test", "hidden"):
        path = team_dir / split / "score.json"
        repeats = sorted((team_dir / split).glob("run*/score.json"))
        doc = json.loads(path.read_text()) if path.exists() else {}
        if repeats or "multi_run" in doc:
            raise SystemExit(f"{team_dir / split}: {len(repeats) or 'several'} runs (--repeat); "
                             "official results are single runs (docs/participants.md §7). "
                             f"Keep one run as {split}/score.json and move the rest out.")
        if not doc:
            continue
        out[split] = {r["id"]: r for r in doc.get("images", [])}
        out["meta"][split] = {k: doc.get(k) for k in ("seconds", "peak_gpu_mib", "isolation",
                                                       "status", "image_digest", "photos_in_run")}
        out["meta"][split]["date"] = datetime.date.fromtimestamp(path.stat().st_mtime).isoformat()
    return out


def reference_rows(refs_dir, test_ids, hidden_ids=()):
    """The reference agents' rows: results_draft_<agent>.json cut to the test photos, and
    results_hidden_<agent>.json when the board has hidden photos."""
    out = []
    for name, label in REF_NAMES.items():
        p = Path(refs_dir) / f"results_draft_{name}.json"
        if not p.exists():
            continue
        rows = {r["id"]: r for r in _rows(json.loads(p.read_text())) if r["id"] in test_ids}
        hidden = {}
        if hidden_ids:
            ph = Path(refs_dir) / f"results_hidden_{name}.json"
            if ph.exists():
                hidden = {r["id"]: r for r in _rows(json.loads(ph.read_text()))
                          if r["id"] in hidden_ids}
        out.append({"name": label, "key": name, "test": rows, "hidden": hidden, "meta": {},
                    "reference": True})
    return out


def anchors(refs, test_ids, hidden_ids=()):
    """(S_A, S_O): the mean image score of A and of the oracle over the board's photos."""
    by_key = {r["key"]: r for r in refs}
    if ZERO not in by_key or HUNDRED not in by_key:
        raise SystemExit(f"the 0-100 scale needs results_draft_{ZERO}.json and "
                         f"results_draft_{HUNDRED}.json in --refs")
    out = []
    for key in (ZERO, HUNDRED):
        r = by_key[key]
        if hidden_ids and len(r["hidden"]) < len(hidden_ids):
            raise SystemExit(f"--refs: results_hidden_{key}.json is missing or lacks photos; "
                             "the final board's 0-100 scale is set on test + hidden")
        rows = [r["test"].get(i) for i in sorted(test_ids)] + \
               [r["hidden"].get(i) for i in sorted(hidden_ids)]
        out.append(float(np.mean([x["score"] if x else 0.0 for x in rows])))
    return tuple(out)


def stats(rows, n_total=None):
    """Board numbers for one set of score rows (dicts as selfdox.score --out writes them):
    raw mean (a missing photo counts 0 when n_total is given) with its bootstrap interval,
    and the cleared, leak, false-safe and pass rates."""
    rs = [from_dict(r) for r in rows if r]
    scores = [r.score for r in rs] + [0.0] * max(0, (n_total or 0) - len(rs))
    agg = aggregate(rs) if rs else None
    return {"raw": float(np.mean(scores)) if scores else None, "raw_ci": bootstrap_ci(scores),
            "n": len(rs), "cleared": agg and agg["cleared_rate"], "leak": agg and agg["leak_rate"],
            "false_safe": agg and agg["false_safe_rate"],
            "n_safe": agg and agg["n_safe_verdicts"], "pass": agg and agg["pass_rate"]}


def with_scale(entry, s_a, s_o):
    """entry plus "score" and "ci" on the 0-100 scale."""
    ci = entry.get("raw_ci")
    return {**entry, "score": normalize(entry["raw"], s_a, s_o),
            "ci": [normalize(c, s_a, s_o) for c in ci] if ci else None}


def summarize(team, test_ids, hidden_ids):
    t_rows = [team["test"].get(i) for i in sorted(test_ids)]
    h_rows = [team["hidden"].get(i) for i in sorted(hidden_ids)]
    pooled = [r["score"] if r else 0.0 for r in t_rows + h_rows]
    both = aggregate([from_dict(r) for r in t_rows + h_rows if r]) if any(t_rows + h_rows) else None
    n_test = sum(r is not None for r in t_rows)
    secs = (team["meta"].get("test") or {}).get("seconds")
    per_run = (team["meta"].get("test") or {}).get("photos_in_run") or n_test   # the final run: test + hidden in one run
    t = stats(t_rows, len(t_rows)) if any(t_rows) else None
    h = stats(h_rows, len(h_rows)) if any(h_rows) else None
    return {
        "name": team["name"], "reference": team.get("reference", False),
        "raw": float(np.mean(pooled)) if pooled else None,
        "raw_ci": bootstrap_ci(pooled),
        "per_photo": dict(zip(sorted(test_ids) + sorted(hidden_ids), pooled)),
        "false_safe_all": both["false_safe_rate"] if both else None,   # the tie-break
        "test": t, "hidden": h,
        "sec_per_photo": (secs / per_run) if secs and per_run else None,
        "peak_gpu_mib": (team["meta"].get("test") or {}).get("peak_gpu_mib"),
        "isolation": (team["meta"].get("test") or {}).get("isolation"),
        "date": (team["meta"].get("test") or {}).get("date"),
    }


def paired_delta(a, b):
    """Bootstrap 95% interval of mean(a - b) over the photos both have (raw points)."""
    ids = sorted(set(a["per_photo"]) & set(b["per_photo"]))
    d = np.array([a["per_photo"][i] - b["per_photo"][i] for i in ids])
    if not len(d):
        return None
    idx = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(d), (BOOTSTRAP_N, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def fmt(v, pct=False, d=1):
    if v is None:
        return "–"
    return f"{100 * v:.0f}%" if pct else f"{v:.{d}f}"


def _score_cell(r):
    ci = r.get("ci")
    return f"**{fmt(r['score'])}** ({fmt(ci[0]) if ci else '–'}–{fmt(ci[1]) if ci else '–'})"


def rank_key(r):
    # raw, rounded so that the same scores summed in another order still tie
    return (-round(r["raw"] or 0, 6), r.get("false_safe_all") or 0)


SCALE_NOTE = ("Score = the mean image score (“Raw”, each photo 0–100) rescaled so that "
              "**0 = doing nothing** (reference agent A) and **100 = the agent that knows the "
              "answer key** (the oracle), both on the same photos; clipped to 0–100. The "
              "ranking is the same as by Raw.")


def interim_markdown(teams, refs_doc):
    """The interim board: each team's last bot-scored entry plus the reference rows."""
    t = refs_doc["test"]
    n_test = refs_doc["n_test"]
    head = (f"Scorer {C.VERSION}. Updated {datetime.date.today().isoformat()}. "
            f"**Interim board:** test only ({n_test} photos). The live board is on the "
            "[submission site](https://huggingface.co/spaces/wpghks7/selfdox); this page keeps "
            "the reference rows. Each team row is its last submission, scored on the team's own "
            "test outputs. After "
            "the deadline we run each team's last agent once, on the test and hidden photos "
            "together (90 minutes for all 200), for the final board. A missing photo counts 0. "
            "How to get on it: "
            "[`docs/participants.md`](docs/participants.md) §6.")
    lines = ["# SelfDox leaderboard", "", head, "", SCALE_NOTE, ""]
    if not teams:
        lines += ["*No team submissions yet. The reference agents below show the range.*", ""]
    lines += ["| # | Team | Score 0–100 (95% CI) | Raw | Cleared | Leaks | False-safe (# safe) "
              "| Pass | Submissions | Scored |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    rows = [(str(i + 1), e) for i, e in enumerate(teams)] + \
           [("", with_scale(r, t["do_nothing"], t["perfect"])) for r in refs_doc["rows"]]
    for rank, r in rows:
        subs = f"{r['n_submission']}/{MAX_SUBMISSIONS}" if "n_submission" in r else "–"
        lines.append(
            f"| {rank} | {r['name']} | {_score_cell(r)} | {fmt(r['raw'])} | "
            f"{fmt(r.get('cleared'), True)} | {fmt(r.get('leak'), True)} | "
            f"{fmt(r.get('false_safe'), d=2)} ({r.get('n_safe', '–')}) | "
            f"{fmt(r.get('pass'), True)} | {subs} | {(r.get('scored_at') or '–')[:10]} |")
    lines += ["", f"0 = A’s mean on these photos ({t['do_nothing']:.1f} raw), 100 = the "
              f"oracle’s ({t['perfect']:.1f} raw). Reference rows are not ranked. Cleared = "
              "`safe`, truly safe and E ≥ 0.9; leak = a false `safe` or GPS left "
              "(`docs/design.md` §7.5). Ties go to the lower false-safe rate.", ""]
    return "\n".join(lines)


def final_markdown(rows, n_test, n_hidden, s_a, s_o):
    teams = [r for r in rows if not r["reference"]]
    head = (f"Scorer {C.VERSION}. Updated {datetime.date.today().isoformat()}. "
            f"{FINAL_MARK} one run of each team's last submission on all {n_test + n_hidden} "
            f"photos ({n_test} test + {n_hidden} hidden together, one 90-minute clock), made by "
            "the organizers after the deadline. Raw = mean image score over test + hidden; a missing "
            "photo counts 0. Rules: [`docs/participants.md`](docs/participants.md) §7.")
    lines = ["# SelfDox leaderboard", "", head, "", SCALE_NOTE, ""]
    lines += [
        "| # | Team | Score 0–100 (95% CI) | Δ next (95% CI) | Raw | Test | Hidden | Cleared | "
        "Leaks | False-safe (# safe) | Pass | Hidden pass | s/photo | Peak GPU | Network cut "
        "| Run |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        rank = "" if r["reference"] else str(teams.index(r) + 1)
        nxt = None
        if not r["reference"]:
            j = teams.index(r)
            if j + 1 < len(teams):
                nxt = paired_delta(r, teams[j + 1])
        t, h = r["test"] or {}, r["hidden"] or {}
        lines.append(
            f"| {rank} | {r['name']} | {_score_cell(r)} | "
            + (f"{scale(nxt[0], s_a, s_o):+.1f} ({scale(nxt[1][0], s_a, s_o):+.1f}–"
               f"{scale(nxt[1][1], s_a, s_o):+.1f})" if nxt else "–")
            + f" | {fmt(r['raw'])} | {fmt(t.get('raw'))} | {fmt(h.get('raw'))} | "
            f"{fmt(t.get('cleared'), True)} | {fmt(t.get('leak'), True)} | "
            f"{fmt(t.get('false_safe'), d=2)} ({t.get('n_safe', '–')}) | "
            f"{fmt(t.get('pass'), True)} | {fmt(h.get('pass'), True)} | "
            f"{fmt(r['sec_per_photo'])} | "
            f"{(str(round(r['peak_gpu_mib'] / 1024, 1)) + ' GB') if r['peak_gpu_mib'] else '–'} | "
            f"{r['isolation'] or '–'} | {r['date'] or '–'} |")
    lines += ["", f"0 = A’s mean on these photos ({s_a:.1f} raw), 100 = the oracle’s "
              f"({s_o:.1f} raw). Test and Hidden are raw means. Reference rows are not ranked. "
              "Cleared, leaks, false-safe and pass are on the test photos; hidden pass is the "
              "pass rate on hidden. Cleared = `safe`, truly safe and E ≥ 0.9; leak = a false "
              "`safe` or GPS left (`docs/design.md` §7.5). Ties go to the lower false-safe "
              "rate over the board's photos.", ""]
    return "\n".join(lines)


def load_interim(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {"entries": []}


def team_key(name):
    return " ".join(str(name).split()).casefold()


def team_slug(name):
    """The folder name of a team's final runs (runs_final/<slug>/): team_key with every run
    of other characters made "_", e.g. "Team Seven (KAIST)" -> "team_seven_kaist"."""
    return re.sub(r"[^\w.-]+", "_", team_key(name)).strip("_")


def load_settings(path=SETTINGS):
    """leaderboard/settings.json (maintainers): {"deadline_utc": ISO date and time or null (no
    offset means UTC; a date alone is refused), "team_members": {team name: [GitHub logins,
    a leading @ is fine]}}. Raises ValueError on a bad deadline, so a typo stops the export
    instead of the bot."""
    p = Path(path)
    s = json.loads(p.read_text()) if p.exists() else {}
    deadline = s.get("deadline_utc")
    if deadline:
        if "T" not in str(deadline):           # a date alone would mean 00:00 UTC
            raise ValueError(f"deadline_utc {deadline!r}: give a time too, e.g. "
                             "2026-11-30T23:59:59+09:00")
        deadline = datetime.datetime.fromisoformat(str(deadline))
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=datetime.timezone.utc)
    return {"deadline_utc": deadline or None,
            "team_members": {team_key(k): {str(x).strip().lstrip("@").casefold() for x in v}
                             for k, v in (s.get("team_members") or {}).items()}}


def counted(entries, members):
    """The entries that count: for a team whose accounts settings.json lists, only those
    accounts' entries (a name someone else claimed first is given back to its team)."""
    return [e for e in entries if team_key(e["team"]) not in members or
            (e.get("author") or "").casefold() in members[team_key(e["team"])]]


def last_entries(interim, members=None):
    """Each team's last counted entry, with its submission count, ranked."""
    by_team = {}
    for e in counted(interim["entries"], members or {}):
        by_team.setdefault(team_key(e["team"]), []).append(e)
    out = [{**es[-1], "name": es[-1]["team"], "n_submission": len(es)} for es in by_team.values()]
    return sorted(out, key=rank_key)


def refs_doc(refs_dir, test_ids):
    """leaderboard/refs.json: the 0/100 anchors on test and the reference rows (aggregates
    only, no per-photo numbers)."""
    refs = reference_rows(refs_dir, test_ids)
    s_a, s_o = anchors(refs, test_ids)
    rows = []
    for r in refs:
        s = stats([r["test"].get(i) for i in sorted(test_ids)], len(test_ids))
        rows.append({"name": r["name"], **s})
    return {"scorer_version": C.VERSION, "n_test": len(test_ids),
            "test": {"do_nothing": s_a, "perfect": s_o}, "rows": rows}


FINAL_MARK = "**Final board:**"     # final_markdown() writes it; nothing else may overwrite such a file


def is_final_board(path):
    """True if the file at path is a final board (final_markdown)."""
    path = Path(path)
    return path.is_file() and FINAL_MARK in path.read_text(encoding="utf-8", errors="replace")


def write_interim_board(out, interim_path=INTERIM, refs_path=REFS, settings_path=SETTINGS):
    out = Path(out)
    if is_final_board(out):
        raise SystemExit(f"{out} is the final board; pass --final --runs runs_final, or "
                         "another --out")
    doc = json.loads(Path(refs_path).read_text())
    teams = [with_scale(e, doc["test"]["do_nothing"], doc["test"]["perfect"])
             for e in last_entries(load_interim(interim_path),
                                   load_settings(settings_path)["team_members"])]
    md = interim_markdown(teams, doc)
    out.write_text(md)
    return md, teams


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_final", help="with --final: our final runs")
    ap.add_argument("--refs", default="outputs/results_300draft_s04")
    ap.add_argument("--split", default="data/private/split.json")
    ap.add_argument("--out", default="LEADERBOARD.md")
    ap.add_argument("--json")
    ap.add_argument("--interim", default=INTERIM, help="the bot's entries")
    ap.add_argument("--refs-json", default=REFS, help="0/100 anchors and reference rows")
    ap.add_argument("--settings", default=SETTINGS, help="the team accounts (team_members)")
    ap.add_argument("--write-refs", action="store_true",
                    help="(maintainers) rebuild --refs-json from --refs and the split, then exit")
    ap.add_argument("--final", action="store_true",
                    help="the final board: test + hidden runs made after the deadline")
    ap.add_argument("--allow-missing", action="store_true",
                    help="with --final: publish even if a team lacks a final run (counts 0)")
    args = ap.parse_args(argv)
    if args.write_refs:
        split = json.loads(Path(args.split).read_text())
        doc = refs_doc(args.refs, set(split["test"]))
        Path(args.refs_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.refs_json).write_text(json.dumps(doc, indent=1) + "\n")
        print(f"{args.refs_json}: 0 = {doc['test']['do_nothing']:.2f}, "
              f"100 = {doc['test']['perfect']:.2f} (raw, {doc['n_test']} test photos)")
        return 0
    if not args.final:
        md, teams = write_interim_board(args.out, args.interim, args.refs_json, args.settings)
        if args.json:
            Path(args.json).write_text(json.dumps(teams, indent=1))
        print(md)
        return 0
    split = json.loads(Path(args.split).read_text())
    test_ids, hidden_ids = set(split["test"]), set(split.get("hidden", []))
    if not hidden_ids:
        raise SystemExit("--final needs the hidden photos in the split")
    runs = Path(args.runs)
    teams = [load_team(d) for d in sorted(runs.iterdir())
             if d.is_dir() and ((d / "test").exists() or (d / "hidden").exists())] \
        if runs.exists() else []
    names = {team_slug(t["name"]) for t in teams}
    entered = {team_slug(e["team"]): e["team"] for e in counted(
        load_interim(args.interim)["entries"], load_settings(args.settings)["team_members"])}
    not_run = sorted(f"{entered[s]} (runs_final/{s})" for s in entered if s not in names)
    missing = not_run + [f"{t['name']} ({', '.join(k for k in ('test', 'hidden') if not t[k])})"
                         for t in teams if not t["test"] or not t["hidden"]]
    if missing:
        msg = (f"no final run for {'; '.join(missing)} (missing photos count 0); run them, or "
               "pass --allow-missing")
        if not args.allow_missing:
            raise SystemExit(msg)
        print(f"warning: {msg}", file=sys.stderr)
    refs = reference_rows(args.refs, test_ids, hidden_ids)
    s_a, s_o = anchors(refs, test_ids, hidden_ids)
    for t in teams:                          # shown as the team named itself in its issue
        t["name"] = entered.get(team_slug(t["name"]), t["name"])
    rows = sorted((summarize(t, test_ids, hidden_ids) for t in teams), key=rank_key)
    rows += [summarize(t, test_ids, hidden_ids) for t in refs]
    rows = [with_scale(r, s_a, s_o) for r in rows]
    md = final_markdown(rows, len(test_ids), len(hidden_ids), s_a, s_o)
    Path(args.out).write_text(md)
    if args.json:
        Path(args.json).write_text(json.dumps(
            [{k: v for k, v in r.items() if k != "per_photo"} for r in rows], indent=1))
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
