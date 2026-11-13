"""Score the open submission PRs on the dataset (run by .github/workflows/score-site.yml).

    python -m webapp.queue --dataset wpghks7/selfdox-submissions --check-only
    python -m webapp.queue --dataset wpghks7/selfdox-submissions --gt <decrypted key> \
        --images data/test/images --refs leaderboard/refs.json

Env: HF_TOKEN (comment, close, write state), TEAM_CODES (JSON team -> code). Two passes over
the open PRs, oldest first. The first is cheap (and stops after SCREEN_S, leaving the rest
open, so a flood of junk still leaves time to score): pin the PR's head, read its one
submission.json and find the team by its proof and the PR author; a PR that is not a
current team's site submission is commented and closed, with no record and no commit. The
second takes the rest: apply the rules, download the two zips from the pinned head, check
and score them, write state/submissions.json and state/board.json on main in one commit,
comment the result on the PR and close it. A PR already recorded is only closed. Nothing
per-photo is written, commented or printed."""
import argparse
import datetime
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

from selfdox.config import VERSION as SCORER_VERSION
from tools import bot_score as B
from tools import leaderboard as LB

from . import auth, rules, scoring

log = logging.getLogger("selfdox.queue")
MAX_AGENT_MB, MAX_OUTPUTS_MB = 30, 300
BOARD_KEYS = ("name", "score", "ci", "raw", "cleared", "leak", "false_safe", "n_safe", "pass",
              "n_submission", "scored_at")
SITE = "https://wpghks7-selfdox.static.hf.space"
META_MAX = 64 * 1024
BUDGET_S = 35 * 60
SCREEN_S = 15 * 60      # the first pass stops here, so a junk flood leaves time to score
MAX_STR, MAX_LIST = 500, 200
SUBMISSION_PATH = re.compile(r"^submissions/(?!\.{1,2}/)[\w.-]{1,80}/\d{8}T\d{6}Z/submission\.json$")
STALE = datetime.timedelta(hours=2)
FAILED_REASON = "the scoring failed on our side; this does not count, submit again"


class HubDown(RuntimeError):
    """The Hub could not be reached: stop the run, leave the PR open, retry next time."""


class SetupError(RuntimeError):
    """Our configuration is wrong (team codes); nothing is touched."""


class Junk(Exception):
    """Not a current team's site submission (path, submission.json or proof): the PR is
    commented and closed, and nothing is recorded."""


class Candidate(NamedTuple):
    """A PR that passed the first pass: its head pinned, its folder and its team known."""
    num: int
    author: str
    created: datetime.datetime
    rev: str
    folder: str
    team: str


def clean(x, cap=MAX_STR):
    """Make untrusted data safe to store: no lone surrogates, bounded strings and lists.
    `files` and `revision` were validated (path regex, sha) and are not truncated."""
    if isinstance(x, str):
        return x.encode("utf-8", "replace").decode("utf-8")[:cap]
    if isinstance(x, dict):
        return {clean(str(k)): (clean(v, 10 ** 6) if k in ("files", "revision") else clean(v))
                for k, v in list(x.items())[:MAX_LIST]}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x[:MAX_LIST]]
    return x


def transient(e):
    """A Hub problem worth retrying: connection error, timeout, HTTP 429 or 5xx. Looks at the
    exception and its causes; anything else is that PR's own fault."""
    import requests
    seen = 0
    while e is not None and seen < 8:
        if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
            return True
        code = getattr(getattr(e, "response", None), "status_code", None)
        if isinstance(code, int) and (code == 429 or code >= 500):
            return True
        e, seen = e.__cause__ or e.__context__, seen + 1
    return False


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


def iso(t):
    return t.isoformat(timespec="seconds")


class HubPRs:
    """The dataset's pull requests through huggingface_hub."""

    def __init__(self, repo_id, token=None):
        from huggingface_hub import HfApi
        self.api, self.repo, self.token = HfApi(token=token), repo_id, token

    def open_prs(self):
        ds = self.api.get_repo_discussions(self.repo, repo_type="dataset",
                                           discussion_type="pull_request",
                                           discussion_status="open")
        return sorted(((d.num, d.author, d.created_at) for d in ds), key=lambda t: t[0])

    @staticmethod
    def _call(fn, *args, **kw):
        try:
            return fn(*args, **kw)
        except Exception as e:                 # noqa: BLE001
            if transient(e):
                raise HubDown(type(e).__name__) from None
            raise

    def head(self, num):
        """The PR's head commit sha, resolved once so every file comes from the same commit."""
        return self._call(self.api.repo_info, self.repo, revision=f"refs/pr/{num}",
                          repo_type="dataset").sha

    def pr_files(self, num, rev=None):
        return self._call(self.api.list_repo_files, self.repo, repo_type="dataset",
                          revision=rev or f"refs/pr/{num}")

    def last_change(self, num):
        """The server time of the PR's last change: its creation, any commit pushed to it and
        any status change its author made (a draft published, a reopen). The organizer's
        closes and reopens are left out, so a PR reopened under the README's deadline policy
        keeps its times."""
        d = self._call(self.api.get_discussion_details, self.repo, num, repo_type="dataset")
        who = str(d.author).lower()
        return max([d.created_at] + [e.created_at for e in d.events or []
                                     if e.type == "commit" or (e.type == "status-change"
                                                               and str(e.author).lower() == who)])

    def size(self, num, path, rev=None):
        info = self._call(self.api.get_paths_info, self.repo, [path],
                          revision=rev or f"refs/pr/{num}", repo_type="dataset")
        return getattr(info[0], "size", None) if info else None

    def download(self, num, path, dest, rev=None):
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError
        with tempfile.TemporaryDirectory() as cache:
            try:
                p = hf_hub_download(self.repo, path, repo_type="dataset",
                                    revision=rev or f"refs/pr/{num}", token=self.token,
                                    cache_dir=cache)
            except LocalEntryNotFoundError as e:       # a subclass of EntryNotFoundError:
                if transient(e):                       # any HEAD failure lands here
                    raise HubDown(type(e).__name__) from None
                raise
            except EntryNotFoundError:
                raise FileNotFoundError(path) from None
            except Exception as e:                     # noqa: BLE001
                if transient(e):
                    raise HubDown(type(e).__name__) from None
                raise
            shutil.copyfile(p, dest)
        return Path(dest)

    def comment(self, num, text):
        self.api.comment_discussion(self.repo, num, text, repo_type="dataset")

    def close(self, num):
        self.api.change_discussion_status(self.repo, num, "closed", repo_type="dataset")


def scored_text(team, rec, max_n):
    r = rec["result"]
    lines = [f"**SelfDox · {team} · submission {rec['n']}/{max_n}**", "",
             f"- Score (0–100): **{r['score']:.1f}** (95% CI {r['ci'][0]:.1f}–{r['ci'][1]:.1f})",
             f"- Raw score (mean image score): {r['raw']:.1f}",
             f"- Agent sha256: `{rec['agent_sha256']}`",
             f"- Scorer {SCORER_VERSION}"]
    lines += [f"- {w}" for w in rec.get("warnings") or []]
    lines += ["", f"Leaderboard and your team's history: {SITE}"]
    return "\n".join(lines) + "\n"


def rejected_text(reason):
    return (f"**SelfDox · not scored**\n\n{reason}.\n\nThis does not count as a submission. "
            f"Fix it and submit again on {SITE}\n")


def junk_text(reason):
    return (f"**SelfDox · not scored**: {reason}.\n\nNothing was recorded and this does not "
            f"count as a submission. Submit from the site: {SITE}\n")


def build_board(records, refs, settings, now, teams):
    """The interim board: each current team's (one in `teams`, i.e. TEAM_CODES) last scored
    record, and the reference rows. A removed team's records stay in submissions.json."""
    t = refs["test"]
    teams = set(teams)
    entries = [{"team": r["team"], **r["result"], "false_safe_all": r["result"].get("false_safe"),
                "scored_at": r.get("scored_at")} for r in records
               if r["status"] == "scored" and r.get("team") in teams]
    pick = lambda e: {k: e.get(k) for k in BOARD_KEYS}  # noqa: E731
    return {"mode": "interim", "updated": iso(now), "n_test": refs["n_test"], "anchors": t,
            "max": rules.max_of(settings),
            "teams": [pick(LB.with_scale(e, t["do_nothing"], t["perfect"]))
                      for e in LB.last_entries({"entries": entries})],
            "references": [pick(LB.with_scale(r, t["do_nothing"], t["perfect"]))
                           for r in refs["rows"]]}


def handle(hub, c, records, settings, final, gt, ids, refs, work, now):
    rec = {"pr": c.num, "hf_user": c.author, "created_at": iso(c.created), "team": c.team,
           "revision": c.rev}
    try:
        rec = _handle(hub, c, rec, records, settings, final, gt, ids, refs, work, now)
        rec = clean(rec)
        json.dumps(rec, ensure_ascii=False).encode()    # must be storable, or it fails below
    except Exception as e:                              # noqa: BLE001
        if isinstance(e, HubDown):
            raise                                       # process() decides: retry or give up
        log.error("PR %s failed: %s", c.num, type(e).__name__)
        rec = failed_record(c.num, c.author, c.created, rec.get("team"), rec.get("revision"))
    return rec


def failed_record(num, author, created, team=None, revision=None):
    rec = clean({"pr": num, "hf_user": author, "created_at": iso(created), "team": team,
                 "revision": revision})
    rec.update(status="failed", reason=FAILED_REASON)
    return rec


def _fetch(hub, num, rev, path, dest, cap_bytes, label):
    size = hub.size(num, path, rev)
    if size is None:
        raise B.Refused(f"{label} is missing")
    if size > cap_bytes:
        raise B.Refused(f"{label} is over {cap_bytes // 2 ** 20} MB" if cap_bytes >= 2 ** 20
                        else f"{label} is too large")
    try:
        p = hub.download(num, path, dest, rev)
    except FileNotFoundError:
        raise B.Refused(f"{label} is missing") from None
    if p.stat().st_size > cap_bytes:
        raise B.Refused(f"{label} is too large")
    return p


def screen(hub, num, author, created, codes, work):
    """The first pass, a few small Hub calls: pin the head, find the one submission.json at a
    site path, download it (size-checked) and match its proof. Returns a Candidate; raises
    Junk when the PR is not a current team's site submission."""
    rev = hub.head(num)
    meta_paths = [p for p in hub.pr_files(num, rev) if p.endswith("submission.json")]
    if len(meta_paths) != 1 or not SUBMISSION_PATH.match(meta_paths[0]):
        raise Junk("one submission.json made by the site is needed")
    try:
        meta_file = _fetch(hub, num, rev, meta_paths[0], work / "submission.json", META_MAX,
                           "submission.json")
        meta = json.loads(meta_file.read_text())
    except B.Refused as e:
        raise Junk(str(e)) from None
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError, ValueError):
        raise Junk("submission.json is not valid JSON") from None
    team = auth.team_for_proof(codes, author, meta.get("proof") if isinstance(meta, dict) else None)
    if team is None:
        raise Junk("the team code does not match this Hugging Face account")
    return Candidate(num, author, created, rev, meta_paths[0].rsplit("/", 1)[0], team)


def _handle(hub, c, rec, records, settings, final, gt, ids, refs, work, now):
    try:
        if final:
            raise rules.Closed("submissions are closed: the final board is out")
        changed = hub.last_change(c.num) if (settings or {}).get("deadline_utc") else None
        rules.check_can_submit(records, c.team, settings, c.created, changed)
        paths = {}
        for name, cap in (("agent.zip", MAX_AGENT_MB), ("outputs.zip", MAX_OUTPUTS_MB)):
            paths[name] = _fetch(hub, c.num, c.rev, f"{c.folder}/{name}", work / name,
                                 cap * 2 ** 20, name)
        m, warnings = scoring.quick_check(paths["agent.zip"], paths["outputs.zip"], ids, work / "check")
        result = scoring.score(paths["outputs.zip"], gt, refs, work / "score")
        rec.update(status="scored", n=len(rules.scored(records, c.team)) + 1, scored_at=iso(now()),
                   agent_sha256=m["agent_sha256"], models=m["models"],
                   weights=m.get("weights", []), warnings=warnings, result=result,
                   files={"agent": f"{c.folder}/agent.zip", "outputs": f"{c.folder}/outputs.zip"},
                   agent_zip_sha256=B.file_sha256(paths["agent.zip"]))
    except (B.Refused, rules.Closed) as e:
        rec.update(status="rejected", reason=str(e))
    return rec


def check_setup(store, codes):
    """Every team on file needs its code in TEAM_CODES, or every proof would look wrong."""
    if not codes:
        raise SetupError("TEAM_CODES is empty")
    hashes = {auth.code_hash(c) for c in codes.values()}
    missing = [t.get("team") for t in store.read_json("state/teams.json", []) or []
               if t.get("code_sha256") not in hashes]
    if missing:
        raise SetupError(f"{len(missing)} team(s) on file have no matching code in TEAM_CODES")


def _safely(fn, *args):
    """Comment/close failures must not stop the queue; a recorded PR is closed next run."""
    try:
        fn(*args)
    except Exception as e:                     # noqa: BLE001
        log.error("%s on PR %s failed: %s", fn.__name__, args[0], type(e).__name__)


def _give_up(store, hub_errors, num, t):
    """The Hub failed for this PR. True once it has been failing for more than STALE (give up
    on the PR); else note when it started (first time only) and return False."""
    first = hub_errors.get(str(num))
    try:
        started = datetime.datetime.fromisoformat(first)
    except (TypeError, ValueError):
        started = None
    if started is not None:
        return t - started > STALE
    hub_errors[str(num)] = iso(t)
    store.write_json("state/hub_errors.json", hub_errors, f"PR #{num}: Hub error noted")
    return False


def process(hub, store, gt_path, refs, codes, workdir, now=utcnow, clock=time.monotonic,
            budget=BUDGET_S, screen_budget=SCREEN_S):
    """Handle every open PR; returns how many were recorded. The first pass screens them
    (screen()) and closes the junk without a record or a commit, for at most `screen_budget`
    seconds; the second scores the candidates found, one commit each, until `budget` (from
    the start of the run) is used up. PRs not reached stay open for the next run. A HubDown
    in either pass ends the run, unless that PR has been failing for more than STALE: then
    it is recorded as failed and closed."""
    check_setup(store, codes)
    started = clock()
    records = store.read_json("state/submissions.json", []) or []
    settings = store.read_json("state/settings.json", {}) or {}
    board = store.read_json("state/board.json", {}) or {}
    final = board.get("mode") == "final"
    ids = scoring.test_ids(gt_path)
    known = {r["pr"] for r in records}
    hub_errors = store.read_json("state/hub_errors.json", {}) or {}
    prs = hub.open_prs()
    stale = set(hub_errors) - {str(num) for num, _, _ in prs}
    for k in stale:                            # PRs closed since: dropped at the next commit
        del hub_errors[k]
    errors_dirty = bool(stale)
    n_new = n_junk = 0

    def record(rec):
        """One commit for the PR's record (and the board), then comment and close."""
        nonlocal final, errors_dirty, n_new
        num = rec["pr"]
        if not final and (store.read_json("state/board.json", {}) or {}).get("mode") == "final":
            final = True                       # published while we worked: never overwrite it
            if rec["status"] == "scored":
                rec = {k: rec.get(k) for k in ("pr", "hf_user", "created_at", "team", "revision")}
                rec.update(status="rejected", reason="submissions are closed: the final board is out")
        records.append(rec)
        known.add(num)
        files = {"state/submissions.json": records}
        if hub_errors.pop(str(num), None) is not None or errors_dirty:
            files["state/hub_errors.json"] = hub_errors
            errors_dirty = False
        if rec["status"] == "scored" and not final:
            files["state/board.json"] = build_board(records, refs, settings, now(), codes)
        store.write_jsons(files, f"PR #{num}: {rec['status']}")
        _safely(hub.comment, num, scored_text(rec["team"], rec, rules.max_of(settings))
                if rec["status"] == "scored" else rejected_text(rec["reason"]))
        _safely(hub.close, num)
        n_new += 1

    candidates = []
    for num, author, created in prs:           # first pass: cheap, no commit for junk
        if num in known:
            _safely(hub.close, num)
            continue
        if clock() - started > min(screen_budget, budget):
            log.info("first-pass time used up; leaving the unscreened PRs open")
            break
        work = Path(workdir) / f"pr{num}"
        work.mkdir(parents=True, exist_ok=True)
        try:
            candidates.append(screen(hub, num, author, created, codes, work))
        except Junk as e:
            n_junk += 1
            _safely(hub.comment, num, junk_text(e))
            _safely(hub.close, num)
        except HubDown:
            if not _give_up(store, hub_errors, num, now()):
                raise                          # retry next run, the PR stays open
            record(failed_record(num, author, created))
        except Exception as e:                 # noqa: BLE001
            log.error("PR %s failed in the first pass: %s", num, type(e).__name__)
            _safely(hub.comment, num, rejected_text(FAILED_REASON))
            _safely(hub.close, num)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    for c in candidates:                       # second pass: rules, zips, score, one commit
        if clock() - started > budget:
            log.info("time budget used up; leaving the remaining PRs open")
            break
        work = Path(workdir) / f"pr{c.num}"
        work.mkdir(parents=True, exist_ok=True)
        try:
            try:
                rec = handle(hub, c, records, settings, final, gt_path, ids, refs, work, now)
            except HubDown:
                if not _give_up(store, hub_errors, c.num, now()):
                    raise                      # retry next run, the PR stays open
                rec = failed_record(c.num, c.author, c.created, c.team, c.rev)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        record(rec)
    if n_junk:
        log.info("closed %d PR(s) that are not a current team's site submission", n_junk)
    stored = store.read_json("state/board.json") or {}
    shown = {t.get("name") for t in stored.get("teams") or []}
    if not stored or (stored.get("mode") != "final" and shown - set(codes)):
        store.write_jsons({"state/board.json": build_board(records, refs, settings, now(), codes)},
                          "board without removed teams" if stored else "initial board")
    return n_new


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--check-only", action="store_true", help="print open=<n> and exit")
    ap.add_argument("--gt", help="the decrypted test key")
    ap.add_argument("--images", default="data/test/images")
    ap.add_argument("--refs", default="leaderboard/refs.json")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    hub = HubPRs(args.dataset, token=os.environ.get("HF_TOKEN"))
    if args.check_only:
        print(f"open={len(hub.open_prs())}")
        return 0
    from .store import HubStore
    codes = json.loads(os.environ.get("TEAM_CODES") or "{}")
    store = HubStore(args.dataset, token=os.environ.get("HF_TOKEN"))
    try:
        check_setup(store, codes)
    except SetupError as e:
        print(f"setup error: {e}", file=sys.stderr)
        return 2
    if args.gt is None:
        ap.error("--gt is required unless --check-only")
    with tempfile.TemporaryDirectory() as tmp:
        keydir = Path(tmp) / "key"
        keydir.mkdir()
        shutil.copyfile(args.gt, keydir / "gt.json")
        (keydir / "images").symlink_to(Path(args.images).resolve())
        n = process(hub, store,
                    keydir / "gt.json", json.loads(Path(args.refs).read_text()), codes,
                    Path(tmp) / "work")
    print(f"recorded {n} submission(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
