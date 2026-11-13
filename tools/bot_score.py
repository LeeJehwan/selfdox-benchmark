"""Shared check and score functions used by webapp/queue.py (the submission site's scorer).

The GitHub issue bot built on this module was retired in v1.2; what follows describes its
original design and is kept for the functions it left behind.

The submission bot: score a team's uploaded test outputs for the interim board.

Runs in GitHub Actions on the public repository (.github/workflows/score-submission.yml)
when a "Submission" issue is opened or edited, or by hand as a dry run. Two steps:

    uv run python -m tools.bot_score score --event "$GITHUB_EVENT_PATH" \
        --gt "$RUNNER_TEMP/gt.json" --images data/test/images --out "$RUNNER_TEMP/result.json"
    uv run python -m tools.bot_score publish --result "$RUNNER_TEMP/result.json" \
        --comment "$RUNNER_TEMP/comment.md" --branch main [--dry-run --board-out preview.md]

score (writes nothing in the repository):
1. Reads the issue form (team, models, the agent zip attached by selfdox.submit, a link to
   test_outputs_<team>.zip). One issue is one submission: an issue that was already scored
   is not scored again; one that was refused (bad link, bad zip) is retried when edited.
2. Checks the rules (check_rules): the deadline in leaderboard/settings.json, the team's
   GitHub account (the first account that submits for a team name owns it, unless
   settings.json lists the team's accounts; one account submits for one team only), and at
   most MAX_SUBMISSIONS scored submissions per team.
3. Downloads both zips over https from GitHub only (issue attachments, release assets, raw
   files; storage hosts only as redirect targets), with size and time caps. From the agent
   zip it reads metadata.json and recomputes the agent folder's sha256 from the members'
   bytes (the digest tools.run_container.tree_digest gives the unpacked agent/), refusing a
   mismatch; nothing is extracted or run. The outputs are unpacked safely: only
   predictions.json and images/<file>.jpg|jpeg|png, no links, no absolute or ".." paths,
   capped sizes.
4. Validates and scores the outputs like selfdox.score (a prediction that fails validation
   scores 0 for its photo, a missing photo counts 0) against the decrypted test key.

publish (the only step that writes): fetches the latest default branch, checks the rules
again on it (another run may have finished meanwhile), appends the entry to
leaderboard/interim.json, rebuilds LEADERBOARD.md (tools.leaderboard, interim mode),
commits and pushes, retrying from a fresh fetch if the push is refused. Only then does it
write the comment: the 0-100 score with its 95% interval, the raw score, the submission
count, the agent sha256 and the scorer version. No per-photo numbers, in the comment or the
log (the log of a public repository is public), and text from the submission is quoted in
code spans so it can't mention people or post links.

--dry-run (manual runs are always dry) writes nothing but a board preview (--board-out).
On a manual run the outputs link may be "local:A" or "local:B": our reference agent A or B
is run on the test photos first (our code, not a team's).
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
import zlib
from dataclasses import asdict
from pathlib import Path, PurePosixPath

from selfdox import config as C
from selfdox.schema import submission_format_errors
from selfdox.score import invalid_predictions, score_submission
from tools import leaderboard as LB
from tools.run_container import SKIP_DIRS            # the same files tree_digest hashes

SETTINGS = LB.SETTINGS
load_settings = LB.load_settings
# issue form labels (.github/ISSUE_TEMPLATE/4-submission.yml) -> field ids
LABELS = {"Team name": "team", "Contact": "contact", "Models": "models",
          "Other tools": "tools", "Temperature and seed": "decoding",
          "Agent zip": "agent_zip", "Test outputs link": "outputs"}
TEAM_RE = re.compile(r"^[\w][\w .()-]{0,39}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_AGENT_MB = 30               # the zip (GitHub attaches up to 25 MB)
MAX_AGENT_UNZIPPED_MB = 300
MAX_OUTPUTS_MB = 300
MAX_UNZIPPED_MB = 1500
MAX_MEMBER_MB = 40
MAX_MEMBERS = 1000
TIMEOUT_S = 600
PUSH_ATTEMPTS = 5
IMAGE_NAME = re.compile(r"^images/[\w.-]{1,80}\.(jpg|jpeg|png)$", re.I)
ZIP_ERRORS = (zipfile.BadZipFile, zipfile.LargeZipFile, NotImplementedError, RuntimeError,
              zlib.error, EOFError, ValueError, OSError)


class Refused(Exception):
    """The submission can't be scored; the message goes into the comment."""


def quote(text, n=80):
    """Untrusted text for the comment: one line, no backticks, inside a code span (so it
    cannot mention anyone or become a link)."""
    s = re.sub(r"[`\r\n\t]+", " ", str(text)).strip()[:n]
    return f"`{s}`" if s else "`(empty)`"


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


# --- the issue form, settings and rules ---

def parse_form(body):
    """{field id: text} from an issue form body ("### Label" sections)."""
    out = {}
    for part in re.split(r"^### ", body or "", flags=re.M)[1:]:
        label, _, text = part.partition("\n")
        key = next((v for k, v in LABELS.items() if label.strip().startswith(k)), None)
        text = text.strip()
        if key and text and text != "_No response_":
            out[key] = text
    return out


def first_url(text):
    m = re.search(r"https://[^\s)\]>\"'`]+", text or "")
    return m.group(0) if m else None


def check_team(name):
    name = " ".join((name or "").split())
    if not TEAM_RE.match(name):
        raise Refused("the team name must be 1-40 letters, digits, spaces or `. ( ) - _`")
    return name


def check_rules(interim, settings, team, author, issue, now):
    """The submission number this one would get, or None if the issue was already scored.
    Raises Refused past the deadline, for another team's account, or over the cap."""
    entries = interim["entries"]
    if issue is not None and any(e.get("issue") == issue for e in entries):
        return None
    deadline = settings.get("deadline_utc")
    if deadline and now > deadline:
        raise Refused(f"submissions closed at {deadline.isoformat()}")
    key = LB.team_key(team)
    if author:
        a = author.casefold()
        listed = settings["team_members"].get(key)
        owners = listed or {(e.get("author") or "").casefold() for e in entries
                            if LB.team_key(e["team"]) == key}
        if owners and a not in owners:
            raise Refused(f"team {quote(team)} submits from another GitHub account. If you are "
                          "on that team, ask the organizers in a Question issue to add your "
                          "account")
        others = {LB.team_key(e["team"]) for e in entries
                  if (e.get("author") or "").casefold() == a} | \
                 {k for k, v in settings["team_members"].items() if a in v}
        others.discard(key)
        if others:
            raise Refused("your GitHub account already submits for another team; one account "
                          "submits for one team")
    done = sum(LB.team_key(e["team"]) == key
               for e in LB.counted(entries, settings["team_members"]))
    if done >= LB.MAX_SUBMISSIONS:
        raise Refused(f"team {quote(team)} has used all {LB.MAX_SUBMISSIONS} submissions")
    return done + 1


def board_is_final(board):
    p = Path(board)
    return p.exists() and "**Final board:**" in p.read_text()


# --- downloads ---

def allowed_url(url, redirect=False):
    """A link a team may paste: https on github.com (issue attachments, release assets) or
    raw.githubusercontent.com. Redirects may also go to GitHub's storage hosts."""
    u = urllib.parse.urlparse(url)
    if u.scheme != "https" or u.username or u.password or u.port not in (None, 443):
        return False
    host = (u.hostname or "").lower()
    if host == "github.com":
        parts = u.path.split("/")
        return (len(parts) > 2 and parts[1] == "user-attachments") or \
            (len(parts) > 5 and parts[3] == "releases" and parts[4] == "download")
    if host == "raw.githubusercontent.com":
        return True
    return redirect and (
        host in ("objects.githubusercontent.com", "release-assets.githubusercontent.com") or
        (host.endswith(".s3.amazonaws.com") and host.startswith("github-production-")))


class _Redirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not allowed_url(newurl, redirect=True):
            raise Refused("the link redirects outside GitHub")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url, dest, max_mb, timeout=TIMEOUT_S):
    if not url or not allowed_url(url):
        raise Refused("links must be https GitHub issue attachments, release assets of a "
                      "public repository, or raw files")
    opener = urllib.request.build_opener(_Redirects)
    req = urllib.request.Request(url, headers={"User-Agent": "selfdox-bot"})
    start, n = utcnow(), 0
    try:
        with opener.open(req, timeout=60) as r, open(dest, "wb") as f:
            while chunk := r.read(1 << 20):
                n += len(chunk)
                if n > max_mb * 2 ** 20:
                    raise Refused(f"the file is over {max_mb} MB")
                if (utcnow() - start).total_seconds() > timeout:
                    raise Refused(f"the download took over {timeout // 60} minutes")
                f.write(chunk)
    except Refused:
        raise
    except Exception as e:  # noqa: BLE001 - network errors of any kind
        raise Refused(f"could not download the file ({type(e).__name__}); is the repository "
                      "public?") from None
    return dest


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --- zips ---

def _is_link(info):
    return (info.external_attr >> 16) & 0o170000 == 0o120000


def _unsafe(name):
    p = PurePosixPath(name)
    return name.startswith(("/", "\\")) or ".." in p.parts or "\\" in name or ":" in name


def safe_unzip_outputs(zip_path, dest):
    """Unpack predictions.json and images/* (optionally under one top folder) into dest."""
    try:
        return _unzip_outputs(zip_path, dest)
    except ZIP_ERRORS as e:
        raise Refused(f"could not read the test outputs zip ({type(e).__name__})") from None


def _unzip_outputs(zip_path, dest):
    with zipfile.ZipFile(zip_path) as z:
        # Finder adds __MACOSX/ and .DS_Store: skipped
        infos = [i for i in z.infolist() if not i.is_dir() and "__MACOSX" not in i.filename
                 and PurePosixPath(i.filename).name != ".DS_Store"]
        if len(infos) > MAX_MEMBERS:
            raise Refused(f"the zip has over {MAX_MEMBERS} files")
        names = [i.filename for i in infos]
        for n in names:
            if _unsafe(n):
                raise Refused(f"unsafe path in the zip: {quote(n)}")
        tops = {n.split("/", 1)[0] for n in names}
        prefix = ""
        if "predictions.json" not in names and len(tops) == 1 and \
                next(iter(tops)) + "/predictions.json" in names:
            prefix = next(iter(tops)) + "/"
        total, wanted = 0, []
        for i in infos:
            rel = i.filename[len(prefix):] if i.filename.startswith(prefix) else None
            if rel is None or (rel != "predictions.json" and not IMAGE_NAME.match(rel)):
                raise Refused(f"unexpected file in the zip: {quote(i.filename)} (only "
                              "predictions.json and images/; use selfdox.submit --test-sub)")
            if _is_link(i):
                raise Refused(f"links are not allowed in the zip: {quote(i.filename)}")
            if i.file_size > MAX_MEMBER_MB * 2 ** 20:
                raise Refused(f"{quote(rel)} is over {MAX_MEMBER_MB} MB")
            total += i.file_size
            wanted.append((i, rel))
        if total > MAX_UNZIPPED_MB * 2 ** 20:
            raise Refused(f"the zip unpacks to over {MAX_UNZIPPED_MB} MB")
        if not any(rel == "predictions.json" for _, rel in wanted):
            raise Refused("no predictions.json in the zip")
        dest = Path(dest)
        (dest / "images").mkdir(parents=True, exist_ok=True)
        for i, rel in wanted:
            out, n = dest / rel, 0
            with z.open(i) as src, open(out, "wb") as f:      # sizes in headers can lie
                while chunk := src.read(1 << 20):
                    n += len(chunk)
                    if n > i.file_size or n > MAX_MEMBER_MB * 2 ** 20:
                        raise Refused(f"{quote(rel)} unpacks larger than its header says")
                    f.write(chunk)
    return dest


def agent_digest(z):
    """tools.run_container.tree_digest of the agent/ folder this zip unpacks to, computed
    from the members' bytes (nothing is written or run)."""
    members = []
    for i in z.infolist():
        if i.is_dir() or not i.filename.startswith("agent/"):
            continue
        if _unsafe(i.filename) or _is_link(i):
            raise Refused(f"unsafe file in the agent zip: {quote(i.filename)}")
        rel = PurePosixPath(i.filename[len("agent/"):])
        if rel.parts and not SKIP_DIRS & set(rel.parts):
            members.append((rel, i))
    h, total = hashlib.sha256(), 0
    for rel, i in sorted(members, key=lambda t: t[0]):       # the order of sorted(rglob())
        h.update(str(rel).encode() + b"\0")
        with z.open(i) as f:
            while block := f.read(1 << 20):
                total += len(block)
                if total > MAX_AGENT_UNZIPPED_MB * 2 ** 20:
                    raise Refused(f"the agent zip unpacks to over {MAX_AGENT_UNZIPPED_MB} MB")
                h.update(block)
        h.update(b"\0")
    return h.hexdigest()


def agent_metadata(zip_path):
    """agent_sha256 and models from the agent zip's metadata.json, with the sha256 checked
    against the zip's agent/ files."""
    try:
        with zipfile.ZipFile(zip_path) as z:
            info = z.getinfo("metadata.json")
            if info.file_size > 64 * 1024:
                raise Refused("metadata.json in the agent zip is too large")
            meta = json.loads(z.read(info))
            if not isinstance(meta, dict):
                raise ValueError
            sha = str(meta.get("agent_sha256", ""))
            if not HEX64.match(sha):
                raise Refused("metadata.json has no agent_sha256; pack the agent with "
                              "selfdox.submit")
            if agent_digest(z) != sha:
                raise Refused("the agent files do not match the agent_sha256 in "
                              "metadata.json; pack the agent again with selfdox.submit")
    except Refused:
        raise
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError, *ZIP_ERRORS):
        raise Refused("the agent zip must be the one selfdox.submit wrote (with "
                      "metadata.json)") from None
    models = meta.get("models", meta.get("model"))
    models = [str(m)[:200] for m in (models if isinstance(models, list) else [models]) if m]
    weights = meta.get("weights") if isinstance(meta.get("weights"), list) else []
    return {"agent_sha256": sha, "models": models[:10],
            "weights": [str(w)[:300] for w in weights][:10]}


# --- scoring ---

def score_outputs(gt_path, sub_dir, refs):
    """The board entry numbers for one outputs folder (no per-photo rows kept)."""
    _, errors = submission_format_errors(sub_dir)
    if errors:
        raise Refused("predictions.json does not match the format: "
                      + "; ".join(quote(e, 160) for e in errors[:3])
                      + ". Run `selfdox.validate` on your outputs")
    from selfdox.schema import load_gt
    gt = load_gt(gt_path)
    zero, _ = invalid_predictions(sub_dir, gt)
    _, _, _, results = score_submission(gt_path, sub_dir, zero=zero)
    s = LB.stats([asdict(r) for r in results], len(gt["images"]))
    s = LB.with_scale(s, refs["test"]["do_nothing"], refs["test"]["perfect"])
    s["false_safe_all"] = s["false_safe"]
    n_scored = sum(1 for r in results if r.parts)
    return s, {"n_zero": len(zero), "n_unscored": len(results) - n_scored,
               "n_photos": len(results)}


def comment_text(entry, info, n):
    ci = entry["ci"]
    lines = [f"**SelfDox bot · {entry['team']} · submission {n}/{LB.MAX_SUBMISSIONS}**", "",
             f"- Score (0–100): **{entry['score']:.1f}** (95% CI {ci[0]:.1f}–{ci[1]:.1f})",
             f"- Raw score (mean image score): {entry['raw']:.1f}",
             f"- Agent sha256: `{entry['agent_sha256']}`",
             f"- Scorer {C.VERSION} · {info['n_photos']} test photos"]
    if info["n_zero"] or info["n_unscored"]:
        lines.append(f"- {info['n_zero'] + info['n_unscored']} photo(s) scored 0 (missing, "
                     "invalid or unreadable output); run `selfdox.validate` on your outputs")
    lines += ["", "0 = doing nothing, 100 = the agent that knows the answer key. This is the "
              "interim board (your uploaded test outputs). After the deadline we run the agent "
              "of your last submission on test and hidden for the final board."]
    if n >= LB.MAX_SUBMISSIONS:
        lines.append(f"\nThat was your last submission ({LB.MAX_SUBMISSIONS} in total).")
    return "\n".join(lines) + "\n"


def refusal_text(team, why):
    return (f"**SelfDox bot · {team or 'submission'} · not scored**\n\n{why}.\n\nThis does not "
            "count as a submission. Fix it and edit the issue to try again.\n")


FAILED_TEXT = ("**SelfDox bot · not scored**\n\nThe bot failed on our side. This does not "
               "count as a submission; the organizers will look at it.\n")


def local_outputs(which, images, dest):
    """Dry runs only: our own reference agent A or B on the test photos."""
    mod = {"A": "agents.a_noop", "B": "agents.b_exif_strip"}[which]
    subprocess.run([sys.executable, "-m", mod, "--inputs", str(images), "--out", str(dest)],
                   check=True, stdout=subprocess.DEVNULL)
    return dest


def score_issue(event, gt_path, images, refs_path=LB.REFS, interim_path=LB.INTERIM,
                settings_path=SETTINGS, board="LEADERBOARD.md", now=None):
    """The score step: {"status": "skip" | "refused" | "scored", ...}. Writes nothing."""
    now = now or utcnow()
    refs = json.loads(Path(refs_path).read_text())
    issue = event.get("issue") or {}
    manual = not issue
    if manual:
        inp = event.get("inputs") or {}
        form = {"team": inp.get("team", "dryrun"), "outputs": inp.get("outputs_url", "")}
    else:
        form = parse_form(issue.get("body"))
    author = None if manual else (issue.get("user") or {}).get("login")
    number = issue.get("number")
    base = {"started_at": now.isoformat(timespec="seconds"), "manual": manual}
    try:
        team = check_team(form.get("team"))
    except Refused as e:
        return {**base, "status": "refused", "text": refusal_text(None, str(e))}
    try:
        if not manual and board_is_final(board):
            raise Refused("submissions are closed: the final board is out")
        if not manual and check_rules(LB.load_interim(interim_path), load_settings(settings_path),
                                      team, author, number, now) is None:
            return {**base, "status": "skip"}          # already scored; edits don't re-score
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            outputs = form.get("outputs", "")
            meta = {"agent_sha256": "0" * 64, "models": []}
            links = {}
            if manual and outputs in ("local:A", "local:B"):
                sub = local_outputs(outputs[-1], images, tmp / "sub")
            else:
                if not manual:
                    agent_url = first_url(form.get("agent_zip"))
                    if not agent_url:
                        raise Refused("attach the agent zip from selfdox.submit")
                    zp = download(agent_url, tmp / "agent.zip", MAX_AGENT_MB)
                    meta = agent_metadata(zp)
                    links.update(agent_url=agent_url, agent_zip_sha256=file_sha256(zp))
                url = first_url(outputs)
                if not url:
                    raise Refused("paste the link to test_outputs_<team>.zip")
                zp = download(url, tmp / "outputs.zip", MAX_OUTPUTS_MB)
                links.update(outputs_url=url, outputs_zip_sha256=file_sha256(zp))
                sub = safe_unzip_outputs(zp, tmp / "sub")
            stats, info = score_outputs(gt_path, sub, refs)
    except Refused as e:
        return {**base, "status": "refused", "text": refusal_text(team, str(e))}
    entry = {"team": team, "issue": number, "author": author,
             "scored_at": now.isoformat(timespec="seconds"), "scorer_version": C.VERSION,
             **meta, **links, "declared_models": form.get("models", "")[:500], **stats}
    return {**base, "status": "scored", "entry": entry, "info": info}


# --- publishing ---

def _git(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    return r.returncode == 0


def push_url():
    token, repo = os.environ.get("GH_TOKEN", ""), os.environ.get("GITHUB_REPOSITORY", "")
    return f"https://x-access-token:{token}@github.com/{repo}.git"


def publish(result, refs_path=LB.REFS, interim_path=LB.INTERIM, settings_path=SETTINGS,
            board_out="LEADERBOARD.md", branch="main", dry_run=False, git=_git, remote=None,
            attempts=PUSH_ATTEMPTS):
    """The publish step: the comment text (empty = say nothing). With git (not a dry run):
    fetch the branch, check the rules again on it, append, commit, push; retry on a
    refused push. The score comment is returned only after a successful push."""
    if result["status"] == "skip":
        return ""
    if result["status"] == "refused":
        return result["text"]
    entry, info = result["entry"], result["info"]
    team, now = entry["team"], datetime.datetime.fromisoformat(result["started_at"])
    dry_run = dry_run or result.get("manual")
    for _ in range(attempts if git and not dry_run else 1):
        if git and not dry_run:
            if not (git("fetch", "--quiet", "origin", branch) and
                    git("reset", "--hard", "--quiet", "FETCH_HEAD")):
                continue
        interim = LB.load_interim(interim_path)
        try:
            if not dry_run and board_is_final(board_out):
                raise Refused("submissions are closed: the final board is out")
            n = 1 if dry_run else check_rules(interim, load_settings(settings_path), team,
                                              entry.get("author"), entry.get("issue"), now)
        except Refused as e:
            return refusal_text(team, str(e))
        if n is None:
            return ""                                  # another run scored this issue
        new = {"entries": interim["entries"] + [entry]}
        refs = json.loads(Path(refs_path).read_text())
        if dry_run:
            Path(board_out).write_text(LB.interim_markdown(
                [LB.with_scale(e, refs["test"]["do_nothing"], refs["test"]["perfect"])
                 for e in LB.last_entries(new, load_settings(settings_path)["team_members"])],
                refs))
            return comment_text(entry, info, n)
        Path(interim_path).parent.mkdir(parents=True, exist_ok=True)
        Path(interim_path).write_text(json.dumps(new, ensure_ascii=False, indent=1) + "\n")
        LB.write_interim_board(board_out, interim_path, refs_path, settings_path)
        if git is None:
            return comment_text(entry, info, n)
        if git("add", str(interim_path), str(board_out)) and \
                git("commit", "--quiet", "-m", f"Leaderboard: submission #{entry['issue']}") and \
                git("push", "--quiet", remote or push_url(), f"HEAD:{branch}"):
            return comment_text(entry, info, n)
    return FAILED_TEXT


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--event", required=True, help="the GitHub event JSON ($GITHUB_EVENT_PATH)")
    s.add_argument("--gt", required=True, help="the decrypted test key")
    s.add_argument("--images", default="data/test/images", help="the test photos")
    s.add_argument("--out", required=True, help="the result for the publish step")
    p = sub.add_parser("publish")
    p.add_argument("--result", required=True)
    p.add_argument("--comment", required=True, help="write the comment text here")
    p.add_argument("--branch", default="main")
    p.add_argument("--board-out", default="LEADERBOARD.md")
    p.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "score":
        event = json.loads(Path(args.event).read_text())
        # the key's "file" paths are relative to it: put the test photos next to a copy
        with tempfile.TemporaryDirectory() as keydir:
            gt = Path(keydir) / "gt.json"
            shutil.copyfile(args.gt, gt)
            (Path(keydir) / "images").symlink_to(Path(args.images).resolve())
            result = score_issue(event, gt, args.images)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False))
        print(f"score: {result['status']}")
        return 0
    result = json.loads(Path(args.result).read_text())
    text = publish(result, board_out=args.board_out, branch=args.branch, dry_run=args.dry_run)
    Path(args.comment).write_text(text)
    print(text or "nothing to say (already scored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
