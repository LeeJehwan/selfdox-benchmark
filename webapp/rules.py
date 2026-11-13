"""Submission rules: the deadline (against the PR's creation time and its last change on
the Hub's clock) and at most MAX_SUBMISSIONS scored submissions per team. Rejected and
failed ones don't count."""
import datetime

MAX_SUBMISSIONS = 5
PUBLIC = ("pr", "team", "status", "created_at", "scored_at", "agent_sha256", "models", "n",
          "reason", "warnings")
RESULT = ("score", "ci", "raw", "raw_ci", "cleared", "leak", "false_safe", "n_safe", "pass",
          "n_zero")


class Closed(Exception):
    """The submission can't count; the message goes into the PR comment."""


def parse_deadline(value):
    """As tools.leaderboard.load_settings: date and time, no offset = UTC, date alone refused."""
    if not value:
        return None
    if "T" not in str(value):
        raise ValueError(f"deadline {value!r}: give a time too, e.g. 2026-11-30T23:59:59+09:00")
    d = datetime.datetime.fromisoformat(str(value))
    return d if d.tzinfo else d.replace(tzinfo=datetime.timezone.utc)


def max_of(settings):
    return int((settings or {}).get("max_submissions") or MAX_SUBMISSIONS)


def scored(records, team):
    return [r for r in records if r.get("team") == team and r.get("status") == "scored"]


def left(records, team, settings):
    return max(0, max_of(settings) - len(scored(records, team)))


def _utc(t):
    return t if t.tzinfo else t.replace(tzinfo=datetime.timezone.utc)   # naive = UTC


def check_can_submit(records, team, settings, created_at, changed_at=None):
    """`changed_at`: the PR's last commit or status change (the queue's last_change), so
    files pushed to an open PR, or a draft published, after the deadline don't count."""
    deadline = parse_deadline((settings or {}).get("deadline_utc"))
    if deadline and _utc(created_at) > deadline:
        raise Closed(f"submitted after the deadline ({deadline.isoformat()})")
    if deadline and changed_at is not None and _utc(changed_at) > deadline:
        raise Closed(f"the pull request was changed after the deadline ({deadline.isoformat()})")
    if left(records, team, settings) <= 0:
        raise Closed(f"your team has used all {max_of(settings)} submissions")


def public_view(record):
    """What the site shows of a submission: aggregates only, no file paths or photo ids."""
    out = {k: record.get(k) for k in PUBLIC}
    res = record.get("result")
    out["result"] = {k: res.get(k) for k in RESULT} if res else None
    return out
