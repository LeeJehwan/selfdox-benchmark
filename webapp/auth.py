"""Team codes and the team proof. A code is issued once by tools/site_admin.py; the public
state keeps only its sha256, so the site can recognise a typed code. A submission carries
proof = HMAC-SHA256(code, lower(HF username)): only someone who knows the code can make
it, and it is useless under another HF account. webapp/static/app.js computes the same."""
import hashlib
import hmac
import secrets

PREFIX = "sdx-"


def new_code():
    return PREFIX + secrets.token_urlsafe(18)


def code_hash(code):
    return hashlib.sha256(str(code).strip().encode()).hexdigest()


def proof(code, username):
    return hmac.new(str(code).strip().encode(), str(username).strip().lower().encode(),
                    hashlib.sha256).hexdigest()


MAX_PROOF_LEN = 128   # a real proof is 64 hex chars


def team_for_proof(codes, username, given):
    """The team whose code gives this proof for this HF username, or None.
    `given` comes from an untrusted PR file, so anything not a short string is refused,
    and the comparison is on UTF-8 bytes (compare_digest raises on non-ASCII str)."""
    if not isinstance(given, str) or not given or len(given) > MAX_PROOF_LEN:
        return None
    if not str(username).strip():
        return None
    try:
        given_b = given.encode("utf-8")
    except UnicodeEncodeError:
        return None
    hit = None
    for team, code in codes.items():
        if hmac.compare_digest(proof(code, username).encode("utf-8"), given_b):
            hit = team
    return hit
