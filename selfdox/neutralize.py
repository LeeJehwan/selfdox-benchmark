"""Model-free test of whether a GT clue can still be recovered from the edited image.

For each text line we rescale the crop so the line height is TEXT_NORM px, band-pass both
images (difference of Gaussians), find the best alignment of the edited crop against the
original (shifts, flips, sign), and measure a regularized correlation in 2-character
windows. A line is hidden when almost no window keeps the original structure.
"""
from dataclasses import dataclass
from math import ceil

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from . import config as C


@dataclass
class LineResult:
    hidden: bool
    score: float           # best line-level retention over bands (0 = gone, 1 = intact)
    retained_share: float  # share of windows retained in the worst band
    mode: str              # "band" or "pixel_fallback"
    channel: str = ""


def _lab(rgb):
    """sRGB uint8 -> L (0-255 scale), a, b as float64 planes."""
    c = rgb.astype(np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return {"L": L * 2.55, "a": a, "b": b}


def _resize(plane, size, s):
    if abs(s - 1.0) < 1e-9:
        return plane
    img = Image.fromarray(plane.astype(np.float32))
    resample = Image.Resampling.BOX if s < 1 else Image.Resampling.BICUBIC
    return np.asarray(img.resize(size, resample), dtype=np.float64)


def _band(plane, sigmas):
    return gaussian_filter(plane, sigmas[0], mode="reflect") - \
        gaussian_filter(plane, sigmas[1], mode="reflect")


def _box_sum(a, h, w):
    c = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return c[h:, w:] - c[:-h, w:] - c[h:, :-w] + c[:-h, :-w]


def _rcorr(p, q, eps):
    n = p.size
    return float((p * q).sum() / (np.sqrt((p * p).sum()) * np.sqrt((q * q).sum() + n * eps**2)))


def _windows(rh, rw):
    """2-character windows along the long axis of the line, stride 1 character."""
    if rw >= rh:
        size, stride, length = min(rw, 2 * rh), rh, rw
        starts = sorted(set(list(range(0, max(1, length - size + 1), stride)) + [length - size]))
        return [(0, x, rh, size) for x in starts]
    size, stride, length = min(rh, 2 * rw), rw, rh
    starts = sorted(set(list(range(0, max(1, length - size + 1), stride)) + [length - size]))
    return [(y, 0, size, rw) for y in starts]


def _band_retention(P, Qp):
    """P: original band in the line region. Qp: edited band with a margin on every side.
    Returns (line score, [window scores]) at the best alignment of Qp against P."""
    rh, rw = P.shape
    n = P.size
    pnorm = np.sqrt((P * P).sum())
    best = (-1.0, None)
    for k, Qt in enumerate((Qp, Qp[:, ::-1], Qp[::-1, :], Qp[::-1, ::-1])):
        num = fftconvolve(Qt, P[::-1, ::-1], mode="valid")
        q2 = _box_sum(Qt * Qt, rh, rw)
        r = num / (pnorm * np.sqrt(np.maximum(q2, 0) + n * C.EPS**2))
        idx = np.unravel_index(np.argmax(np.abs(r)), r.shape)
        if abs(r[idx]) > best[0]:
            best = (abs(r[idx]), (Qt, idx, np.sign(r[idx]) or 1.0))
    score, (Qt, (dy, dx), sign) = best
    windows = []
    for wy, wx, wh, ww in _windows(rh, rw):
        Pw = P[wy:wy + wh, wx:wx + ww]
        if np.sqrt((Pw * Pw).mean()) < C.ENERGY_FLOOR:
            continue
        rw_best = -1.0
        for ddy in (-1, 0, 1):
            for ddx in (-1, 0, 1):
                y0, x0 = dy + ddy + wy, dx + ddx + wx
                if y0 < 0 or x0 < 0 or y0 + wh > Qt.shape[0] or x0 + ww > Qt.shape[1]:
                    continue
                rw_best = max(rw_best, sign * _rcorr(Pw, Qt[y0:y0 + wh, x0:x0 + ww], C.EPS))
        windows.append(float(rw_best))
    return float(score), windows


def _changed_share(orig, edited, box):
    x, y, w, h = box
    diff = np.abs(orig[y:y + h, x:x + w].astype(int) - edited[y:y + h, x:x + w].astype(int))
    return float((diff.max(axis=2) > C.FALLBACK_DELTA).mean()) if diff.size else 1.0


def line_scores(orig, edited, box, kind="text"):
    """Raw retention scores for one line: {"mode", "channel", "changed", "bands": [
    {"band", "score", "windows"}]}. Thresholds are applied by decide()."""
    H, W = orig.shape[:2]
    x, y, w, h = (int(round(v)) for v in box)
    x, y = max(0, x), max(0, y)
    w, h = max(1, min(w, W - x)), max(1, min(h, H - y))
    norm = C.TEXT_NORM if kind == "text" else C.NONTEXT_NORM
    char = min(w, h)
    s = min(norm / char, C.MAX_UPSCALE)
    if kind == "text":
        bands = [("coarse", C.COARSE_BAND)]
        if char >= C.MIN_FINE_PX:
            bands.insert(0, ("fine", C.FINE_BAND))
    else:
        bands = [("nontext", C.NONTEXT_BAND)]

    m = int(ceil(C.SHIFT_FRAC * char * s)) + 1
    pad = int(ceil((3 * max(b[1][1] for b in bands) + m) / s))
    cx0, cy0 = max(0, x - pad), max(0, y - pad)
    cx1, cy1 = min(W, x + w + pad), min(H, y + h + pad)
    size = (max(1, round((cx1 - cx0) * s)), max(1, round((cy1 - cy0) * s)))
    rx0, ry0 = round((x - cx0) * s), round((y - cy0) * s)
    rw = max(1, min(round(w * s), size[0] - rx0))
    rh = max(1, min(round(h * s), size[1] - ry0))

    O = {k: _resize(v, size, s) for k, v in _lab(orig[cy0:cy1, cx0:cx1]).items()}
    X = {k: _resize(v, size, s) for k, v in _lab(edited[cy0:cy1, cx0:cx1]).items()}

    # pick the channel where the original line has the most band energy
    ref_band = bands[0][1]
    energy = {k: float(np.sqrt((_band(O[k], ref_band)[ry0:ry0 + rh, rx0:rx0 + rw] ** 2).mean()))
              for k in O}
    ch = max(energy, key=energy.get)
    if energy[ch] < C.ENERGY_FLOOR:
        return {"mode": "pixel_fallback", "channel": ch, "bands": [],
                "changed": _changed_share(orig, edited, (x, y, w, h))}
    out = []
    for name, sigmas in bands:
        BO, BX = _band(O[ch], sigmas), _band(X[ch], sigmas)
        P = BO[ry0:ry0 + rh, rx0:rx0 + rw]
        Qp = np.pad(BX, m)[ry0:ry0 + rh + 2 * m, rx0:rx0 + rw + 2 * m]
        score, windows = _band_retention(P, Qp)
        out.append({"band": name, "score": score, "windows": windows})
    return {"mode": "band", "channel": ch, "bands": out, "changed": None}


TAUS = {"fine": "TAU_FINE", "coarse": "TAU_COARSE", "nontext": "TAU_NONTEXT"}


def decide(scores, taus=None, max_share=None):
    """Apply thresholds to line_scores() output -> LineResult."""
    if scores["mode"] == "pixel_fallback":
        ch = scores["changed"]
        return LineResult(hidden=ch >= C.FALLBACK_SHARE, score=1.0 - ch,
                          retained_share=1.0 - ch, mode="pixel_fallback",
                          channel=scores["channel"])
    taus = taus or {k: getattr(C, v) for k, v in TAUS.items()}
    max_share = C.MAX_RETAINED_SHARE if max_share is None else max_share
    shares = []
    for b in scores["bands"]:
        tau = taus[b["band"]]
        w = b["windows"]
        shares.append(sum(r >= tau for r in w) / len(w) if w else float(b["score"] >= tau))
    return LineResult(hidden=not any(sh > max_share for sh in shares),
                      score=float(max(b["score"] for b in scores["bands"])),
                      retained_share=float(max(shares)), mode="band", channel=scores["channel"])


def line_result(orig, edited, box, kind="text"):
    """Is the structure of one GT line (or non-text region) still in the edited image?"""
    return decide(line_scores(orig, edited, box, kind))


def clue_kind(clue):
    return "non_text" if clue.get("type") in C.NON_TEXT_TYPES else "text"


def clue_neutralized(orig, edited, clue):
    """A clue is neutralized when every one of its lines is hidden."""
    if clue.get("bbox") is None:
        raise ValueError("clue has no bbox (exif_gps clues are checked on metadata)")
    lines = clue.get("lines") or [clue["bbox"]]
    kind = clue_kind(clue)
    return all(line_result(orig, edited, ln, kind=kind).hidden for ln in lines)
