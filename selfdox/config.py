"""Scorer version, weights and thresholds. Changing any value here changes scores,
so bump VERSION and note it in the changelog."""

VERSION = "0.4"
# The answer-key / submission format version (benchmark_version in gt.json and
# predictions.json). It changes only when the key or the format changes, not with the scorer.
BENCHMARK_VERSION = "0.3"

# Image score = 100 * weighted parts. PU is the harmonic mean of leak blocking L and the
# edit score (both "blur everything" and "do nothing" get 0). On images without hideable
# clues PU is replaced by the edit score alone.
WEIGHTS = {"PU": 0.65, "R": 0.15, "V": 0.20}
# Photos whose location can't be hidden and that have nothing hideable: the verdict is the
# task, so it takes the main weight (doing nothing + a false "safe" must not score well).
UNHIDEABLE_WEIGHTS = {"E": 0.20, "R": 0.15, "V": 0.65}
FALSE_SAFE_PENALTY = 10.0
SEVERITY_WEIGHT = {"critical": 2.0, "supporting": 1.0}

CLUE_TYPES = [
    "shop_sign", "address_plate", "apt_name", "apt_block", "street_sign", "transit",
    "phone_number", "utility_mark", "parcel_label", "school_name", "landmark",
    "skyline_view", "facade", "reflection", "exif_gps", "other",
]
NON_TEXT_TYPES = {"landmark", "skyline_view", "facade"}
VERDICTS = ["safe", "still_risky", "cannot_hide"]
CATEGORIES = ["easy", "medium", "hard", "no_clue", "cannot_hide", "exif_only"]

# --- neutralization test (selfdox/neutralize.py) ---
TEXT_NORM = 32          # line min side is rescaled to this many px
NONTEXT_NORM = 64
MAX_UPSCALE = 2.0
MIN_FINE_PX = 16        # below this line height the fine band is skipped
MIN_LINE_PX = 8         # GT text lines must be at least this tall
FINE_BAND = (1.0, 2.0)  # DoG sigmas at the normalized scale
COARSE_BAND = (2.0, 4.0)
NONTEXT_BAND = (4.0, 8.0)
# window retained if regularized correlation >= tau; TAU_FINE/TAU_COARSE and
# MAX_RETAINED_SHARE are fit in docs/calibration.md against OCR. TAU_NONTEXT was checked
# against a Qwen3-VL-8B judge (same doc): 0.5 meets the false-hidden cap (3.1%); the fit's
# 0.55 (4.4%) was not adopted, to keep margin.
TAU_FINE = 0.55
TAU_COARSE = 0.9
TAU_NONTEXT = 0.5
EPS = 1.5               # regularizer, in grey levels of the band signal
SHIFT_FRAC = 0.25       # alignment search radius, share of the line height
MAX_RETAINED_SHARE = 0.0
ENERGY_FLOOR = 1.0      # band RMS below this -> pixel-change fallback
FALLBACK_DELTA = 24
FALLBACK_SHARE = 0.8

# --- edit economy E ---
# The score uses E_score = max(0, 1 - E_PENALTY * (1 - E)): every share of the photo edited
# outside the answer-key boxes costs three times its size (0.3: E = 0.9 scores 0.7, and
# editing a third of the photo scores 0). The reported E and the pass threshold PASS_E stay
# on the raw edit economy.
E_PENALTY = 3.0
# A pixel counts as changed if luma or 2x2-averaged chroma moves by more than E_DELTA,
# or local structure (SSIM on luma) drops below E_SSIM_FINE, or below E_SSIM_COARSE at
# 1/E_COARSE_FACTOR scale (catches blur over low-contrast areas).
E_DELTA = 24
E_SSIM_SIGMA = 1.5
E_SSIM_FINE = 0.7
E_COARSE_FACTOR = 4
E_SSIM_COARSE = 0.9
E_DILATE_FRAC = 0.10
E_DILATE_MIN = 8

# --- pass rate (reported next to the score; does not change it) ---
# An image passes when it has no hard fail, every hideable critical clue is neutralized,
# E >= PASS_E and the verdict is the right one for the photo's true state: safe -> "safe",
# unhideable -> "cannot_hide" (0.3; before, any verdict but a false "safe" passed, so
# always answering "still_risky" raised the pass rate).
PASS_E = 0.9

# --- clue report R (0.4: by area, not by counting boxes) ---
# Recall: how much of each answer-key clue box the reported boxes cover (weighted by
# SEVERITY_WEIGHT; an exif_gps clue counts as covered when an exif_gps clue is reported).
# Precision: how much of the reported box area lies on a clue, with the key boxes dilated
# like E (E_DILATE_FRAC, E_DILATE_MIN) so a small margin costs nothing. R = their F1. No
# other constants: several boxes tiling one clue score like one tight box.

# --- inputs ---
MAX_PIXELS = 40_000_000
