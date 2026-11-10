"""Agent D: blur every text box EasyOCR finds (Korean + English), strip metadata, say safe.

Misses non-text clues (landmarks, skylines) and blurs harmless text too.
Needs the `ocr` extra: uv sync --extra ocr
"""
import numpy as np
from PIL import ImageFilter

from agents.common import has_gps, open_rgb, run

_reader = None


def reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
    return _reader


def agent(path):
    im = open_rgb(path)
    clues = []
    for poly, text, conf in reader().readtext(np.asarray(im)):
        if conf < 0.3:
            continue
        xs, ys = [float(p[0]) for p in poly], [float(p[1]) for p in poly]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        pad = 0.2 * (y1 - y0)
        box = [int(max(0, x0 - pad)), int(max(0, y0 - pad)),
               int(min(im.width, x1 + pad)), int(min(im.height, y1 + pad))]
        patch = im.crop(box).filter(ImageFilter.GaussianBlur(max(6, (y1 - y0) / 2)))
        im.paste(patch, box[:2])
        clues.append({"bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                      "type": "other", "text": text, "reason": "readable text"})
    if has_gps(path):
        clues.append({"bbox": None, "type": "exif_gps", "reason": "GPS in EXIF"})
    return im, clues, "safe"


if __name__ == "__main__":
    run("D_ocr_blur", agent)
