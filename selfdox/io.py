"""Image decoding and metadata checks for agent outputs.

Only JPEG and PNG are accepted. Metadata is parsed from the raw bytes (not through a
viewer library) so nothing that could leak the location slips through: GPS in EXIF or
XMP, embedded thumbnails or MPF images, data after the end marker, orientation tags.
"""
import re
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import config as C

GPS_TEXT = re.compile(rb"GPS(Latitude|Longitude|Position)", re.I)


class ImageError(Exception):
    pass


@dataclass
class MetaReport:
    format: str
    problems: list = field(default_factory=list)   # hard fails
    warnings: list = field(default_factory=list)

    def add(self, code):
        if code not in self.problems:
            self.problems.append(code)


def safe_join(root, rel):
    """root/rel, or None if rel is absolute or escapes root (also through symlinks)."""
    p = Path(str(rel))
    if not str(rel) or p.is_absolute() or ".." in p.parts:
        return None
    base = Path(root).resolve()
    full = (base / p).resolve()
    return full if full.is_relative_to(base) else None


def load_rgb(path, max_pixels=C.MAX_PIXELS):
    """Decode to an HxWx3 uint8 array. EXIF orientation is NOT applied; alpha is dropped
    so colours under transparent pixels count (they are still in the file)."""
    try:
        with Image.open(path) as im:
            if im.width * im.height > max_pixels:
                raise ImageError(f"too_large ({im.width}x{im.height})")
            im.load()
            if im.mode.startswith("I;16") or im.mode == "I":
                a = np.asarray(im).astype(np.float64)
                a = np.clip(a / (257.0 if a.max() > 255 else 1.0), 0, 255).round().astype(np.uint8)
                return np.repeat(a[..., None], 3, axis=2)
            if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
                return np.asarray(im.convert("RGBA"))[..., :3].copy()
            return np.asarray(im.convert("RGB")).copy()
    except ImageError:
        raise
    except Exception as e:  # PIL raises many types, incl. DecompressionBombError
        raise ImageError(f"undecodable: {e}") from e


# --- TIFF / EXIF ---

def _tiff_info(blob):
    """Return (orientation, has_gps_latlon, has_gps_ifd, has_ifd1_image)."""
    e = "<" if blob[:2] == b"II" else ">"
    if blob[:2] not in (b"II", b"MM") or struct.unpack(e + "H", blob[2:4])[0] != 42:
        raise ValueError("not a TIFF header")

    def ifd(off):
        n = struct.unpack(e + "H", blob[off:off + 2])[0]
        tags = {}
        for i in range(n):
            ent = blob[off + 2 + 12 * i: off + 14 + 12 * i]
            tag, typ, cnt = struct.unpack(e + "HHI", ent[:8])
            tags[tag] = (typ, cnt, ent[8:12])
        nxt = struct.unpack(e + "I", blob[off + 2 + 12 * n: off + 6 + 12 * n])[0]
        return tags, nxt

    tags0, next1 = ifd(struct.unpack(e + "I", blob[4:8])[0])
    orientation = None
    if 0x0112 in tags0:
        orientation = struct.unpack(e + "H", tags0[0x0112][2][:2])[0]
    has_gps_ifd = has_latlon = False
    if 0x8825 in tags0:
        has_gps_ifd = True
        gtags, _ = ifd(struct.unpack(e + "I", tags0[0x8825][2])[0])
        has_latlon = 2 in gtags or 4 in gtags
    has_ifd1 = False
    if next1:
        tags1, _ = ifd(next1)
        has_ifd1 = bool({0x0201, 0x0111} & set(tags1))
    return orientation, has_latlon, has_gps_ifd, has_ifd1


def _check_exif(blob, rep):
    if blob.startswith(b"Exif\x00\x00"):
        blob = blob[6:]
    try:
        orientation, latlon, gps_ifd, ifd1 = _tiff_info(blob)
    except Exception:
        rep.add("malformed_exif")
        return
    if latlon:
        rep.add("gps_exif")
    elif gps_ifd:
        rep.warnings.append("empty_gps_ifd")
    if orientation not in (None, 1):
        rep.add("orientation")
    if ifd1:
        rep.add("exif_thumbnail")


def _check_xmp(text, rep):
    if GPS_TEXT.search(text):
        rep.add("gps_xmp")


# --- containers ---

def _jpeg(data, rep):
    i, end = 2, None
    while i < len(data) - 1:
        if data[i] != 0xFF:
            raise ValueError("bad marker")
        while i < len(data) and data[i] == 0xFF:
            i += 1
        marker = data[i]
        i += 1
        if marker == 0xD9:
            end = i
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        length = struct.unpack(">H", data[i:i + 2])[0]
        payload = data[i + 2:i + length]
        i += length
        if marker == 0xE1 and payload.startswith(b"Exif\x00\x00"):
            _check_exif(payload, rep)
        elif marker == 0xE1:
            _check_xmp(payload, rep)
        elif marker == 0xE2 and payload.startswith(b"MPF\x00"):
            rep.add("mpf")
        elif GPS_TEXT.search(payload):
            rep.add("gps_xmp")
        if marker == 0xDA:  # skip entropy-coded data up to the next real marker
            while i < len(data) - 1:
                if data[i] == 0xFF and not (data[i + 1] == 0x00 or 0xD0 <= data[i + 1] <= 0xD7
                                            or data[i + 1] == 0xFF):
                    break
                i += 1
    if end is None:
        raise ValueError("no EOI marker")
    if data[end:].strip(b"\x00"):
        rep.add("trailing_data")


def _png_text(typ, body):
    key, _, rest = body.partition(b"\x00")
    if typ == b"tEXt":
        return key, rest
    if typ == b"zTXt":
        return key, zlib.decompress(rest[1:])
    flag, _method = rest[0], rest[1]
    rest = rest[2:]
    _lang, _, rest = rest.partition(b"\x00")
    _tkey, _, text = rest.partition(b"\x00")
    return key, zlib.decompress(text) if flag else text


def _png(data, rep):
    i, end = 8, None
    while i + 8 <= len(data):
        length = struct.unpack(">I", data[i:i + 4])[0]
        typ = data[i + 4:i + 8]
        body = data[i + 8:i + 8 + length]
        i += 12 + length
        if typ == b"eXIf":
            _check_exif(body, rep)
        elif typ in (b"tEXt", b"zTXt", b"iTXt"):
            key, text = _png_text(typ, body)
            if key.lower().startswith(b"raw profile type"):
                hexpart = b"".join(text.split(b"\n")[3:])
                _check_exif(bytes.fromhex(hexpart.decode()), rep)
            else:
                _check_xmp(text, rep)
        if typ == b"IEND":
            end = i
            break
    if end is None:
        raise ValueError("no IEND chunk")
    if data[end:]:
        rep.add("trailing_data")


def metadata_report(path):
    data = open(path, "rb").read()
    if data[:2] == b"\xff\xd8":
        rep, parse = MetaReport("jpeg"), _jpeg
    elif data[:8] == b"\x89PNG\r\n\x1a\n":
        rep, parse = MetaReport("png"), _png
    else:
        rep = MetaReport("other")
        rep.add("unsupported_format")
        return rep
    try:
        parse(data, rep)
    except Exception:
        rep.add("malformed_file")
    if GPS_TEXT.search(data) and "gps_exif" not in rep.problems:
        rep.add("gps_xmp")
    return rep
