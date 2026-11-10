"""Agent B: strip metadata only (what today's upload tools do), keep every pixel."""
from agents.common import has_gps, open_rgb, run


def agent(path):
    clues = [{"bbox": None, "type": "exif_gps", "reason": "GPS in EXIF"}] if has_gps(path) else []
    return open_rgb(path), clues, "safe"


if __name__ == "__main__":
    run("B_exif_strip", agent, fmt="jpg")
