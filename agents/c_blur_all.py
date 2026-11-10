"""Agent C: blur the whole photo. Hides everything and keeps nothing."""
from PIL import ImageFilter

from agents.common import open_rgb, run


def agent(path):
    return open_rgb(path).filter(ImageFilter.GaussianBlur(25)), [], "safe"


if __name__ == "__main__":
    run("C_blur_all", agent, fmt="jpg")
