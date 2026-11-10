"""Agent A: do nothing. Returns the input file unchanged and says it is safe."""
from agents.common import run


def agent(path):
    return None, [], "safe"


if __name__ == "__main__":
    run("A_noop", agent)
