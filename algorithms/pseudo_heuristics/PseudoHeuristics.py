__all__ = ["PseudoHeuristic"]


class PseudoHeuristic:
    def __init__(self, name):
        print(f"WARNING:{name}:This is a pseudo-heuristic. It is not a real heuristic.")
        self.name = name

    def __call__(self, *args) -> int:
        raise NotImplementedError("This method must be implemented by the subclass.")
