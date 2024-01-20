class Heuristic:
    def __init__(self, name):
        self.name = name

    def __call__(self, *args) -> int:
        raise NotImplementedError("This method must be implemented by the subclass.")
