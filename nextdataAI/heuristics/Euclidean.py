from .Heuristic import Heuristic

__all__ = ["Euclidean"]


class Euclidean(Heuristic):
    def __init__(self):
        super().__init__(name="Euclidean", formula=lambda x1, y1, x2, y2: ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)

    def __call__(self, *args) -> int:
        """
        :type args: tuple
        :param args: start, target
        :return: distance between start and target (Euclidean distance)
        """
        start = args[0]
        target = args[1]
        x1, y1 = start
        x2, y2 = target
        return self.formula(x1, y1, x2, y2)
