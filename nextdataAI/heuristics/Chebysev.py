__all__ = ["Chebysev"]

from .Heuristic import Heuristic


class Chebysev(Heuristic):
    def __init__(self):
        super().__init__(name="Cebysev", formula=lambda x1, y1, x2, y2: max(abs(x1 - x2), abs(y1 - y2)))

    def __call__(self, *args) -> int:
        """
        :type args: tuple
        :param args: start, target
        :return: distance between start and target (Chebyshev distance)
        """
        start = args[0]
        target = args[1]
        x1, y1 = start
        x2, y2 = target
        return self.formula(x1, y1, x2, y2)
