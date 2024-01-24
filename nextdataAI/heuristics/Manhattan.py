from .Heuristic import Heuristic

__all__ = ["Manhattan"]


class Manhattan(Heuristic):
    def __init__(self):
        super().__init__(name="Manhattan", formula=lambda x1, y1, x2, y2: abs(x1 - x2) + abs(y1 - y2))

    def __call__(self, *args) -> int:
        """
        Manhattan distance heuristic.
        :type args: tuple
        :param args: start, target, others
        :return: distance between start and target (Manhattan distance)
        """
        start = args[0]
        target = args[1]
        x1, y1 = start
        x2, y2 = target
        return self.formula(x1, y1, x2, y2)
