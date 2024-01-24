from .Heuristic import Heuristic

__all__ = ["SManhattan"]


class SManhattan(Heuristic):
    def __init__(self):
        super().__init__(name="SManhattam", formula=lambda x1, y1, x2, y2: (abs(x1 - x2) + abs(y1 - y2)) * 3)

    def __call__(self, *args) -> int:
        """
        :type args: tuple
        :param args: start, target
        :return: Mean Absolute Error between start and target ('Square' Manhattan distance)
        """
        start = args[0]
        target = args[1]
        x1, y1 = start
        x2, y2 = target
        return self.formula(x1, y1, x2, y2)
