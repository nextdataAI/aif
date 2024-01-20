from .Heuristic import Heuristic

class Manhattan(Heuristic):
    def __init__(self):
        super().__init__(name="Manhattan")

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
        return abs(x1 - x2) + abs(y1 - y2)