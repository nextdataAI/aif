__all__ = ["Heuristic"]


class Heuristic:
    def __init__(self, name, formula):
        self.name = name
        self.formula = formula

    def __call__(self, *args) -> int:
        """
        :type args: any
        :param args: start, target
        :return: Custom Distance between start and target
        """
        start = args[0]
        target = args[1]
        x1, y1 = start
        x2, y2 = target
        return self.formula(x1, y1, x2, y2)
