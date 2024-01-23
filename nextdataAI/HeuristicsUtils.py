from .heuristics.Manhattan import Manhattan
from .heuristics.Euclidean import Euclidean
from .heuristics.SManhattan import SManhattan
from .heuristics.Chebysev import Chebysev
from .pseudo_heuristics.NNManhattan import NNManhattan

heuristics = dict(manhattan=Manhattan, euclidean=Euclidean, smanhattan=SManhattan, chebysev=Chebysev,
                  nnmanhattan=NNManhattan)

__all__ = ['get_heuristic']


def get_heuristic(heuristic: str):
    if heuristic.lower() in heuristics.keys():
        h = heuristics[heuristic.lower()]()
        return h
    else:
        raise Exception("Heuristic not supported!")