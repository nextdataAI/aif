if __name__ == '__main__':
    from .heuristics import Manhattan, Euclidean, SManhattan, Chebysev
    from .pseudo_heuristics import NNManhattan

    heuristics = {
        'manhattan': Manhattan(),
        'euclidean': Euclidean(),
        'smanhattan': SManhattan(),
        'chebysev': Chebysev(),
        'nnmanhattan': NNManhattan(),
    }

__all__ = ['get_heuristic']


def get_heuristic(heuristic: str):
    if heuristic.lower() in heuristics.keys():
        h = heuristics[heuristic.lower()]
        return h
    else:
        raise Exception("Heuristic not supported!")