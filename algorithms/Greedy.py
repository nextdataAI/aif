from typing import Union

from algorithms.utils import get_valid_moves, get_heuristic
from .Algorithm import Algorithm
from .heuristics.Heuristic import Heuristic

__all__ = ['Greedy']


class Greedy(Algorithm):
    def __init__(self, env_name: str = "MiniHack-MazeWalk-15x15-v0", h: Union[callable, str, Heuristic] = 'manhattan', name: str = "Greedy"):
        super().__init__(env_name, name)
        self.h = get_heuristic(h) if isinstance(h, str) else h if isinstance(h, Heuristic) else h

    def __call__(self, seed: int):
        self.start_timer()
        local_env, local_state, local_game_map, start, target = super().initialize_env(seed)

        queue = [start]
        visited = []
        path = []
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.append(node)
                path.append(node)
                if node == target:
                    return True, path, list(visited), self.stop_timer()
                for neighbor in get_valid_moves(local_game_map, node):
                    queue.append(neighbor)
                queue.sort(key=lambda x: self.h(x, target))
        return False, None, list(visited), self.stop_timer()
