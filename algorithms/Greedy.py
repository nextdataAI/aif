from typing import Union
from .Algorithm import Algorithm
import time
from .utils import get_valid_moves, manhattan_distance, get_heuristic

__all__ = ['Greedy']


class Greedy(Algorithm):
    def __init__(self, env_name: str = "MiniHack-MazeWalk-15x15-v0", h: Union[callable, str] = manhattan_distance, name: str = "Greedy"):
        super().__init__(env_name, name)
        self.h = get_heuristic(h) if isinstance(h, str) else h

    def __call__(self, seed: int, return_visited: bool = False, return_time: bool = False):
        start_time = time.time()
        local_env, local_state, local_game_map, start, target = super().initialize_env(seed)

        queue = [start]
        visited = set()
        path = []
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                path.append(node)
                if node == target:
                    return path, list(visited), time.time() - start_time
                for neighbor in get_valid_moves(local_game_map, node):
                    queue.append(neighbor)
                queue.sort(key=lambda x: self.h(x, target))
        return None, list(visited), time.time() - start_time
