from typing import Union

from nextdataAI.HeuristicsUtils import get_heuristic
from .Algorithm import Algorithm
from ..utils import get_valid_moves

__all__ = ['Greedy']
from ..AnimateGif import Animator


class Greedy(Algorithm):
    def __init__(self, env_name: str = "MiniHack-MazeWalk-15x15-v0", h: Union[callable, str] = 'manhattan',
                 name: str = "Greedy", animate: bool = False):
        super().__init__(env_name, name, animate)
        self.h = get_heuristic(h) if isinstance(h, str) else h

    def __call__(self, seed: int):
        self.start_timer()
        local_env, local_state, local_game_map, start, target = super().initialize_env(seed)

        queue = []
        visited = set()
        visited.add(start)
        queue.append(start)
        parent = {start: None}
        while queue:
            node = queue.pop(0)
            if node == target:
                path = self.build_path(parent, target)
                time = self.stop_timer()
                if self.animate:
                    Animator(size=self.size, game_map=local_state, path=list(path),
                             visited=list(visited), file_path=f'{self.name}.gif')()
                return True, path, list(visited), time
            for neighbor in get_valid_moves(local_game_map, node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = node
                    queue.append(neighbor)
            queue.sort(key=lambda x: self.h(x, target))
        return False, None, list(visited), self.stop_timer()
