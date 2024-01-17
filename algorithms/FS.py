from gym.core import Env
from .utils import get_valid_moves
from .Algorithm import Algorithm

__all__ = ['BFS', 'DFS']


class FS(Algorithm):
    def __init__(self, env_name: str = "MiniHack-MazeWalk-15x15-v0", informed: bool = True, name: str = "BFS", pop_index=0):
        super().__init__(env_name, name)
        self.informed = informed
        self.pop_index = pop_index  # BFS default

    def __call__(self, seed: int):
        local_env, local_state, local_game_map, start, target = super().initialize_env(seed)

        queue = [start]
        visited = set(start)
        path = set(start)

        while queue:
            node = queue.pop(self.pop_index)
            for neighbor in get_valid_moves(local_game_map, node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    path.add(neighbor)
                    if self.informed and neighbor == target:
                        return list(path)
                    elif not self.informed and neighbor == '@':
                        return list(path)


class BFS(FS):
    def __init__(self, env_name: str = "MiniHack-MazeWalk-15x15-v0", informed: bool = True):
        super().__init__(env_name=env_name, informed=informed, name='BFS', pop_index=0)

    def __call__(self, seed: int):
        super().__call__(seed)


class DFS(FS):
    def __init__(self, env_name: str = "MiniHack-MazeWalk-15x15-v0", informed: bool = True):
        super().__init__(env_name=env_name, informed=informed, name='DFS', pop_index=-1)

    def __call__(self, seed: int):
        super().__call__(seed)
