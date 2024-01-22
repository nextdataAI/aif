from .Algorithm import Algorithm
from .utils import get_valid_moves

__all__ = ['BFS', 'DFS']


class FS(Algorithm):
    def __init__(self, name: str, env_name: str = "MiniHack-MazeWalk-15x15-v0", informed: bool = True, pop_index=0):
        super().__init__(env_name, name)
        self.informed = informed
        self.pop_index = pop_index  # BFS default

    def __call__(self, seed: int):
        self.start_timer()
        local_env, local_state, local_game_map, start, target = super().initialize_env(seed)

        queue = [start]
        visited = [start]
        path = [start]

        while queue:
            node = queue.pop(self.pop_index)
            for neighbor in get_valid_moves(local_game_map, node):
                if neighbor not in visited:
                    visited.append(neighbor)
                    queue.append(neighbor)
                    path.append(neighbor)
                    if neighbor == target or neighbor == '@':
                        return True, list(path), list(visited), self.stop_timer()


class BFS(FS):
    def __init__(self, env_name: str = "MiniHack-MazeWalk-15x15-v0", informed: bool = True, name='BFS'):
        super().__init__(env_name=env_name, informed=informed, name=name, pop_index=0)

    def __call__(self, seed: int):
        return super().__call__(seed)


class DFS(FS):
    def __init__(self, env_name: str = "MiniHack-MazeWalk-15x15-v0", informed: bool = True, name='DFS'):
        super().__init__(env_name=env_name, informed=informed, name=name, pop_index=-1)

    def __call__(self, seed: int):
        return super().__call__(seed)
