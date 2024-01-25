from .Algorithm import Algorithm
from ..utils import get_valid_moves
from ..AnimateGif import Animator
__all__ = ['BFS', 'DFS']


class FS(Algorithm):
    def __init__(self, name: str, env_name: str = "MiniHack-MazeWalk-15x15-v0", informed: bool = True,
                 pop_index=0, animate: bool = False):
        super().__init__(env_name, name, animate)
        self.informed = informed
        self.pop_index = pop_index  # BFS default

    def __call__(self, seed: int):
        self.start_timer()
        local_env, local_state, local_game_map, start, target = super().initialize_env(seed)

        queue = []
        visited = set()
        queue.append(start)
        visited.add(start)
        parent = {start: None}
        while queue:
            node = queue.pop(self.pop_index)
            if node == target:
                path = self.build_path(parent, target)
                time = self.stop_timer()
                if self.animate:
                    Animator(size=self.size, game_map=local_state, path=path, visited=visited, file_path=f'{self.name}.gif')()
                return True, list(path), list(visited), time
            for neighbor in get_valid_moves(local_game_map, node):
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)
                    parent[neighbor] = node
        return False, None, None, self.stop_timer()


class BFS(FS):
    def __init__(self, env_name: str = "MiniHack-MazeWalk-15x15-v0", informed: bool = True, name='BFS', animate=False):
        super().__init__(env_name=env_name, informed=informed, name=name, pop_index=0, animate=animate)

    def __call__(self, seed: int):
        return super().__call__(seed)


class DFS(FS):
    def __init__(self, env_name: str = "MiniHack-MazeWalk-15x15-v0", informed: bool = True, name='DFS', animate=False):
        super().__init__(env_name=env_name, informed=informed, name=name, pop_index=-1, animate=animate)

    def __call__(self, seed: int):
        return super().__call__(seed)