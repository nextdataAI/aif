from gym.core import Env
from .utils import get_valid_moves, get_player_location, get_target_location
from typing import Tuple, List

__all__ = ['bfs', 'dfs']


def fs_search(env: Env, seed: int, informed: bool = True, kind: str = 'bfs') -> List[Tuple[int, int]]:
    local_env = env.unwrapped
    local_env.seed(seed)
    local_state = local_env.reset()
    local_game_map = local_state.get('chars')
    start = get_player_location(local_game_map)
    target = get_target_location(local_game_map) if informed else None

    queue = [start]
    visited = set(start)
    path = set(start)

    while queue:
        node = queue.pop(0) if kind == 'bfs' else queue.pop()
        for neighbor in get_valid_moves(local_game_map, node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                path.add(neighbor)
                if informed and neighbor == target:
                    return list(path)
                elif not informed and neighbor == '@':
                    return list(path)


def bfs(env: Env, seed: int, informed: bool = True) -> List[Tuple[int, int]]:
    return fs_search(env, seed, informed, 'bfs')


def dfs(env: Env, seed: int, informed: bool = True) -> List[Tuple[int, int]]:
    return fs_search(env, seed, informed, 'dfs')
