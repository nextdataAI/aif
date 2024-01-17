from typing import Union

from gym.core import Env
from .utils import get_valid_moves, get_player_location, get_target_location, manhattan_distance, get_heuristic

__all__ = ['greedy_search']


def greedy_search(env: Env, seed: int, h: Union[callable | str] = manhattan_distance):
    local_env = env.unwrapped
    local_env.seed(seed)
    local_state = local_env.reset()
    local_game_map = local_state.get('chars')
    start = get_player_location(local_game_map)
    target = get_target_location(local_game_map)

    h = get_heuristic(h) if isinstance(h, str) else h

    queue = [start]
    visited = set()
    path = []
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            path.append(node)
            if node == target:
                return path
            for neighbor in get_valid_moves(local_game_map, node):
                queue.append(neighbor)
            queue.sort(key=lambda x: h(x, target))
    return path
