from typing import Tuple, List, Any, Union
import pandas as pd
import math
from platform import system as system_name  # Returns the system/OS name
from subprocess import call as system_call  # Execute a shell command
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from functools import partial

__all__ = ['get_player_location', 'get_target_location', 'get_edges_location', 'is_wall', 'get_heuristic', 'animate',
           'get_valid_moves', 'actions_from_path', 'euclidean_distance', 'manhattan_distance', 'clear_screen',
           'get_distances']


def assign_scores_and_retrieve_ordered_list(paths_time_took_name: list):
    """
    Assigns a score to each path and returns a list of paths ordered by score.
    """
    ordered_paths = []
    for path, time_took, name in paths_time_took_name:
        score = 0
        for i in range(len(path) - 1):
            score += abs(path[i][0] - path[i + 1][0]) + abs(path[i][1] - path[i + 1][1])
        path_length = len(path)
        ordered_paths.append((score, path_length, time_took, path, name))
    ordered_paths.sort(key=lambda x: x[0])
    # Normalize scores
    max_score = ordered_paths[-1][0]
    for i in range(len(ordered_paths)):
        ordered_paths[i] = (
            1 - ordered_paths[i][0] / max_score, ordered_paths[i][1], ordered_paths[i][2], ordered_paths[i][3],
            ordered_paths[i][4])

    # return pandas df
    return pd.DataFrame(ordered_paths, columns=['score', 'path_length', 'took', 'path', 'name'])


def get_player_location(game_map: np.ndarray, symbol: str = "@"):
    x, y = np.where(game_map == ord(symbol))
    return (x[0], y[0]) if len(x) > 0 else (None, None)


def get_target_location(game_map: np.ndarray, symbol: str = ">"):
    x, y = np.where(game_map == ord(symbol))
    return x[0], y[0]


def get_edges_location(game_map: np.ndarray, symbol: str = "X") -> tuple[
    np.ndarray[Any, np.dtype[Union[np.signedinteger[Any], np.longlong]]],
    np.ndarray[Any, np.dtype[Union[np.signedinteger[Any], np.longlong]]]]:
    x, y = np.where(game_map == ord(symbol))
    return x, y


def is_wall(position_element: Union[int, chr]) -> bool:
    obstacles = "|- "
    return chr(position_element) in obstacles


def get_valid_moves(game_map: np.ndarray, current_position: Tuple[int, int], mode='coord') -> List[Tuple[int, int]]:
    x_limit, y_limit = game_map.shape
    valid = []
    x, y = current_position
    # North
    if (y - 1 > 0) and not is_wall(game_map[x, y - 1]):
        if mode == 'coord':
            valid.append((x, y - 1))
        elif mode == 'action':
            valid.append(0)
        elif mode == 'both':
            valid.append((0, (x, y - 1)))
        # East
    if x + 1 < x_limit and not is_wall(game_map[x + 1, y]):
        if mode == 'coord':
            valid.append((x + 1, y))
        elif mode == 'action':
            valid.append(1)
        elif mode == 'both':
            valid.append((1, (x + 1, y)))
        # South
    if y + 1 < y_limit and not is_wall(game_map[x, y + 1]):
        if mode == 'coord':
            valid.append((x, y + 1))
        elif mode == 'action':
            valid.append(2)
        elif mode == 'both':
            valid.append((2, (x, y + 1)))
        # West
    if x - 1 > 0 and not is_wall(game_map[x - 1, y]):
        if mode == 'coord':
            valid.append((x - 1, y))
        elif mode == 'action':
            valid.append(3)
        elif mode == 'both':
            valid.append((3, (x - 1, y)))

    return valid


def actions_from_path(start: Tuple[int, int], path: List[Tuple[int, int]]) -> List[int]:
    action_map = {
        "N": 0,
        "E": 1,
        "S": 2,
        "W": 3
    }
    actions = []
    x_s, y_s = start
    for (x, y) in path:
        if x_s == x:
            if y_s > y:
                actions.append(action_map["W"])
            else:
                actions.append(action_map["E"])
        elif y_s == y:
            if x_s > x:
                actions.append(action_map["N"])
            else:
                actions.append(action_map["S"])
        else:
            raise Exception("x and y can't change at the same time. oblique moves not allowed!")
        x_s = x
        y_s = y

    return actions


def euclidean_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def manhattan_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> int:
    x1, y1 = point1
    x2, y2 = point2
    return abs(x1 - x2) + abs(y1 - y2)


def manhattan_distance_with_obstacles(point1: Tuple[int, int], point2: Tuple[int, int], game_map: np.ndarray) -> float:
    x1, y1 = point1
    x2, y2 = point2
    out = 0
    for i in range(x1, x2):
        for j in range(y1, y2):
            if is_wall(game_map[i, j]):
                out += 1
    return - (abs(x1 - x2) + abs(y1 - y2) + out)


heuristics = {
    'manhattan': manhattan_distance,
    'euclidean': euclidean_distance
}


def get_heuristic(heuristic: str):
    if heuristic in heuristics.keys():
        return heuristics[heuristic]
    else:
        raise Exception("Heuristic not supported!")


def clear_screen():
    """
    Clears the terminal screen.
    """

    # Clear screen command as function of OS
    command = 'cls' if system_name().lower().startswith('win') else 'clear'

    # Action
    system_call([command])


def update_image(im, *, data):
    im.set_data(data)
    return im


def animate(fig, im, data):
    animation.FuncAnimation(fig, partial(update_image, im=im, data=data), frames=data,
                            interval=1)
    plt.show()


def get_distances(game_map: np.ndarray):
    # retrive the player distance from the nearest walls in each direction
    player_pos = get_player_location(game_map)
    x, y = player_pos
    # North
    north = -1
    east = -1
    south = -1
    west = -1
    north_east = -1
    north_west = -1
    south_east = -1
    south_west = -1
    for i, j in enumerate(game_map):
        if north == -1 and is_wall(game_map[x, y - i]):
            north = i - 1
        if east == -1 and is_wall(game_map[x + i, y]):
            east = i - 1
        if south == -1 and is_wall(game_map[x, y + i]):
            south = i - 1
        if west == -1 and is_wall(game_map[x - i, y]):
            west = i - 1
        if north_east == -1 and is_wall(game_map[x + i, y - i]):
            north_east = i - 1
        if north_west == -1 and is_wall(game_map[x - i, y - i]):
            north_west = i - 1
        if south_east == -1 and is_wall(game_map[x + i, y + i]):
            south_east = i - 1
        if south_west == -1 and is_wall(game_map[x - i, y + i]):
            south_west = i - 1
        if (north != -1 and east != -1 and south != -1 and west != -1 and north_east != -1
                and north_west != -1 and south_east != -1 and south_west != -1):
            break
    # if target is not None:
    #     return [north, east, south, west, north_east, north_west, south_east, south_west,
    #             euclidean_distance(player_pos, target)]
    return [east, north, west, south, south_east, north_east, south_west, north_west]
