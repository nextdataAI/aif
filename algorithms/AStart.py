from queue import PriorityQueue
from gym.core import Env
from .utils import get_valid_moves, get_player_location, get_target_location, get_heuristic
from typing import Union, Optional, Any


def build_path(parent, target):
    path = []
    while target is not None:
        path.append(target)
        target = parent[target]
    path.reverse()
    return path


def a_star(env: Env, seed: int, h: Union[callable | str] = "manhattan") -> Optional[list[Any]]:
    local_env = env.unwrapped
    local_env.seed(seed)
    local_state = local_env.reset()
    local_game_map = local_state.get('chars')
    start = get_player_location(local_game_map)
    target = get_target_location(local_game_map)

    h = get_heuristic(h) if isinstance(h, str) else h
    # initialize open and close list
    open_list = PriorityQueue()
    close_list = []
    # additional dict which maintains the nodes in the open list for an easier access and check
    support_list = {}

    starting_state_g = 0
    starting_state_h = h(start, target)
    starting_state_f = starting_state_g + starting_state_h

    open_list.put((starting_state_f, (start, starting_state_g)))
    support_list[start] = starting_state_g
    parent = {start: None}

    while not open_list.empty():
        # get the node with lowest f
        _, (current, current_cost) = open_list.get()
        # add the node to the close list
        close_list.append(current)

        if current == target:
            print("Target found!")
            path = build_path(parent, target)
            return path

        for neighbor in get_valid_moves(local_game_map, current):
            # check if neighbor in close list, if so continue
            if neighbor in close_list:
                continue

            # compute neighbor g, h and f values
            neighbor_g = 1 + current_cost
            neighbor_h = h(neighbor, target)
            neighbor_f = neighbor_g + neighbor_h
            parent[neighbor] = current
            neighbor_entry = (neighbor_f, (neighbor, neighbor_g))
            # if neighbor in open_list
            if neighbor in support_list.keys():
                # if neighbor_g is greater or equal to the one in the open list, continue
                if neighbor_g >= support_list[neighbor]:
                    continue

            # add neighbor to open list and update support_list
            open_list.put(neighbor_entry)
            support_list[neighbor] = neighbor_g

    print("Target node not found!")
    return None