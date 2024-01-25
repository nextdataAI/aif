from queue import PriorityQueue
from typing import Optional, Any

from .Algorithm import Algorithm
from ..utils import get_valid_moves

from ..AnimateGif import Animator
__all__ = ['Dijkstra']


class Dijkstra(Algorithm):
    def __init__(self, env_name: str = "MiniHack-MazeWalk-15x15-v0", name: str = 'Dijkstra', animate: bool = False):
        super().__init__(env_name, name, animate)

    def __call__(self, seed: int) -> Optional[Any]:
        self.start_timer()
        local_env, local_state, local_game_map, start, target = super().initialize_env(seed)

        # initialize open and close list
        open_list = PriorityQueue()
        close_list = []
        # additional dict which maintains the nodes in the open list for an easier access and check
        support_list = {}

        starting_state_g = 0
        starting_state_f = starting_state_g

        open_list.put((starting_state_f, (start, starting_state_g)))
        support_list[start] = starting_state_g
        parent = {start: None}

        while not open_list.empty():
            # get the node with lowest f
            _, (current, current_cost) = open_list.get()
            # add the node to the close list
            close_list.append(current)

            if current == target:
                path = self.build_path(parent, target)
                time = self.stop_timer()
                if self.animate:
                    Animator(size=self.size, game_map=local_state, path=path, visited=close_list, file_path=f'{self.name}.gif')()
                return True, list(path), list(close_list), time

            for neighbor in get_valid_moves(local_game_map, current):
                # check if neighbor in close list, if so continue
                if neighbor in close_list:
                    continue

                # compute neighbor g, h and f values
                neighbor_g = 1 + current_cost
                neighbor_f = neighbor_g
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

        return False, None, list(close_list), self.stop_timer()

    @staticmethod
    def build_path(parent, target):
        path = []
        while target is not None:
            path.append(target)
            target = parent[target]
        path.reverse()
        return path
