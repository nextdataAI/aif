import time

import gym
import minihack
from ..utils import get_player_location, get_target_location

__all__ = ['Algorithm']


class Algorithm:
    def __init__(self, env_name: str, name: str = "Algorithm", animate=False, override_maze=None):
        self.stop = None
        self.start = None
        self.animate = animate
        self.env_name = env_name
        self.size = None
        self.ovveride_maze = override_maze
        self.set_size()
        self.env = gym.make(env_name, observation_keys=("chars", "pixel"), 
                            max_episode_steps=self.get_size())
        self.name = name
  
    def get_size(self):
        if self.size == 'large':
            return 45*19*2
        elif self.size == 'medium':
            return 15*15*2
        elif self.size == 'small':
            return 9*9*2
        return 1000
  
    def set_size(self):
        if self.env_name == 'MiniHack-MazeWalk-Mapped-45x19-v0':
            self.size = 'large'
        elif self.env_name == 'MiniHack-MazeWalk-Mapped-15x15-v0':
            self.size = 'medium'
        elif self.env_name == 'MiniHack-MazeWalk-Mapped-9x9-v0':
            self.size = 'small'

    def initialize_env(self, seed: int, informed: bool = True):
        self.env.seed(seed)
        local_state = self.env.reset()
        local_game_map = local_state.get('chars')
        if self.ovveride_maze is not None:
            local_game_map = self.ovveride_maze
            self.env_name = 'Custom-MazeWalk'
        start = get_player_location(local_game_map)
        target = get_target_location(local_game_map) if informed else None
        return self.env, local_state, local_game_map, start, target

    def start_timer(self):
        self.start = time.time()

    def stop_timer(self):
        self.stop = time.time()
        return self.stop - self.start

    @staticmethod
    def build_path(parent, target):
        path = []
        while target is not None:
            path.append(target)
            target = parent[target]
        path.reverse()
        return path