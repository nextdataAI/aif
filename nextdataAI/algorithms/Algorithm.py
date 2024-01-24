import time

import gym
import minihack
from ..utils import get_player_location, get_target_location

__all__ = ['Algorithm']


class Algorithm:
    def __init__(self, env_name: str, name: str = "Algorithm", animate=False):
        self.stop = None
        self.start = None
        self.animate = animate
        self.env_name = env_name
        if '45x19' in env_name:
            self.env = gym.make(env_name, observation_keys=("chars", "pixel"),
                                max_episode_steps=45 * 19)
        elif '15x15' in env_name:
            self.env = gym.make(env_name, observation_keys=("chars", "pixel"),
                                max_episode_steps=15 * 15)
        elif '9x9' in env_name:
            self.env = gym.make(env_name, observation_keys=("chars", "pixel"),
                                max_episode_steps=9 * 9)
        self.name = name

    def initialize_env(self, seed: int, informed: bool = True):
        self.env.seed(seed)
        local_state = self.env.reset()
        local_game_map = local_state.get('chars')
        start = get_player_location(local_game_map)
        # plt.imshow(local_state.get('pixel'))
        # plt.show()
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