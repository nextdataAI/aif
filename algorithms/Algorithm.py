import gym
import minihack
from .utils import get_player_location, get_target_location
import matplotlib.pyplot as plt

__all__ = ['Algorithm']


class Algorithm:
    def __init__(self, env_name: str, name: str = "Algorithm"):
        self.env = gym.make(env_name, observation_keys=("chars", "pixel"),)
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
