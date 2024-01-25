import copy
import os
import pickle

import numpy as np
from tqdm import tqdm

from ..AnimateGif import Animator
from .Algorithm import Algorithm
from ..utils import get_player_location

__all__ = ["QTable"]


class QTable(Algorithm):

    def __init__(self, env_name, name='QTable', learning_rate=0.95, discount_factor=0.95, epsilon=0.1, render=False, animate=False):
        super().__init__(
            name=name,
            env_name=env_name,
            animate=animate
        )
        self.epsilon = epsilon
        self.render = render
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def update(self, s_t, a_t, r_t, s_t1):
        self.q_table[s_t][a_t] += self.learning_rate * (
                r_t + self.discount_factor * np.max(self.q_table[s_t1]) - self.q_table[s_t][a_t])

    def get_action(self, s_t):
        return np.argmax(self.q_table[s_t]) if np.random.random() > self.epsilon else np.random.randint(0, 4)

    def __call__(self, seed):
        self.start_timer()
        local_env, local_state, local_game_map, start, target = super().initialize_env(seed)
        original_state = copy.deepcopy(local_state)
        if os.path.exists(f'{os.path.dirname(__file__)}/models/QTable.npy'):
            with open(f'{os.path.dirname(__file__)}/models/QTable.npy', 'rb') as f:
                self.q_table = pickle.load(f)
        else:
            self.q_table = np.zeros(shape=(local_game_map.shape[0], local_game_map.shape[1], 4))
        s_t = start
        a_t = self.get_action(s_t)
        done = False
        path = [s_t]
        r_t = 0
        with tqdm(total=1000, desc='Training', unit='steps') as pbar:
            while not done:
                pbar.update(1)
                s_t1, r_t, done, info = local_env.step(a_t)
                s_t1 = get_player_location(s_t1['chars'])
                if s_t1 == (None, None):
                    break
                path.append(s_t1)
                if s_t1 == target:
                    break
                if r_t == 0.0 and not done:
                    r_t = -0.01
                if r_t < 0 and not done:
                    r_t = -1
                a_t1 = self.get_action(s_t1)
                self.update(s_t, a_t, r_t, s_t1)
                s_t = s_t1
                a_t = a_t1

        if self.animate:
            Animator(size=self.size, game_map=original_state, path=path, visited=path, file_path=f'{self.name}.gif', fps=120)()
        with open(f'{os.path.dirname(__file__)}/models/QTable.npy', 'wb') as f:
            pickle.dump(self.q_table, f)
        return True if r_t == 1.0 else False, list(path), list(path), self.stop_timer()
