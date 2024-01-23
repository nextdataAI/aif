import numpy as np

from .Algorithm import Algorithm
from ..utils import get_player_location, clear_screen

__all__ = ["Qlearning"]


class Qlearning(Algorithm):

    def __init__(self, env_name, learning_rate=0.95, discount_factor=0.95, epsilon=0.1, render=False):
        super().__init__(
            name="Qlearning",
            env_name=env_name,
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
        self.q_table = np.zeros(shape=(local_game_map.shape[0], local_game_map.shape[1], 4))
        s_t = start
        a_t = self.get_action(s_t)
        done = False
        path = []
        for i in range(100):
            while not done:
                if self.render:
                    clear_screen()
                    local_env.render()
                s_t1, r_t, done, info = local_env.step(a_t)
                s_t1 = get_player_location(s_t1['chars'])
                if s_t1 == (None, None):
                    break
                path.append(s_t1)
                if s_t1 == target:
                    break
                a_t1 = self.get_action(s_t1)
                self.update(s_t, a_t, r_t, s_t1)
                s_t = s_t1
                a_t = a_t1

        return list(path), list(path), self.stop_timer()
