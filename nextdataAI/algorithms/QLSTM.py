from .Algorithm import Algorithm
from ..qlearning.Agent import LSTMAgent, train
from ..qlearning.ExperienceReplay import PrioritizedExperienceReplay

__all__ = ['QLSTM']


class QLSTM(Algorithm):
    def __init__(self, env_name: str = "MiniHack-MazeWalk-15x15-v0", name: str = "QLSTM", animate: bool = False):
        super().__init__(env_name, name, animate)
        if '45x19' in env_name:
            self.batch_size = 300
            self.past_states_seq_len = 200
            self.max_epoch = 10
            self.memory = PrioritizedExperienceReplay(memory_capacity=45 * 19 * 2 * self.max_epoch)
            self.epsilon_decay = 140
            self.learning_rate = 1e-4
        if '15x15' in env_name:
            self.batch_size = 300
            self.past_states_seq_len = 100
            self.max_epoch = 10
            self.memory = PrioritizedExperienceReplay(memory_capacity=15 * 15 * 2 * self.max_epoch)
            self.epsilon_decay = 35
            self.learning_rate = 1e-3
        if '9x9' in env_name:
            self.batch_size = 40
            self.past_states_seq_len = 20
            self.max_epoch = 10
            self.memory = PrioritizedExperienceReplay(memory_capacity=9 * 9 * 2 * self.max_epoch)
            self.epsilon_decay = 15
            self.learning_rate = 1e-5

        self.env_name = env_name
        self.agent = None

    def __call__(self, seed: int):
        self.start_timer()
        local_env, _, local_game_map, start, target = super().initialize_env(seed)
        input_dim = local_game_map.shape[0] * local_game_map.shape[1]
        self.agent = LSTMAgent(self.memory, input_dim, 4, self.epsilon_decay, self.env_name, batch_size=self.batch_size,
                               learning_rate=self.learning_rate, gamma=0.9, agent=self.agent)
        target_reached, explored_positions = train(self.agent, local_env, local_game_map, start, target,
                                                   self.batch_size, self.past_states_seq_len, max_epoch=self.max_epoch)
        self.agent.plot_learning_curve()
        print(target_reached)
        if not target_reached:
            return False, None, explored_positions, self.stop_timer()
        return True, explored_positions, explored_positions, self.stop_timer()
