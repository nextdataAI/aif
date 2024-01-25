from .Algorithm import Algorithm
from ..qlearning.Agent import train, NNAgent
from ..qlearning.ExperienceReplay import PrioritizedExperienceReplay

__all__ = ['QNN']


class QNN(Algorithm):
    def __init__(self, env_name: str = "MiniHack-MazeWalk-15x15-v0", name: str = "QLSTM", animate: bool = False):
        super().__init__(env_name, name, animate)
        if '45x19' in env_name:
            self.batch_size = 300
            self.max_epoch = 10
            self.memory = PrioritizedExperienceReplay(memory_capacity=45 * 19 * 2 * self.max_epoch)
            self.epsilon_decay = 140
            self.learning_rate = 1e-4
        if '15x15' in env_name:
            self.batch_size = 400
            self.max_epoch = 10
            self.memory = PrioritizedExperienceReplay(memory_capacity=15 * 15 * 2 * self.max_epoch)
            self.epsilon_decay = 35
            self.learning_rate = 1e-4
        if '9x9' in env_name:
            self.batch_size = 30
            self.max_epoch = 10
            self.memory = PrioritizedExperienceReplay(memory_capacity=9 * 9 * 2 * self.max_epoch)
            self.epsilon_decay = 15
            self.learning_rate = 1e-4

        self.agent = None

    def __call__(self, seed: int):
        self.start_timer()
        local_env, _, local_game_map, start, target = super().initialize_env(seed)
        input_dim = local_game_map.shape[0] * local_game_map.shape[1]
        self.agent = NNAgent(self.memory, input_dim, 4, self.epsilon_decay,
                             batch_size=self.batch_size, learning_rate=self.learning_rate, gamma=0.9, agent=self.agent)
        target_reached, explored_positions = train(self.agent, local_env, local_game_map, start, target,
                                                   self.batch_size, max_epoch=self.max_epoch)
        self.agent.plot_learning_curve()
        print(target_reached)
        if not target_reached:
            return False, None, explored_positions, self.stop_timer()
        return True, explored_positions, explored_positions, self.stop_timer()
