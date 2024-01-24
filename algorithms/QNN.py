from .Algorithm import Algorithm
from aif.qlearning.Agent import LSTMAgent, train, NNAgent
from aif.qlearning.ExperienceReplay import PrioritizedExperienceReplay

__all__ = ['QNN']


class QNN(Algorithm):
    def __init__(self, env_name: str = "MiniHack-MazeWalk-15x15-v0", name: str = "QLSTM"):
        super().__init__(env_name, name)
        self.batch_size = 400
        self.memory = PrioritizedExperienceReplay(memory_capacity=10000)
        self.agent = None

    def __call__(self, seed: int):
        self.start_timer()
        local_env, _, local_game_map, start, target = super().initialize_env(seed)
        input_dim = local_game_map.shape[0] * local_game_map.shape[1]
        self.agent = NNAgent(memory=self.memory, state_dim=input_dim, action_dim=4, batch_size=self.batch_size,
                             gamma=0.9, agent=self.agent)
        target_reached, explored_positions = train(self.agent, local_env, local_game_map, start, target,
                                                   self.batch_size)
        self.agent.plot_learning_curve()
        print(target_reached)
        if not target_reached:
            return False, None, explored_positions, self.stop_timer()
        return True, explored_positions, explored_positions, self.stop_timer()
