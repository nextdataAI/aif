from .Algorithm import Algorithm
from .utils import get_valid_moves
__all__ = ['AlphaBeta']





class AlphaBeta(Algorithm):
    def __init__(self, env_name):
        super().__init__(env_name=env_name, name='AlphaBeta')

    def __call__(self, seed):
        self.start_timer()
        local_env, local_state, local_game_map, start, target = super().initialize_env(seed)

        done = False
        while not done:
            valid_moves = get_valid_moves(local_game_map, local_state)
            # Get the best move
            best_move = self.get_best_move(local_env, local_state, valid_moves)
            # Make the move
            local_state, reward, done, info = local_env.step(best_move)
            # Update the game map
            local_game_map[best_move] = 1
            # Render the environment
            local_env.render()