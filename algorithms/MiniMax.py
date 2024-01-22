from .Algorithm import Algorithm
from .utils import get_valid_moves, get_target_location


class MiniMax(Algorithm):
    def __init__(self, env_name: str):
        super().__init__(env_name, name='MiniMax')
        self.depth = 2
        self.visited = []

    def __call__(self, seed: int):
        self.start_timer()
        local_env, local_state, local_game_map, start, target = super().initialize_env(seed)
        self.local_game_map = local_game_map
        path = []
        action = self.Minimax_Decision(start)
        while not local_state == target:
            local_state = self.Result(action, local_state)
            action = self.Minimax_Decision(local_state, self.depth)
            path.append(local_state)
        return local_state.get('pixel'), True, path, self.visited, self.stop_timer()

    def Minimax_Decision(self, state, depth=2):
        local_max = -float('inf')
        action = None
        for a in get_valid_moves(self.local_game_map, state, mode='action'):
            v = self.MinValue(state, self.Result(a, state), depth)
            if v > local_max:
                local_max = v
                action = a
        return action

    def MaxValue(self, old_state, state, depth=5):
        self.set_player_position(old_state, state)
        if state == get_target_location(self.local_game_map):
            return 100
        v = -float('inf')
        for a, s in get_valid_moves(self.local_game_map, state, mode='both'):
            self.set_player_position(state, s)
            if depth == 0:
                return v
            v = max(v, self.MinValue(s, self.Result(a, s)), depth - 1)
        return v

    def MinValue(self, old_state, state, depth=5):
        self.set_player_position(old_state, state)
        if state == get_target_location(self.local_game_map):
            return 100
        v = float('inf')
        for a, s in get_valid_moves(self.local_game_map, state, mode='both'):
            self.set_player_position(state, s)
            if depth == 0:
                return v
            v = min(v, self.MaxValue(s, self.Result(a, s)), depth - 1)
        return v

    def Result(self, action, state):
        # Go up 0, down 2, left 3, right 1
        if action == 0:
            return state[0], state[1] - 1
        elif action == 1:
            return state[0] + 1, state[1]
        elif action == 2:
            return state[0], state[1] + 1
        elif action == 3:
            return state[0] - 1, state[1]
        else:
            return state

    def set_player_position(self, old_state, state):
        self.local_game_map[old_state] = ord('.')
        self.local_game_map[state] = ord('@')
