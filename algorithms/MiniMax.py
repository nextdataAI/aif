from .Algorithm import Algorithm
from .utils import get_valid_moves, get_target_location


class MiniMax(Algorithm):
    def __init__(self, env_name: str):
        super().__init__(env_name, name='MiniMax')
        self.depth = 5
        self.visited = []

    def __call__(self, seed: int):
        self.start_timer()
        local_env, local_state, local_game_map, start, target = super().initialize_env(seed)
        self.local_game_map = local_game_map
        path = []
        action = self.Minimax_Decision(start)
        while not self.Terminal_Test(local_state):
            local_state = self.Result(action, local_state)
            action = self.Minimax_Decision(local_state)
            path.append(local_state)
        return local_state.get('pixel'), True, path, self.visited, self.stop_timer()

    def Minimax_Decision(self, state):
        local_max = -float('inf')
        action = None
        for a in self.Actions(state):
            v = self.MinValue(self.Result(a, state))
            if v > local_max:
                local_max = v
                action = a
        return action

    def MaxValue(self, state):
        if self.Terminal_Test(state):
            return self.Utility(state)
        v = -float('inf')
        for a, s in self.Successors(state):
            v = max(v, self.MinValue(self.Result(a, s)))
        return v

    def MinValue(self, state):
        if self.Terminal_Test(state):
            return self.Utility(state)
        v = float('inf')
        for a, s in self.Successors(state):
            v = min(v, self.MaxValue(self.Result(a, s)))
        return v

    def Actions(self, state):
        # Returns all valid moves from the current state
        return get_valid_moves(self.local_game_map, state, mode='action')

    def Result(self, action, state):
        # Go up 0, down 2, left 1, right 3
        if action == 0:
            return state[0], state[1] - 1
        elif action == 1:
            return state[0] - 1, state[1]
        elif action == 2:
            return state[0], state[1] + 1
        elif action == 3:
            return state[0] + 1, state[1]
        else:
            return state

    def Terminal_Test(self, state):
        return state == get_target_location(self.local_game_map)

    def Utility(self, state):
        if state == get_target_location(self.local_game_map):
            return 100
        else:
            return -1

    def Successors(self, state):
        valid_moves = get_valid_moves(self.local_game_map, state, mode='both')
        return valid_moves
