import copy
import random

import numpy as np

from .Algorithm import Algorithm
from .utils import get_valid_moves, get_player_location

__all__ = ['AlphaBeta']


class Node:
    def __init__(self, state, player, action=None, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.score = 0
        self.action = action
        self.coordinates = player
        self.valid_moves = get_valid_moves(state, player)
        self.is_leaf = False

    def add_child(self, child):
        self.children.append(child)

    def get_children(self):
        return self.children

    def get_parent(self):
        return self.parent

    def get_state(self):
        return self.state

    def get_score(self):
        return self.score

    def set_score(self, score):
        self.score = score

    def get_valid_moves(self):
        return self.valid_moves

    def set_valid_moves(self, valid_moves):
        self.valid_moves = valid_moves

    def is_leaf(self):
        return self.is_leaf

    def set_is_leaf(self, is_leaf):
        self.is_leaf = is_leaf

    def search(self, depth, alpha, beta, maximizing_player):
        if depth == 0 or self.is_leaf:
            return {'value': self.score, 'action': self.action}
        if maximizing_player:
            value = float('-inf')
            best = None
            for child in self.children:
                value = max(value, child.search(depth - 1, alpha, beta, False)['value'])
                alpha = max(alpha, value)
                if alpha >= beta:
                    best = child
                    break
            return {'value': value, 'action': [best.action if best is not None else None]}
        else:
            best = None
            value = float('inf')
            for child in self.children:
                value = min(value, child.search(depth - 1, alpha, beta, True)['value'])
                beta = min(beta, value)
                if alpha >= beta:
                    best = child
                    break
            return {'value': value, 'action': [best.action if best is not None else None]}


class AlphaBeta(Algorithm):
    def __init__(self, env_name, depth=5):
        super().__init__(env_name=env_name, name='AlphaBeta')
        self.memory = []
        self.depth = depth

    def create_tree(self, player, target, local_game_map, parent=None):
        if player in self.memory:
            return
        self.memory.append(player)
        for action, coordinates in get_valid_moves(local_game_map, player, 'both'):
            local_game_map[coordinates] = 64
            local_game_map[player] = 46
            player = coordinates
            child = Node(local_game_map, player, action, parent)
            child.set_score(parent.get_score() - 1)
            parent.add_child(child)
            self.create_tree(player, target, local_game_map, child)

    def __call__(self, seed):
        self.start_timer()
        local_env, local_state, local_game_map, start, target = super().initialize_env(seed)
        done = False
        action = None
        self.tree = None
        path = []
        reward = float('-inf')
        while not done:
            # clear_screen()
            agent_location = get_player_location(local_game_map)
            self.tree = Node(copy.deepcopy(local_game_map), start, action, self.tree)
            self.create_tree(agent_location, target, copy.deepcopy(local_game_map), self.tree)
            res = self.tree.search(self.depth, float('-inf'), np.inf, maximizing_player=True)
            if res['action'][0] is None:
                action = random.randint(0, 3)
            else:
                action = res['action'][0]
            path.append(agent_location)
            local_game_map, reward, done, _ = local_env.step(action)
            local_game_map = local_state.get('chars')
            # local_env.render()
        return True if reward == 1.0 else False, path, path, self.stop_timer()
