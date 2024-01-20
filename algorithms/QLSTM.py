import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from .utils import get_player_location, get_valid_moves, clear_screen, animate
from .Algorithm import Algorithm
import matplotlib.pyplot as plt
from gym import Env

__all__ = ['QLSTM']

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


# Experience Replay class to efficiently store and sample experiences
class ExperienceReplay:
    def __init__(self):
        self.memory_capacity = 5 * 1000
        self.buffer = deque(maxlen=self.memory_capacity)

    def push(self, state_sequence, action, reward, next_state, done):
        self.buffer.append((state_sequence, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def path(self):
        return self.buffer

    def __len__(self):
        return len(self.buffer)


def weights_init(m):
    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    m.bias.data.fill_(0)


class DQNAgent(nn.Module):

    def __init__(self, input_dim: int, batch_size: int = 1, agent=None):
        """
        Construct a Deep Q-Learning Agent.

        :param input_dim: The number of states in the environment
        :param batch_size: The size of the batch to be used during training
        """
        super(DQNAgent, self).__init__()
        self.action_dim = 4
        self.hidden_dim = 128
        self.n_layers = 1
        self.gamma = 0.99
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.epsilon_decay = 3000
        self.epsilon_end = 1e-3
        self.epsilon_start = 1.0

        if agent is None:
            self.epsilon = 1.0
            self.frame_idx = 0
            self.memory = ExperienceReplay()

            # LSTM layer for handling sequential states
            self.lstm = nn.LSTM(input_size=input_dim,
                                hidden_size=self.hidden_dim,
                                num_layers=self.n_layers,
                                batch_first=True).to(self.device)
            self.dropout_lstm = nn.Dropout(p=0.2)

            # Fully connected layers for generating Q-values from LSTM's hidden states
            self.fc1 = nn.Linear(in_features=self.hidden_dim,
                                 out_features=64).to(self.device)
            self.fc1.apply(weights_init)
            self.activation1 = nn.ELU().to(self.device)

            self.fc2 = nn.Linear(in_features=64,
                                 out_features=64).to(self.device)
            self.fc2.apply(weights_init)
            self.activation2 = nn.ELU().to(self.device)

            self.fc3 = nn.Linear(in_features=64,
                                 out_features=64).to(self.device)
            self.fc3.apply(weights_init)
            self.activation3 = nn.ELU().to(self.device)

            self.fc_final = nn.Linear(in_features=64,
                                      out_features=self.action_dim).to(self.device)
            self.fc_final.apply(weights_init)
        else:
            self.epsilon = agent.epsilon
            self.frame_idx = agent.frame_idx
            self.memory = agent.memory

            self.lstm = agent.lstm
            self.dropout_lstm = agent.dropout_lstm
            self.activation1 = agent.activation1
            self.fc1 = agent.fc1
            self.activation2 = agent.activation2
            self.fc2 = agent.fc2
            self.activation3 = agent.activation3
            self.fc3 = agent.fc3
            self.fc_final = agent.fc_final

    def update_epsilon(self):
        self.epsilon = (self.epsilon_end + (self.epsilon_start - self.epsilon_end)
                        * math.exp(-1. * self.frame_idx / self.epsilon_decay))

    def forward(self, x):
        """
        Forward pass through the network.

        :param x: The input tensor/data
        :return: The output tensor/data
        """

        if len(x.shape) == 2 and x.shape[0] == self.batch_size:
            x = x.unsqueeze(1)
        elif len(x.shape) == 2:
            x = x.unsqueeze(0)

        out, (_, _) = self.lstm(x)
        out = self.dropout_lstm(out)
        out = self.fc1(out[:, -1, :])
        out = self.activation1(out)
        out = self.fc2(out)
        out = self.activation2(out)
        out = self.fc3(out)
        out = self.activation3(out)
        out = self.fc_final(out)

        return out

    # Choose a discrete action given the state
    def act(self, state):
        """
        Chooses an action based on given state.

        :param state: The current state of the environment
        :return: The action selected by the agent
        """

        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).to(self.device)

            # Explore: select a random action.
            if np.random.uniform() < self.epsilon:
                action = random.randint(0, self.action_dim - 1)

            # Exploit: select the action with max q value (future reward)
            else:
                q_values = self.forward(state).mean(0)
                action = torch.argmax(q_values).item()

            return action

    # Cache a new experience
    def remember(self, state, action, reward, next_state, done):
        """
        Stores experience in memory.

        :param state: The current state of the environment
        :param action: The action taken by the agent
        :param reward: The reward received after taking the action
        :param next_state: The state of the environment after the action was taken
        :param done: Whether the environment is done (an episode is finished)
        """

        self.memory.push(state, action, reward, next_state, done)

    # Train by sampling from cached experiences
    def experience_replay(self, optimizer, batch_size):
        """
        Trains the network using the batch from experiences in memory.

        :param optimizer: The optimizer to use for updating the weights of the network
        :param batch_size: The number of experiences to use for each update
        """

        if len(self.memory) < batch_size:
            return
        experiences = self.memory.sample(batch_size)
        state, action, reward, next_state, done = zip(*experiences)

        state = torch.from_numpy(np.array(state)).float().to(self.device)
        next_state = torch.from_numpy(np.array(next_state)).float().to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.int64).to(self.device)
        done = torch.tensor(done, dtype=torch.int8).to(self.device)

        current_q_values = self.forward(state)
        max_next_q_values = self.forward(next_state).detach().max(1)[0]
        expected_q_values = reward + self.gamma * max_next_q_values * (1 - done)

        loss = F.mse_loss(current_q_values.gather(1, action.unsqueeze(-1)), expected_q_values.unsqueeze(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train(agent: DQNAgent, env: Env, single_state: np.ndarray, start: (int, int), target: (int, int),
          batch_size: int = 1, sequence_length: int = 4):

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3, amsgrad=True)

    single_state = single_state.flatten()
    # Initialize a sequence of states based on sequence_length
    state = np.zeros((sequence_length,) + single_state.shape)
    # Set the last state in sequence to the initial state
    state[-1] = single_state

    done = False
    target_reached = False
    total_reward = 0
    frame_idx = 0
    explored_positions = [start]

    # Run until an episode is done
    while not done:
        agent.update_epsilon()

        # Ask the agent to decide on an action based on the state sequence
        action = agent.act(state)

        # Take the action and get the new state, reward and done flag
        next_state, reward, done, _ = env.step(action)
        agent_position = get_player_location(next_state['chars'])
        if agent_position == (None, None):
            agent_position = target
        if agent_position not in explored_positions:
            explored_positions.append(agent_position)
        next_state = next_state['chars'].flatten()

        if reward == 1:
            target_reached = True

        # env.render()

        # Shift the states in the sequence and append the new state
        state[:-1] = state[1:]
        state[-1] = next_state

        # Store this transition in the agent's memory for future training
        agent.remember(state, action, reward, next_state, done)

        # Perform experience replay and update the weights of the DQN
        agent.experience_replay(optimizer, batch_size)

        agent.frame_idx += 1
        frame_idx += 1
        total_reward += reward
        total_reward = round(total_reward, 2)
        print(f'\rFrame: {frame_idx} | Reward: {total_reward} ', end='', flush=True)

    # Print the total reward for this episode
    return target_reached, explored_positions


class QLSTM(Algorithm):
    def __init__(self, env_name: str = "MiniHack-MazeWalk-15x15-v0", name: str = "QLSTM"):
        super().__init__(env_name, name)
        self.batch_size = 100
        self.past_states_seq_len = 200
        self.agent = None

    def __call__(self, seed: int):
        self.start_timer()
        local_env, _, local_game_map, start, target = super().initialize_env(seed)
        input_dim = local_game_map.shape[0] * local_game_map.shape[1]
        self.agent = DQNAgent(input_dim, self.batch_size, self.agent)
        target_reached, explored_positions = train(self.agent, local_env, local_game_map, start, target, self.batch_size,
                                                   self.past_states_seq_len)
        print(target_reached)
        if not target_reached:
            return None, explored_positions, self.stop_timer()
        return explored_positions, explored_positions, self.stop_timer()
