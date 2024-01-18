import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from gym.core import Env
from .Algorithm import Algorithm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Experience Replay class to efficiently store and sample experiences
class ExperienceReplay:
    def __init__(self):
        self.memory_capacity = 1000
        self.buffer = deque(maxlen=self.memory_capacity)

    def push(self, state_sequence, action, reward, next_state, done):
        self.buffer.append((state_sequence, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent(nn.Module):
    def __init__(self, input_dim: int, batch_size: int = 1):
        """
        Construct a Deep Q-Learning Agent.

        :param input_dim: The number of states in the environment
        :param batch_size: The size of the batch to be used during training
        """
        super(DQNAgent, self).__init__()
        self.action_dim = 4
        self.hidden_dim = 128
        self.n_layers = 1
        self.memory = ExperienceReplay()
        self.gamma = 0.99
        self.batch_size = batch_size
        self.device = torch.device(device)

        # LSTM layer for handling sequential states
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.n_layers,
                            batch_first=True).to(self.device)

        # Fully connected layer for generating Q-values from LSTM's hidden states
        self.fc = nn.Linear(in_features=self.hidden_dim,
                            out_features=self.action_dim).to(self.device)

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

        # Initial hidden and cell states for LSTM
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)

        out, (_, _) = self.lstm(x, (h0, c0))
        # Pass the LSTM's output through the fully connected layer
        out = self.fc(out[:, -1, :])
        out = torch.relu(out)
        return out

    # Choose a discrete action given the state
    def act(self, state, epsilon: float = 0.5):
        """
        Chooses an action based on given state.

        :param state: The current state of the environment
        :param epsilon: The exploration rate.
        It's the probability that a random action will be selected.
        :return: The action selected by the agent
        """

        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).to(self.device)

            # Explore: select a random action.
            if np.random.uniform() < epsilon:
                action = random.randint(0, self.action_dim - 1)

            # Exploit: select the action with max value (future reward)
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


def epsilon_update(epsilon_start: float, epsilon_final: float, epsilon_decay: int, frame_idx: int):
    return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


def train(agent: DQNAgent, env: Env, batch_size: int = 1, sequence_length: int = 4):
    # Set up an Adam optimizer for the training
    optimizer = torch.optim.Adam(agent.parameters())

    # Reset the environment and get the initial state
    single_state = env.reset()
    single_state = single_state['chars'].flatten()
    # Initialize a sequence of states based on sequence_length
    state = np.zeros((sequence_length,) + single_state.shape)
    # Set the last state in sequence to the initial state
    state[-1] = single_state

    done = False
    total_reward = 0
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500
    frame_idx = 0
    explored_positions = []

    # Run until an episode is done
    while not done:
        epsilon = epsilon_update(epsilon_start, epsilon_final, epsilon_decay, frame_idx)

        # Ask the agent to decide on an action based on the state sequence
        action = agent.act(state, epsilon)
        # Take the action and get the new state, reward and done flag
        next_state, reward, done, _ = env.step(action)
        agent_position = get_position(next_state['chars'])
        explored_positions.append(agent_position)
        next_state = next_state['chars'].flatten()
        frame_idx += 1

        # env.render()

        # Shift the states in the sequence and append the new state
        state[:-1] = state[1:]
        state[-1] = next_state

        # Store this transition in the agent's memory for future training
        agent.remember(state, action, reward, next_state, done)

        # Perform experience replay and update the weights of the DQN
        agent.experience_replay(optimizer, batch_size)

        # Accumulate the reward
        total_reward += reward

    # Print the total reward for this episode
    print(f"Total Reward: {total_reward}")
    return explored_positions


def get_position(chars_matrix, agent_char='@'):
    # '@' is an example, the agent's symbol can be any character based on your environment settings
    # chars_matrix is the state['chars'] of your environment

    for i in range(len(chars_matrix)):
        for j in range(len(chars_matrix[i])):
            if chars_matrix[i][j] == agent_char:
                return i, j  # return as soon as you find the agent char
    return None  # return None if the agent char is not found


class QLSTM(Algorithm):
    def __init__(self, env_name: str = "MiniHack-MazeWalk-15x15-v0", name: str = "QLSTM"):
        super().__init__(env_name, name)
        self.batch_size = 8
        self.past_states_seq_len = 8

    def __call__(self, seed: int):
        self.start_timer()
        local_env, _, local_game_map, _, _ = super().initialize_env(seed)
        input_dim = local_game_map.shape[0] * local_game_map.shape[1]
        explored_positions = train(DQNAgent(input_dim, self.batch_size), local_env, self.batch_size, self.past_states_seq_len)
        return None, explored_positions, self.stop_timer()
