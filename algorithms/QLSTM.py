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
        self.hidden_dim = 32
        self.n_layers = 2
        self.memory = ExperienceReplay()
        self.gamma = 0.99
        self.batch_size = batch_size
        # LSTM layer for handling sequential data
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, self.n_layers, batch_first=True)
        # Fully connected layer for generating Q-values from LSTM's hidden states
        self.fc = nn.Linear(self.hidden_dim, self.action_dim)

        # Forward pass through the model

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

        # Pass through LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # Pass the LSTM's output through the fully connected layer
        out = self.fc(out[:, -1, :])
        return out

    # Choose a discrete action given the state
    def act(self, state):
        """
        Chooses an action based on given state.

        :param state: The current state of the environment
        :return: The action selected by the agent
        """

        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)  # Convert to tensor
            state.to(device)
            q_values = self.forward(state)  # Get Q-values
            action = np.argmax(q_values.cpu().data.numpy())  # Choose action with highest Q-value
            return action

    # Cache a new experience
    def remember(self, state, action, reward, next_state, done):
        """
        Stores experience in memory.

        :param state: The current state of the environment
        :param action: The action taken by the agent
        :param reward: The reward received after taking the action
        :param next_state: The state of the environment after the action was taken
        :param done: Whether the environment is done (episode is finished)
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

        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        done = torch.tensor(done, dtype=torch.int8)

        current_q_values = self.forward(state)
        max_next_q_values = self.forward(next_state).detach().max(1)[0]
        expected_q_values = reward + self.gamma * max_next_q_values * (1 - done)

        loss = F.smooth_l1_loss(current_q_values.gather(1, action.unsqueeze(-1)), expected_q_values.unsqueeze(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


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

    # Initialize the done flag to False
    done = False
    # Initialize total reward to 0
    total_reward = 0

    # Run until an episode is done
    while not done:
        # Ask the agent to decide on an action based on the state sequence
        action = agent.act(state)

        # Take the action and get the new state, reward and done flag
        next_state, reward, done, _ = env.step(action)
        next_state = next_state['chars'].flatten()

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


class QLSTM(Algorithm):
    def __init__(self, env_name: str = "MiniHack-MazeWalk-15x15-v0", name: str = "QLSTM"):
        super().__init__(env_name, name)
        self.batch_size = 32
        self.past_states_seq_len = 8

    def __call__(self, seed: int):
        local_env, _, local_game_map, _, _ = super().initialize_env(seed)
        input_dim = local_game_map.shape[0] * local_game_map.shape[1]
        train(DQNAgent(input_dim, self.batch_size), local_env, self.batch_size, self.past_states_seq_len)
