import math
import os
import random
import string
from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym import Env

from .ExperienceReplay import ExperienceReplay
from ..utils import get_player_location

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

__all__ = ['Agent', 'LSTMAgent', 'train', 'NNAgent']


def init_lstm(m):
    for name, param in m.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            param.data.fill_(0)
        # Forget Gate defaulting to 1
        elif 'bias_ih' in name:
            param.data.fill_(0)
            n = param.size(0)
            start, end = n // 4, n // 2
            param.data[start:end].fill_(1.)


def init_kaiming_normal(m):
    nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
    m.bias.data.fill_(0)


def init_kaiming_uniform(m):
    nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
    m.bias.data.fill_(0)


class Agent(nn.Module):

    @abstractmethod
    def __init__(self, memory: ExperienceReplay, state_dim: int, action_dim: int, epsilon_decay: int,
                 batch_size: int = 1, gamma: float = 0.99, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.batch_size = batch_size

        # with this configuration, it reaches epsilon = 0.01 around frame = 700 for start = 1
        self.epsilon = 1.0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = epsilon_decay

        self.frame_idx = 0
        self.gamma = gamma
        self.memory = memory
        self.loss_values = []

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def remember(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def experience_replay(self, batch_size: int):
        pass


class NNAgent(Agent):
    def __init__(self, memory: ExperienceReplay, state_dim: int, action_dim: int, epsilon_decay: int,
                 batch_size: int = 1, learning_rate: float = 1e-3, gamma: float = 0.99, agent: Agent = None):
        super().__init__(memory, state_dim, action_dim, epsilon_decay, batch_size=batch_size, gamma=gamma)

        if agent is None:
            self.fc1 = nn.Linear(self.state_dim, 128).to(self.device)
            self.fc1.apply(init_kaiming_normal)
            self.activation1 = nn.SELU().to(self.device)

            self.fc2 = nn.Linear(128, 64).to(self.device)
            self.fc2.apply(init_kaiming_normal)
            self.activation2 = nn.SELU().to(self.device)

            self.fc3 = nn.Linear(64, 64).to(self.device)
            self.fc3.apply(init_kaiming_normal)
            self.activation3 = nn.SELU().to(self.device)

            self.fc4 = nn.Linear(64, 64).to(self.device)
            self.fc4.apply(init_kaiming_normal)
            self.activation4 = nn.SELU().to(self.device)

            self.fc_final = nn.Linear(64, self.action_dim).to(self.device)

            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, amsgrad=True)
        else:
            self.fc1 = agent.fc1
            self.activation1 = agent.activation1

            self.fc2 = agent.fc2
            self.activation2 = agent.activation2

            self.fc3 = agent.fc3
            self.activation3 = agent.activation3

            self.fc4 = agent.fc4
            self.activation4 = agent.activation4

            self.fc_final = agent.fc_final

            self.optimizer = agent.optimizer
            self.epsilon_start = 1.0
            self.loss_values = agent.loss_values

    def update_epsilon(self):
        self.epsilon = (self.epsilon_end + (self.epsilon_start - self.epsilon_end)
                        * math.exp(-1. * self.frame_idx / self.epsilon_decay))

    def forward(self, x):
        """
        Forward pass through the network.

        :param x: The input tensor/data
        :return: The output tensor/data
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        out = self.fc1(x)
        out = self.activation1(out)

        out = self.fc2(out)
        out = self.activation2(out)

        out = self.fc3(out)
        out = self.activation3(out)

        out = self.fc4(out)
        out = self.activation4(out)

        out = self.fc_final(out)

        return out

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

    def experience_replay(self, batch_size):
        """
        Trains the network using the batch from experiences in memory.

        :param batch_size: The number of experiences to use for each update
        """

        if len(self.memory) < batch_size:
            return
        experiences, indices, weights = self.memory.sample(batch_size)
        state, action, reward, next_state, done = zip(*experiences)

        state = torch.from_numpy(np.array(state)).float().to(self.device)
        next_state = torch.from_numpy(np.array(next_state)).float().to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.int64).to(self.device)
        done = torch.tensor(done, dtype=torch.int8).to(self.device)

        current_q_values = self.forward(state)
        max_next_q_values = self.forward(next_state).detach().max(1)[0]
        expected_q_values = reward + self.gamma * max_next_q_values * (1 - done)

        td_errors = torch.abs(current_q_values.gather(1, action.unsqueeze(-1)) - expected_q_values.unsqueeze(-1))
        loss = F.mse_loss(current_q_values.gather(1, action.unsqueeze(-1)), expected_q_values.unsqueeze(-1))
        self.loss_values.append(np.mean(loss.detach().cpu().numpy()))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors.squeeze().detach().cpu().numpy().tolist())

    def plot_learning_curve(self):
        plt.plot(self.loss_values)
        plt.title('Learning Curve')
        plt.xlabel('Time step')
        plt.ylabel('Loss')
        plt.show()


class LSTMAgent(Agent):
    def __init__(self, memory: ExperienceReplay, state_dim: int, action_dim: int, epsilon_decay: int, env_name: string,
                 learning_rate: float = 1e-3, batch_size: int = 1, gamma: float = 0.99, agent: Agent = None):

        super().__init__(memory, state_dim, action_dim, epsilon_decay, batch_size=batch_size, gamma=gamma)
        if '45x19' in env_name:
            self.hidden_dim = 400
            self.num_layers = 1
        if '15x15' in env_name:
            self.hidden_dim = 200
            self.num_layers = 1
        if '9x9' in env_name:
            self.hidden_dim = 16
            self.num_layers = 1

        if agent is None:
            self.lstm = nn.LSTM(input_size=self.state_dim,
                                hidden_size=self.hidden_dim,
                                num_layers=self.num_layers,
                                batch_first=True).to(self.device)
            self.lstm.apply(init_lstm)

            self.fc1 = nn.Linear(self.hidden_dim, 32).to(self.device)
            self.fc1.apply(init_kaiming_uniform)
            self.activation1 = nn.SELU().to(self.device)

            self.fc2 = nn.Linear(32, 32).to(self.device)
            self.fc2.apply(init_kaiming_uniform)
            self.activation2 = nn.SELU().to(self.device)

            self.fc3 = nn.Linear(32, 32).to(self.device)
            self.fc3.apply(init_kaiming_uniform)
            self.activation3 = nn.SELU().to(self.device)

            self.fc_final = nn.Linear(32, self.action_dim).to(self.device)

            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, amsgrad=True)

        else:
            self.lstm = agent.lstm

            self.fc1 = agent.fc1
            self.activation1 = agent.activation1

            self.fc2 = agent.fc2
            self.activation2 = agent.activation2

            self.fc3 = agent.fc3
            self.activation3 = agent.activation3

            self.fc_final = agent.fc_final

            self.optimizer = agent.optimizer
            self.epsilon_start = 1.0
            self.loss_values = agent.loss_values

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

        out = self.fc1(out[:, -1, :])
        out = self.activation1(out)

        out = self.fc2(out)
        out = self.activation2(out)

        out = self.fc3(out)
        out = self.activation3(out)

        out = self.fc_final(out)

        return out

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

    def experience_replay(self, batch_size):
        """
        Trains the network using the batch from experiences in memory.

        :param batch_size: The number of experiences to use for each update
        """

        if len(self.memory) < batch_size:
            return
        experiences, indices, weights = self.memory.sample(batch_size)
        state, action, reward, next_state, done = zip(*experiences)

        state = torch.from_numpy(np.array(state)).float().to(self.device)
        next_state = torch.from_numpy(np.array(next_state)).float().to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.int64).to(self.device)
        done = torch.tensor(done, dtype=torch.int8).to(self.device)

        current_q_values = self.forward(state)
        max_next_q_values = self.forward(next_state).detach().max(1)[0]
        expected_q_values = reward + self.gamma * max_next_q_values * (1 - done)

        td_errors = torch.abs(current_q_values.gather(1, action.unsqueeze(-1)) - expected_q_values.unsqueeze(-1))
        loss = F.mse_loss(current_q_values.gather(1, action.unsqueeze(-1)), expected_q_values.unsqueeze(-1))
        self.loss_values.append(np.mean(loss.detach().cpu().numpy()))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors.squeeze().detach().cpu().numpy().tolist())

    def plot_learning_curve(self):
        plt.plot(self.loss_values)
        plt.title('Learning Curve')
        plt.xlabel('Time step')
        plt.ylabel('Loss')
        plt.show()


def train(agent: Agent, env: Env, single_state: np.ndarray, start: (int, int), target: (int, int),
          batch_size: int = 1, sequence_length: int = 4, max_epoch: int = 3):
    single_state = single_state.flatten()

    for epoch in range(max_epoch):
        if epoch == 0:
            agent.epsilon_start = 1.0
        if isinstance(agent, LSTMAgent):
            state = np.zeros((sequence_length,) + single_state.shape)
            state[-1] = single_state
        else:
            state = single_state

        done = False
        target_reached = False
        total_reward = 0
        frame_idx = 0
        explored_positions = [start]

        # Resetting agent's position
        env.reset()

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

            # Store this transition in the agent's memory for future training
            agent.remember(state, action, reward, next_state, done)

            if isinstance(agent, LSTMAgent):
                # Shift the states in the sequence and append the new state
                state[:-1] = state[1:]
                state[-1] = next_state
            else:
                state = next_state

            # Perform experience replay and update the weights of the DQN
            agent.experience_replay(batch_size)

            agent.frame_idx += 1
            frame_idx += 1
            total_reward += reward
            total_reward = round(total_reward, 2)
            print(f'\rEpoch: {epoch + 1} | Frame: {frame_idx} | Reward: {total_reward} ', end='', flush=True)

        if target_reached:
            break
        agent.epsilon_start = max(0.1, agent.epsilon_start - 0.3)

    return target_reached, explored_positions
