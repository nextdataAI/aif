import random
import numpy as np
from collections import deque
from abc import abstractmethod

__all__ = ['ExperienceReplay', 'DefaultExperienceReplay', 'PrioritizedExperienceReplay']


class ExperienceReplay:

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def push(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def sample(self, batch_size):
        pass

    @abstractmethod
    def __len__(self):
        pass


class DefaultExperienceReplay(ExperienceReplay):

    def __init__(self, memory_capacity: int):
        super().__init__()
        self.memory_capacity = memory_capacity
        self.experience_buffer = deque(maxlen=self.memory_capacity)

    def push(self, state, action, reward, next_state, done):
        self.experience_buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if len(self) < batch_size:
            return None
        experiences = random.sample(self.experience_buffer, batch_size)
        return experiences

    def __len__(self):
        return len(self.experience_buffer)


class PrioritizedExperienceReplay(ExperienceReplay):
    def __init__(self, memory_capacity: int, alpha: float = 0.6):
        super().__init__()
        self.alpha = alpha
        self.memory_capacity = memory_capacity
        self.experience_buffer = deque(maxlen=self.memory_capacity)
        self.priorities = deque(maxlen=self.memory_capacity)

    def push(self, state_sequence, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.experience_buffer.append((state_sequence, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        if len(self) < batch_size:
            return None
        sampling_probabilities = np.array(self.priorities) ** self.alpha
        sampling_probabilities /= sampling_probabilities.sum()
        indices = np.random.choice(len(self.experience_buffer), batch_size, p=sampling_probabilities)
        experiences = [self.experience_buffer[index] for index in indices]
        return experiences, indices, sampling_probabilities[indices]

    def update_priorities(self, indices, errors):
        new_priorities = [error + 1e-4 for error in errors]
        for index, priority_value in zip(indices, new_priorities):
            self.priorities[index] = priority_value

    def __len__(self):
        return len(self.experience_buffer)
