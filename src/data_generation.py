import os
import gym
import minihack
import pickle

os.makedirs('../data', exist_ok=True)
for i in range(100):
    env = gym.make('MiniHack-MazeWalk-45x19-v0')
    state = env.reset()
    with (open(f'../data/Maze_{i}.pkl', 'wb')) as f:
        pickle.dump(state, f)
