#
# from algorithms import *
# my_star = AStar(env_name='MiniHack-MazeWalk-Mapped-45x19-v0', h=CNNHeuristic(), name='ASTAR-CNN')
# my_star(1)
#
import gym
import minihack
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

# def create_env(local_seed: int, env_name, maze_num, kind, minimum_moves):
#     env = gym.make(env_name, observation_keys=("chars", "pixel"),  # reward_manager=reward_manager,
#                    max_episode_steps=100)
#     env.seed(local_seed)
#     local_state = env.reset()
#     local_game_map = local_state.get('chars')
#     save_path = f'data/maze_ascii_dataset/{kind}/{local_seed}_{env_name}_{maze_num}_{minimum_moves}.ascii'
#     local_game_map = np.array(local_game_map)
#     np.save(save_path, local_game_map)
#     env.close()


for dataset in ['train', 'val', 'test']:
    descriptor = pd.DataFrame(columns=['file_name', 'label'])
    figures = os.listdir(f'data/maze_images_dataset/{dataset}')
    npy = os.listdir(f'data/maze_ascii_dataset/{dataset}')
    print(len(figures))
    print(len(npy))
    for file in tqdm(npy, desc=f'Removing .ascii from {dataset} dataset'):
        elem = file.split(".")
        if elem[1].endswith('ascii'):
            os.rename(f'data/maze_ascii_dataset/{dataset}/{file}', f'data/maze_ascii_dataset/{dataset}/{elem[0]}.{elem[2]}')
    # for figure in tqdm(figures, desc=f'Creating {dataset} dataset'):
    #     if figure.endswith('.png'):
    #         info = figure.split('_')
    #         seed = info[0]
    #         name = info[1]
    #         maze_number = info[2]
    #         min_moves = info[3].split('.')[0]
    #         new_name = f'{seed}_{name}_{maze_number}_{min_moves}'
    #         descriptor = pd.concat([descriptor, pd.DataFrame([[new_name, min_moves]], columns=['file_name', 'label'])])
    #         create_env(int(seed), name, maze_number, dataset, min_moves)
    # descriptor.to_csv(f'data/maze_images_dataset/{dataset}_data.csv', index=False)




