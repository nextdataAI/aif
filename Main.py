import sys
from pathlib import Path


from typing import Callable
from nextdataAI import BFS, DFS, Greedy, AStar, Dijkstra

from typing import Callable
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path.cwd().parent))
dft = pd.DataFrame(
    columns=['name', 'solved', 'path', 'visited', 'time', 'maze', 'seed', 'path_length', 'explored'],
    dtype=object)

num_examples = 1

env_name1 = 'MiniHack-MazeWalk-Mapped-45x19-v0'
env_name2 = "MiniHack-MazeWalk-Mapped-15x15-v0"
env_name3 = "MiniHack-MazeWalk-Mapped-9x9-v0"

# NNHeuristic1 = NNManhattan(model='NNManhattan')

algorithms = [

    AStar(env_name1, name='ASTAR-Manhattan-Env1'),
    # AStar(env_name2, name='ASTAR-Manhattan-Env2'),
    # AStar(env_name3, name='ASTAR-Manhattan-Env3'),
    #
    # AStar(env_name1, h='chebysev', name='ASTAR-Chebysev-Env1'),
    # AStar(env_name2, h='chebysev', name='ASTAR-Chebysev-Env2'),
    # AStar(env_name3, h='chebysev', name='ASTAR-Chebysev-Env3'),
    #
    # AStar(env_name1, h='euclidean', name='ASTAR-EUCLIDEAN-Env1'),
    # AStar(env_name2, h='euclidean', name='ASTAR-EUCLIDEAN-Env2'),
    # AStar(env_name3, h='euclidean', name='ASTAR-EUCLIDEAN-Env3'),
    #
    AStar(env_name1, h='smanhattan', name='ASTAR-SManhattan-Env1'),
    # AStar(env_name2, h='smanhattan', name='ASTAR-SManhattan-Env2'),
    # AStar(env_name3, h='smanhattan', name='ASTAR-SManhattan-Env3'),
    #
    # BFS(env_name1, name='BFS-Env1'),
    # BFS(env_name2, name='BFS-Env2'),
    # BFS(env_name3, name='BFS-Env3'),
    #
    # DFS(env_name1, name='DFS-Env1'),
    # DFS(env_name2, name='DFS-Env2'),
    # DFS(env_name3, name='DFS-Env3'),
    #
    Greedy(env_name1, name='GREEDY-Env1'),
    # Greedy(env_name2, name='GREEDY-Env2'),
    # Greedy(env_name3, name='GREEDY-Env3'),
    #
    # Greedy(env_name1, h='smanhattan', name='GREEDY-SManhattan-Env1'),
    # Greedy(env_name2, h='smanhattan', name='GREEDY-SManhattan-Env2'),
    # Greedy(env_name3, h='smanhattan', name='GREEDY-SManhattan-Env3'),

    # Dijkstra(env_name1, name='DIJKSTRA-Env1'),
    # Dijkstra(env_name2, name='DIJKSTRA-Env2'),
    # Dijkstra(env_name3, name='DIJKSTRA-Env3'),

    # AStar(env_name1, h='NNManhattan', name='ASTAR-NNManhattan-Env1'),
    # AStar(env_name2, h=NNHeuristic1, name='ASTAR-NNManhattan-Env2'),
    # AStar(env_name3, h=NNHeuristic1, name='ASTAR-NNManhattan-Env3'),

    # Genetic(env_name3),
    #
    #
    #
    # QLSTM(env_name),
    # Qlearning(env_name)
    # AlphaBeta(env_name, depth=5),
]


def call(algorithm: Callable, seed: int, i: int, name: str, env_name: str, pbar: tqdm):
    output = tuple([name]) + algorithm(seed) + tuple([f'{env_name}_{i}']) + tuple([seed])
    pbar.update(1)
    return output


rand_seed = np.random.randint(0, sys.maxsize)
with tqdm(total=num_examples * len(algorithms)) as pbar:
    for i in range(num_examples):
        # insert into df
        df = pd.DataFrame(
            [call(algorithm=alg, seed=rand_seed, i=i, name=alg.name, env_name=alg.env_name, pbar=pbar) for alg in
             algorithms],
            columns=['name', 'solved', 'path', 'visited', 'time', 'maze', 'seed'])
        # Save for each maze keep only the one with the lowest path length
        df['path_length'] = df['path'].apply(lambda x: len(x))
        df['explored'] = df['visited'].apply(lambda x: len(x))
        # df = df.sort_values(by=['maze', 'path_length'])
        # df = df.drop_duplicates(subset=['maze'], keep='first')
        df = df.reset_index(drop=True)

        if dft.empty:
            dft = df
        else:
            dft = pd.concat([dft, df])

dft.to_csv('results_all2.csv')
