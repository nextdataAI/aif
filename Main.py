import sys
import pandas as pd
from algorithms import *
import numpy as np
df = pd.DataFrame(columns=['name', 'path', 'visited', 'time', 'maze', 'explored', 'solution', 'score'], dtype=object)

num_examples = 10

# env_name = 'MiniHack-MazeWalk-Mapped-45x19-v0'
env_name = "MiniHack-MazeWalk-Mapped-15x15-v0"

algorithms = [
    BFS(env_name),
    DFS(env_name),
    AStar(env_name),
    AStar(env_name, h='euclidean', name='ASTAR-EUCLIDEAN'),
    Greedy(env_name),
    Dijkstra(env_name),
    # QLSTM(env_name),
    # Genetic(env_name),
    # Qlearning(env_name)
]

results = []
for i in range(num_examples):
    rand_seed = np.random.randint(0, sys.maxsize)
    results = [tuple([alg.name]) + alg(rand_seed) + tuple([f'Maze_{i}']) for alg in algorithms]
    local_df = pd.DataFrame(results, columns=['name', 'path', 'visited', 'time', 'maze'])
    df = compute_score(local_df, df)

df = df[['name', 'time', 'explored', 'solution', 'path_score', 'score']]
df = df.groupby('name').agg({'time': ['mean', 'std'], 'explored': ['mean', 'std'], 'solution': ['mean', 'std'],
                             'path_score': ['mean', 'std'], 'score': ['mean', 'std']})
df.columns = ['_'.join(col) for col in df.columns.values]
df = df.reset_index()
df.to_csv('results.csv')
