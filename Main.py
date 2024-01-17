import sys
import pandas as pd
from algorithms import *
import numpy as np
df = pd.DataFrame(columns=['path', 'name'], dtype=object)

num_examples = 100

env_name = 'MiniHack-MazeWalk-Mapped-45x19-v0'

algorithms = [
    BFS(env_name),
    DFS(env_name),
    AStar(env_name),
    Greedy(env_name),
    Dijkstra(env_name),
    # hmm,
    # lstm
]

results = []
for i in range(num_examples):
    rand_seed = np.random.randint(0, sys.maxsize)
    results.append([(alg(rand_seed), alg.name) for alg in algorithms])
    # insert into df
    local_df = pd.DataFrame(results[-1], columns=['path', 'name'])
    df = df.append(local_df, ignore_index=True)
    print(df)
    break

