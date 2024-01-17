import sys
import pandas as pd
from algorithms import *
import numpy as np
df = pd.DataFrame(columns=['name', 'path', 'visited', 'time', 'maze', 'explored', 'solution', 'score'], dtype=object)

num_examples = 100

env_name = 'MiniHack-MazeWalk-Mapped-45x19-v0'

algorithms = [
    BFS(env_name),
    DFS(env_name),
    AStar(env_name),
    AStar(env_name, h='euclidean', name='ASTAR-EUCLIDEAN'),
    Greedy(env_name),
    Dijkstra(env_name),
    # hmm,
    # lstm
]

results = []
for i in range(num_examples):
    rand_seed = np.random.randint(0, sys.maxsize)
    results = [tuple([alg.name]) + alg(rand_seed) + tuple([f'Maze_{i}']) for alg in algorithms]
    # insert into df
    local_df = pd.DataFrame(results, columns=['name', 'path', 'visited', 'time', 'maze'])
    local_df['explored'] = local_df['visited'].apply(lambda x: len(x))
    local_df['solution'] = local_df['path'].apply(lambda x: len(x))
    local_df['ex_sol_score'] = local_df['solution']/local_df['explored']
    #  Score = sum of difference between steps taken in solution
    local_df['score'] = local_df['path'].apply(lambda x: sum(abs(x[i][0] - x[i+1][0]) + abs(x[i][1] - x[i+1][1]) for i in range(len(x)-1)))
    local_df['score'] = 1 - local_df['score']/ max(local_df['score'])
    df = pd.concat([local_df, df])

df = df[['name', 'time', 'explored', 'solution', 'ex_sol_score', 'score']]
df = df.groupby('name').agg({'time': ['mean', 'std'], 'explored': ['mean', 'std'], 'solution': ['mean', 'std'], 'ex_sol_score': ['mean', 'std'], 'score': ['mean', 'std']})
df.columns = ['_'.join(col) for col in df.columns.values]
df = df.reset_index()
df.to_csv('results.csv')
