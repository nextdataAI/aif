from .utils import get_player_location, get_valid_moves
from .Algorithm import Algorithm
from tqdm import tqdm
try :
    print("Using MLX")
    import mlx as np
except ImportError:
    print("Using numpy")
    import numpy as np

class Brain(Algorithm):
    def __init__(self, env_name, input_size, output_size):
        super().__init__(env_name, name="Brain")
        self.weights = np.random.uniform(-1, 1, (input_size, output_size))
        self.bias = np.random.uniform(-1, 1, (1, output_size))
        self.fitness_score = -np.inf

    def __call__(self, seed):
        local_env, local_state, local_game_map, start, target = super().initialize_env(seed)
        done = False
        total_reward = 0
        while not done:
            # Get the best move
            best_move = self.get_action(local_state['chars'])
            # Make the move
            local_state, reward, done, info = local_env.step(best_move)
            # Update the game map
            local_game_map[best_move] = 1
            # Render the environment
            total_reward += reward
            if done:
                return total_reward

    def get_action(self, state):
        result = (np.dot(np.array(state).flatten(), self.weights) + self.bias)[0]
        while True:
            out = np.argmax(result)
            valid_moves = get_valid_moves(state, get_player_location(state), 'action')
            if out in valid_moves:
                break
            result[out] = float('-inf')

        return out


class Genetic(Algorithm):
    def __init__(self, env_name):
        super().__init__(env_name, name="Genetic")
        self.population_size = 50
        self.population = []
        self.mutation_rate = 0.02
        self.max_generations = 2
        self.env_name = env_name
        self.generation = 0
        self.best = None
        self.best_fitness = -np.inf

    def initialize_population(self):
        self.population = []
        self.generation = 0
        if self.best:
            self.best.fitness_score = 0
            self.population.append(self.best)
            self.best_fitness = 0
        for _ in range(self.population_size - 1):
            self.population.append(Brain(self.env_name, self.input_size, self.output_size))

    def run(self):
        self.initialize_population()
        with tqdm(total=self.population_size * self.max_generations, disable=False) as pbar:
            for _ in range(self.max_generations):
                self.generation += 1
                self.fitness(pbar=pbar)
                self.selection()
                self.crossover()
                self.mutation()
                # Remove one random brain from the population
                self.population.pop(np.random.randint(0, len(self.population)))
                self.population.append(self.best)
                if self.best_fitness == 1.0:
                    break

    def fitness(self, pbar):
        for rocket in self.population:
            rocket.fitness_score = rocket(self.seed)
            if rocket.fitness_score > self.best_fitness:
                self.best_fitness = rocket.fitness_score
                self.best = rocket
            pbar.update(1)
            pbar.set_description(f"Generation: {self.generation} | Best Fitness: {self.best_fitness}")
            if self.best_fitness == 1.0:
                break

    def selection(self):
        self.population.sort(key=lambda x: x.fitness_score, reverse=True)
        self.population = self.population[:int(self.population_size / 2)]

    def crossover(self):
        for i in range(int(self.population_size / 2)):
            parent1 = self.population[i]
            parent2 = self.population[i + 1]
            child = Brain(self.env_name, self.input_size, self.output_size)
            child.weights = np.concatenate((parent1.weights[:int(parent1.weights.shape[0] / 2)],
                                            parent2.weights[int(parent2.weights.shape[0] / 2):]))
            self.population.append(child)

    def mutation(self):
        for i in range(int(self.population_size / 2), self.population_size):
            self.population[i].weights += np.random.uniform(-self.mutation_rate, self.mutation_rate,
                                                            self.population[i].weights.shape)
            self.population[i].bias += np.random.uniform(-self.mutation_rate, self.mutation_rate,
                                                         self.population[i].bias.shape)

    def get_action(self, state):
        return self.best.get_action(state)

    def __call__(self, seed):
        self.start_timer()
        local_env, local_state, local_game_map, start, target = super().initialize_env(seed)
        self.input_size = local_game_map.shape[0] * local_game_map.shape[1]
        self.output_size = 4
        self.seed = seed
        self.run()
        # with open('best_brain.npy', 'wb') as f:
        #     self.best.env = None
        #     pickle.dump(self.best, f)
        done = False
        path = []
        while not done:
            # Get the best move
            agent_pos = get_player_location(local_state['chars'])
            best_move = self.get_action(local_state['chars'])
            # Make the move
            local_state, reward, done, info = local_env.step(best_move)
            # Update the game map
            local_game_map[best_move] = 1
            # Render the environment
            local_env.render()
            path.append(agent_pos)

            if done:
                return True if reward == 1.0 else False, path, path, self.stop_timer()
