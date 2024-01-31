import time
import numpy as np
import gym
import multiprocessing
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Concept untested way to skip Frames which are not needed for training.
class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip):
        super(FrameSkip, self).__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, not_needed, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, not_needed, info


env_name = 'SuperMarioBros-v0'
env = gym.make(env_name, apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = FrameSkip(env, skip=4)

input_size = 240 * 256 * 3 * 3
output_size = env.action_space.n
hidden_layers = [32, 32, 32]
batch_size = 5
population_size = 50
generations = 100
elitism_percentage = 0.2
mutation_rate = 0.02
adaptive_mutation = True
initial_generation = 0
dropout_rate = 0.5


class TemporalNeuralNetwork:
    def __init__(self, input_size, output_size, hidden_layers):
        self.weights = [np.random.randn(input_size, hidden_layers[0])]
        for i in range(len(hidden_layers) - 1):
            self.weights.append(np.random.randn(hidden_layers[i], hidden_layers[i + 1]))
        self.weights.append(np.random.randn(hidden_layers[-1], output_size))

    def predict(self, input_data):
        frames = [data for data in input_data]
        flattened_data = np.concatenate(frames).flatten()

        for i, weight in enumerate(self.weights[:-1]):
            flattened_data = np.dot(flattened_data, weight)
            flattened_data = np.maximum(0.01 * flattened_data, flattened_data)
            dropout_mask = (np.random.rand(*flattened_data.shape) > dropout_rate).astype(np.float32)
            flattened_data *= dropout_mask

        flattened_data = np.dot(flattened_data, self.weights[-1])
        return np.argmax(flattened_data)


def shaped_reward(reward, done, info, coins_collected, last_y_pos):
    shaped = reward
    if 'x_pos' in info:
        shaped += info['x_pos'] * 0.01
    if 'y_pos' in info:
        shaped += (info['y_pos'] - last_y_pos) * 0.000001
        last_y_pos = info['y_pos']
    shaped += coins_collected * 5
    if done and info['flag_get']:
        shaped += 1000000
    if done and not info['flag_get']:
        shaped -= 500
    if done and coins_collected < 5:
        shaped -= 1000

    return shaped, last_y_pos


def evaluate(network):
    start = time.time()
    local_env = gym.make(env_name, apply_api_compatibility=True, render_mode="human")
    local_env = JoypadSpace(local_env, SIMPLE_MOVEMENT)
    total_reward = 0
    obs, _ = local_env.reset()
    done = False

    past_frames = [np.zeros_like(obs) for _ in range(2)]

    last_x_position = None
    last_movement_time = start
    last_y_pos = 0
    while not done:
        current_input = past_frames + [obs]
        action = network.predict(current_input)

        next_obs, reward, done, _, info = local_env.step(action)

        coins_collected = info['coins']
        reward, last_y_pos = shaped_reward(reward, done, info, coins_collected, last_y_pos)
        total_reward += reward
        past_frames.pop(0)
        past_frames.append(next_obs)
        obs = next_obs

        current_x_position = info['x_pos']
        if last_x_position is None or current_x_position != last_x_position:
            last_x_position = current_x_position
            last_movement_time = time.time()
        elif time.time() - last_movement_time >= 10:
            total_reward -= 50
            break

    local_env.close()
    print(f"time_taken {time.time() - start}")
    print(f"reward: {total_reward}")
    return total_reward


def crossover(parent1, parent2):
    child = TemporalNeuralNetwork(input_size, output_size, hidden_layers)
    for i in range(len(child.weights)):
        for j in range(len(child.weights[i])):
            if np.random.rand() < 0.5:
                child.weights[i][j] = parent1.weights[i][j]
            else:
                child.weights[i][j] = parent2.weights[i][j]
    return child


def mutate(network, mutation_factor=1.0):
    for w in network.weights:
        if np.random.rand() < mutation_rate * mutation_factor:
            w += np.random.randn() * 0.5


def tournament_selection(population, scores, k=3):
    selected_indices = np.random.choice(len(population), k)
    selected_scores = [scores[i] for i in selected_indices]
    winner_index = selected_indices[np.argmax(selected_scores)]
    return population[winner_index]


def parallel_evaluate(networks):
    with multiprocessing.Pool(processes=4) as pool:
        scores = pool.map(evaluate, networks)
    return scores


def save_network(network, generation):
    weights_dict = {}
    for idx, weight in enumerate(network.weights):
        weights_dict[f"layer_{idx}"] = weight
    np.savez(f"bestv3_network_gen_{generation}.npz", **weights_dict)


def load_network(generation):
    data = np.load(f"bestv3_network_gen_{generation}.npz", allow_pickle=True)
    network = TemporalNeuralNetwork(input_size, output_size, hidden_layers)
    for idx in range(len(network.weights)):
        network.weights[idx] = data[f"layer_{idx}"]
    print(generation)
    return network


if initial_generation > 0:
    best_network = load_network(initial_generation)
    population = [best_network] + [TemporalNeuralNetwork(input_size, output_size, hidden_layers) for _ in
                                   range(population_size - 1)]
else:
    population = [TemporalNeuralNetwork(input_size, output_size, hidden_layers) for _ in range(population_size)]

avg_scores = []
for generation in range(initial_generation, generations):
    scores = parallel_evaluate(population)

    avg_scores.append(np.mean(scores))
    if adaptive_mutation:
        if len(avg_scores) > 2 and avg_scores[-1] < avg_scores[-2]:
            mutation_factor = 1.2
        else:
            mutation_factor = 0.8
    else:
        mutation_factor = 1.0

    print(
        f"Generation {generation}: Max Score = {max(scores)}, Average Score = {np.mean(scores)}, Median Score = {np.median(scores)}, Std Deviation = {np.std(scores)}")

    num_elites = int(elitism_percentage * population_size)
    elites_indices = np.argsort(scores)[-num_elites:]
    elites = [population[i] for i in elites_indices]

    children = []
    for _ in range(population_size - num_elites):
        parent1 = tournament_selection(population, scores)
        while True:
            parent2 = tournament_selection(population, scores)
            if parent2 != parent1:
                break
        child = crossover(parent1, parent2)
        mutate(child, mutation_factor)
        children.append(child)

    population = elites + children

    best_network = population[np.argmax(scores)]
    save_network(best_network, generation)

env.close()

# def play_best_network(generation):
#     best_network = load_network(generation)
#
#     local_env = gym.make(env_name, apply_api_compatibility=True, render_mode="human")
#     local_env = JoypadSpace(local_env, SIMPLE_MOVEMENT)
#     obs, _ = local_env.reset()
#
#     past_frames = [np.zeros_like(obs) for _ in range(2)]
#
#     done = False
#     while not done:
#         local_env.render()
#         current_input = past_frames + [obs]
#         action = best_network.predict(current_input)
#         next_obs, _, done, _, _ = local_env.step(action)
#         past_frames.pop(0)
#         past_frames.append(next_obs)
#         obs = next_obs
#
#     local_env.close()
#
# play_best_network(3)
