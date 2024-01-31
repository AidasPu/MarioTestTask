import numpy as np
import gym
import multiprocessing
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


env_name = 'SuperMarioBros-v0'
env = gym.make(env_name, apply_api_compatibility=True, render_mode="human")
env.metadata["render_fps"] = 200
env = JoypadSpace(env, SIMPLE_MOVEMENT)

class TemporalNeuralNetwork:
    def __init__(self, input_size, output_size, hidden_layers):
        self.weights = [np.random.randn(input_size, hidden_layers[0])]
        for i in range(len(hidden_layers) - 1):
            self.weights.append(np.random.randn(hidden_layers[i], hidden_layers[i + 1]))
        self.weights.append(np.random.randn(hidden_layers[-1], output_size))

    def predict(self, input_data):
        frames = [data for data in input_data]

        # Continue as before
        flattened_data = np.concatenate(frames).flatten()
        for weight in self.weights[:-1]:
            flattened_data = np.tanh(np.dot(flattened_data, weight))

        flattened_data = np.dot(flattened_data, self.weights[-1])
        return np.argmax(flattened_data)


def shaped_reward(reward, done, info, coins_collected):
    shaped = reward
    if done and info['flag_get']:
        shaped += 50
    if done and not info['flag_get']:
        shaped -= 50
    if done and coins_collected < 5:
        shaped -= 100
    return shaped


def evaluate(network):
    local_env = gym.make(env_name, apply_api_compatibility=True)
    local_env = JoypadSpace(local_env, SIMPLE_MOVEMENT)
    total_reward = 0
    obs, _ = env.reset()  # Get the observation directly
    done = False
    coins_collected = 0
    # print("Type of observation:", type(obs))
    # print("Raw observation value:", obs)
    past_frames = [np.zeros_like(obs) for _ in range(2)]

    while not done:
        current_input = past_frames + [obs]
        action = network.predict(current_input)
        next_obs, reward, done, _, info = env.step(action)
        coins_collected = info['coins']
        total_reward += shaped_reward(reward, done, info, coins_collected)
        past_frames.pop(0)
        past_frames.append(next_obs)
        obs = next_obs
    local_env.close()
    print(f"reward: {total_reward}")
    return total_reward


def crossover(parent1, parent2):
    child = TemporalNeuralNetwork(input_size, output_size, hidden_layers)
    for i in range(len(child.weights)):
        print(
            f"Layer {i}: Child shape = {child.weights[i].shape}, Parent1 shape = {parent1.weights[i].shape}, Parent2 shape = {parent2.weights[i].shape}")
        for j in range(len(child.weights[i])):
            if np.random.rand() < 0.5:
                child.weights[i][j] = parent1.weights[i][j]
            else:
                child.weights[i][j] = parent2.weights[i][j]
    return child


def mutate(network, rate=0.02):
    for w in network.weights:
        if np.random.rand() < rate: w += np.random.randn() * 0.5


def parallel_evaluate(networks):
    with multiprocessing.Pool(processes=1) as pool:
        scores = pool.map(evaluate, networks)
    return scores


def save_network(network, generation):
    weights_dict = {}
    for idx, weight in enumerate(network.weights):
        weights_dict[f"layer_{idx}"] = weight
    np.savez(f"best_network_gen_{generation}.npz", **weights_dict)

def load_network(generation):
    data = np.load(f"best_network_gen_{generation}.npz", allow_pickle=True)
    network = TemporalNeuralNetwork(input_size, output_size, hidden_layers)
    for idx in range(len(network.weights)):
        network.weights[idx] = data[f"layer_{idx}"]
    print(generation)
    return network

input_size = 240 * 256 * 3 * 3
output_size = env.action_space.n
hidden_layers = [16, 16]
batch_size = 5
population_size = 50
generations = 100
elitism_count = 10

initial_generation = 10

if initial_generation > 0:
    best_network = load_network(initial_generation)

    population = [best_network] + [TemporalNeuralNetwork(input_size, output_size, hidden_layers) for _ in range(population_size - 1)]
else:
    population = [TemporalNeuralNetwork(input_size, output_size, hidden_layers) for _ in range(population_size)]

for generation in range(initial_generation, generations):
    scores = parallel_evaluate(population)
    print(f"Generation {generation}: Max Score = {max(scores)}, Average Score = {np.mean(scores)}")

    elites = [population[i] for i in np.argsort(scores)[-elitism_count:]]
    children = []

    for _ in range(population_size - elitism_count):
        parent1, parent2 = np.random.choice(elites, 2)
        child = crossover(parent1, parent2)
        mutate(child)
        children.append(child)
    population = elites + children

    best_network = population[np.argmax(scores)]
    save_network(best_network, generation)

env.close()

