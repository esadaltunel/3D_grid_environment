from agent import DQNAgent
from env import Env
import time
import numpy as np
import csv
import os ; os.system("clear")


env = Env()
state_size = len(env._get_obs())
action_size = len(env.action_space)
agent = DQNAgent(state_size = state_size, action_size = action_size, seed = 42)

EPISODES = 100_000

epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.999
epsilon = epsilon_start
all_rewards = 0.0
gmr = 0.0

file_exists = os.path.isfile("rewards.csv")
with open("rewards.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(["Episode", "GMR", "Reward", "Epsilon"])
        
print("Training Started...")
for episode in range(1, EPISODES):
    print(f"Episode: {episode} started!")
    done = False
    state, info = env.reset()
    while not done:
        action = agent.act(state, epsilon)
        next_state, reward, done, info = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
    epsilon = max(epsilon_end, epsilon_decay*epsilon)
    all_rewards += env.total_reward
    gmr = round((all_rewards / episode), 2)
    with open("rewards.csv", mode="a", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([episode, gmr, round(env.total_reward, 2), epsilon])
agent.save_model()