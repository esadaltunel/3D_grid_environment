from agent import DQNAgent
from env import Env
import time
import os ; os.system("clear")


env = Env()
state_size = len(env._get_obs())
action_size = len(env.action_space)

agent = DQNAgent(state_size = state_size, action_size = action_size, seed = 42)

EPISODES = 100_000

print("Training Started...")
for episode in range(1, EPISODES):
    print(f"Episode: {episode} started!")
    state, info = env.reset()    
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    done = False
    
    epsilon = epsilon_start
    while not done:
        action = agent.act(state, epsilon)
        next_state, reward, done, info = env.step(action)
        agent.step(state, action, reward, next_state, done)
        # print(f"State: {state}", f"Action: {action}", f"Reward: {reward}", f"Next State: {next_state}", sep="\n")
        # time.sleep(0.5)
        state = next_state
    epsilon = max(epsilon_end, epsilon_decay*epsilon)
    with open("rewards.txt", "a") as file:
        file.write(f"Epsiode: {episode} Reward: {sum(env.rewards)}")
    

agent.save_model()
        
