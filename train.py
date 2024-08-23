from dqn import DQNAgent
from env import Env
import os
os.system("clear")

env = Env()
state = env._get_obs()
# action_size: all possible actions for 3D binary vector like [1, 0, 1]

agent = DQNAgent(len(state), action_size=8, seed=42)

EPISODES = 100_000

print("Training Started...")
for episode in range(1, EPISODES):
    print(f"Episode: {episode} started!")
    state, info = env.reset()
    done = False
    
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.99
    
    epsilon = epsilon_start
    while not done:
        action = agent.act(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        if terminated or truncated:
            done = True
        epssilon = max(epsilon_end, epsilon_decay*epsilon)
    print(f"Epsiode: {episode}",f"Episode: Reward: {sum(env.rewards)}", sep="\t")

agent.save_model()
        
