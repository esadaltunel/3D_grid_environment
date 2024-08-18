import numpy as np
from env import Env
num_steps = 5
obstacle_list = []
for step in range(num_steps):
    num_obstacles = np.random.randint(1, 5)  # Random number of obstacles between 1 and 4 for each step
    obstacles = [np.random.randint(0, 10, size=3) for _ in range(num_obstacles)]  # Random coordinates between 0 and 9
    obstacle_list.append(obstacles)

# Convert to a NumPy array of objects to handle inhomogeneous data
obstacle_list = np.array(obstacle_list, dtype=object)

env = Env()
env.transactions = np.array([[1, 2, 3], [2, 3, 4], [13, 14, 14], [17, 18, 16]])
env.detected_obstacles = obstacle_list
env.rewards = np.array([4, 4, 4, 4])
env.render()