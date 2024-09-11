import numpy as np
import itertools 
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import random

"""
Acion space: 26
Observation space: 800
"""

class Tools:
    def space_creater(min_range, max_range, dim):
        return list(itertools.product(range(min_range, max_range), repeat=dim))
    
    def is_empty(arr):
        return len(arr) == 0
    
    def pair_sorter(lst):
        lst_result = lst
        return sorted(lst_result, key=lambda x: x[1])
    def get_unique_list(lst):
    # Helper function to convert sublist (array + float) to a tuple
        def convert_to_tuple(sublist):
            array_as_tuple = tuple(sublist[0])  # Convert numpy array to tuple
            return (array_as_tuple, sublist[1])

        # Create a set of unique tuples
        unique_set = set(convert_to_tuple(sublist) for sublist in lst)

        # Convert back tuples to original format with numpy arrays
        unique_lst_with_arrays = [[np.array(tup[0]), tup[1]] for tup in unique_set]

        return unique_lst_with_arrays

            

class Env(gym.Env):
    metadata = {"render_mode":["rgb", "terminal"],
                "render_fps": 1}
    
    def __init__(self, render_mode = "rgb", size = 20, num_obst = 200, 
                 seed = 42, max_step = 1000):
        if render_mode not in self.metadata["render_mode"]:
            print("Invalid render mode !")
        
        # Basic variables
        self._render_mode = render_mode
        self._size = size
        self._num_obst = num_obst
        self._seed = seed
        self._max_step = max_step
        self._start = np.array([2, 4, 3])
        self._target = np.array([18, 16, 19])
        self.observation_space = Tools.space_creater(0, 20, 3)
        """
        observation_space:
        elements -> tuple
        type -> list
        """
        self.action_space = Tools.space_creater(-1, 2, 3)
        """
        action_space:
        elements -> tuple
        type -> list
        """
        
        # OBSTACLE CREATION
        start_idx = self.observation_space.index(tuple(self._start))
        target_idx = self.observation_space.index(tuple(self._target))
        space = Tools.space_creater(0, 20, 3)
        space.pop(start_idx)
        space.pop(target_idx)
        self.obstacles = random.sample(space, 200)
        random.seed(self._seed)
        self.reset()
        
    def reset(self):
        self._agent = self._start
        self.done = False
        self._t_step = 0
        self.total_reward = 0.0
        self.transactions = []
        self.all_detected_obst = []
        self._sensor()
        return self._get_obs(), self.info()
    
    def step(self, action):
        dir = np.array(self.action_space[action])
        temp_pos = tuple(self._agent + dir)
        
        if temp_pos in self.observation_space:
            self._agent = np.array(temp_pos)
        
        self._t_step += 1
        self.transactions.append(tuple(self._agent))
        
        if np.array_equal(self._agent, self._target) or self._t_step == self._max_step:
            self.done = True
            
        return self._get_obs(), self.reward(), self.done, self.info()
    
    def reward(self):
        # Initialize the reward for the step
        reward = 0
        
        # Check if the agent reached the goal
        if np.array_equal(self._agent, self._target):
            reward += 100  # Large reward for reaching the goal
        else:
            # Penalize for each step to encourage shorter paths
            reward -= 1
            
            # Distance-based reward to encourage getting closer to the goal
            distance_to_goal = np.linalg.norm(self._agent - self._target)
            reward -= round(distance_to_goal, 2)  # Adjust the scaling factor as needed
            
            # Penalize revisiting states to prevent loops
            if tuple(self._agent) in self.transactions:
                reward -= 10  # Adjust penalty as needed
            
            # Penalize collision with obstacles
            if tuple(self._agent) in self.obstacles:
                reward -= 50  # Large penalty for colliding with an obstacle
        self.total_reward += reward
        return reward
    
    def _get_obs(self): 
        self._sensor()
        self.pre_state = Tools.get_unique_list(self.obs_dist)
        self.pre_state = Tools.pair_sorter(self.pre_state)
        state = np.array([])
        if len(self.pre_state) >= 5:
            for i in range(5):
                pos, dirr = self.pre_state[i]
                state = np.append(state, pos)
                state = np.append(state, dirr)
        else:
            for i in range(len(self.pre_state)):
                pos, dirr = self.pre_state[i]
                state = np.append(state, pos)
                state = np.append(state, dirr)
            for _ in range(20 - len(state)):
                state = np.append(state, 0)
        return state
            
    def _sensor(self, depth = 2):
        self.detected_obst = []
        self.obs_dist = []
        for dep in range(1, depth + 1):
            for dir in self.action_space:
                neighbor = (self._agent + np.array(dir))*dep
                if tuple(neighbor) in self.obstacles:
                    self.detected_obst.append(neighbor)
                    self.all_detected_obst.append(neighbor)
                    
                    dist = np.linalg.norm(self._agent - neighbor)
                    pair = [neighbor, round(dist, 3)]
                    self.obs_dist.append(pair)

    def info(self):
        return {
            "agent position": self._agent,
            "distance to target": np.linalg.norm(self._agent - self._target),
            "target point": self._target
        }
    
    def render(self, show_all_obst = False):
        if self._render_mode == "rgb":
            self._render_frame(show_all_obst)
        elif self._render_mode == "terminal":
            print("Transactions\n")
            for tran in self.transactions:
                print(tran)
    
    def _render_frame(self, show_all_obst): # TODO
        # Create a 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Plot the start point
        ax.scatter(self._start[0], self._start[1], self._start[2], color='green', s=100, label='Start')

        # Plot the end point
        ax.scatter(self._target[0], self._target[1], self._target[2], color='red', s=100, label='End')

        # Plot the agent's initial position
        ax.scatter(self._agent[0], self._agent[1], self._agent[2], color='blue', s=100, label='Agent')

        # Plot the transactions
        if not Tools.is_empty(self.transactions):
            for i in range(len(self.transactions)):
                pre_t = self.transactions[i - 1]
                cur_t = self.transactions[i]
                if i == 0:
                    ax.scatter(cur_t[0], cur_t[1], cur_t[2], color="orange", s=50)
                else:
                    # Draw a line between the start and end points of each transaction
                    ax.plot([pre_t[0], cur_t[0]], [pre_t[1], cur_t[1]], [pre_t[2], cur_t[2]], color='orange')
                    # Plot the end point of each transaction
                    ax.scatter(cur_t[0], cur_t[1], cur_t[2], color='orange', s=50)
                
        if not show_all_obst:
            for obst in self.all_detected_obst:
                ax.scatter(obst[0], obst[1], obst[2], color="black", s=30)  
        else:
            for obst in self.obstacles:
                ax.scatter(obst[0], obst[1], obst[2], color="black", s=30)

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Grid Environment with Transactions')

        # Add a legend
        ax.legend()

        # Add text outside the 3D plot
        plt.figtext(0.15, 0.95, f'Start Point: {self._start}', fontsize=12, color='green', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.5'))
        plt.figtext(0.15, 0.90, f'End Point: {self._target}', fontsize=12, color='red', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))
        plt.figtext(0.15, 0.85, f'Total Rewards: {self.total_reward}', fontsize=12, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

        # Show the plot
        plt.show()