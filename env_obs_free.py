import numpy as np
import gymnasium as gym
from gymnasium import spaces
import itertools
import matplotlib.pyplot as plt
import time

class Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}
    
    def __init__(self, render_mode="human", size=20):
        """
        vehicle: 0
        start_point: 1
        end_point: 2
        """
        if render_mode in self.metadata["render_modes"]:
            self.render_mode = render_mode
        else:
            raise ValueError("Render mode should be 'human'!")
        
        self.size = size
        self.observation_space = list(itertools.product(range(20), repeat=3))
        self.action_space = spaces.MultiBinary(3)
        self._action_to_direction = {
            0:np.array([0, 0, 0]),
            1:np.array([0, 0, 1]),
            2:np.array([0, 1, 0]),
            3:np.array([0, 1, 1]),
            4:np.array([1, 0, 0]),
            5:np.array([1, 0, 1]),
            6:np.array([1, 1, 0]),
            7:np.array([1, 1, 1])
        }
        
        self.actions = {0: -1, # if zero go left/down/back 
                        1: +1} # if one go front/up/right
        self.seed = 42
        self.max_step = 1000
        
        self.start_point = np.array([1, 2, 4])
        self.end_point = np.array([18, 17, 15])
        self.reset()
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.transactions = []
        self.rewards = []
        self.t_step = 0
        self.done = False
        self.agent_position = self.start_point
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        # Initialize boundary values
        self.up_boundary = 0
        self.down_boundary = 0
        self.left_boundary = 0
        self.right_boundary = 0
        self.front_boundary = 0
        self.back_boundary = 0

        # Initialize vehicle position relative to endpoint
        self.veh_up_end = 0
        self.veh_down_end = 0
        self.veh_left_end = 0
        self.veh_right_end = 0
        self.veh_front_end = 0
        self.veh_back_end = 0

        if self.agent_position[1] == 0:
            self.down_boundary = 1
        if self.agent_position[1] == self.size - 1:
            self.up_boundary = 1

        # Check for left and right boundaries
        if self.agent_position[0] == 0:
            self.left_boundary = 1
        if self.agent_position[0] == self.size - 1:
            self.right_boundary = 1

        # Check for front and back boundaries
        if self.agent_position[2] == 0:
            self.back_boundary = 1
        if self.agent_position[2] == self.size - 1:
            self.front_boundary = 1


        if self.agent_position[0] > self.end_point[0]:
            self.veh_left_end = 1
        elif self.agent_position[0] < self.end_point[0]:
            self.veh_right_end = 1 # sağında
            
        
        if self.agent_position[1] > self.end_point[1]:
            self.veh_up_end = 1
        elif self.agent_position[1] < self.end_point[1]:
            self.veh_down_end = 1


        if self.agent_position[2] > self.end_point[2]:
            self.veh_front_end = 1
        elif self.agent_position[2] < self.end_point[2]:
            self.veh_back_end = 1

        return np.array([
            self.start_point[0], self.start_point[1], self.start_point[2],
            self.agent_position[0], self.agent_position[1], self.agent_position[2],
            self.up_boundary, self.down_boundary, self.left_boundary, self.right_boundary, 
            self.front_boundary, self.back_boundary,
            self.veh_up_end, self.veh_down_end, self.veh_left_end, self.veh_right_end,
            self.veh_front_end, self.veh_back_end
        ])
    
    def _get_info(self):
        return {
            "agent_position": self.agent_position,
            "end_point": self.end_point,
            "steps": self.t_step
        }
    
    def reward(self):
        reward = -1
        if np.array_equal(self.start_point, self.end_point):
            reward = 0
        self.rewards.append(reward)
        return reward
    
    def step(self, action):
        self.pre_pos = self.agent_position
        action = self._action_to_direction[int(action)]
        self.t_step += 1
        temp_pos = (self.agent_position[0] + self.actions[action[0]],
                    self.agent_position[1] + self.actions[action[1]],
                    self.agent_position[2] + self.actions[action[2]])
        
        if temp_pos in self.observation_space:
            self.agent_position = np.array(temp_pos)
                
        if np.array_equal(self.agent_position, self.end_point) or self.t_step == self.max_step:
            self.done = True
        
        transaction = np.concatenate((self.pre_pos, self.agent_position))
        if self.t_step == 1:
            transaction = np.concatenate((self.start_point, self.agent_position))
        self.transactions.append(tuple(transaction))
        
        return self._get_obs(), self.reward(), self.done, self._get_info()
               # observation, reward, done, info
               
    def render(self):
        self._render_frame()
                
    def _render_frame(self):
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the start point
        ax.scatter(self.start_point[0], self.start_point[1], self.start_point[2], color='green', s=100, label='Start')

        # Plot the end point
        ax.scatter(self.end_point[0], self.end_point[1], self.end_point[2], color='red', s=100, label='End')

        # Plot the agent's initial position
        ax.scatter(self.agent_position[0], self.agent_position[1], self.agent_position[2], color='blue', s=100, label='Agent')

        # Plot the transactions
        for t in self.transactions:
                # Draw a line between the start and end points of each transaction
            ax.plot([t[0], t[3]], [t[1], t[4]], [t[2], t[5]], color='orange')
                # Plot the end point of each transaction
            ax.scatter(t[3], t[4], t[5], color='orange', s=50)
            
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Grid Environment with Transactions')

        # Add a legend
        ax.legend()

        # Show the plot
        plt.show()
        time.sleep(5)
        plt.close(fig)