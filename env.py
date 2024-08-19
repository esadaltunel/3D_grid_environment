import numpy as np
import gymnasium as gym
from gymnasium import spaces
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}
    
    def __init__(self, render_mode="human", size=20, num_obstacles=400):
        """
        vehicle: 0
        start_point: 1
        end_point: 2
        obstacle: 3
        """
        if render_mode in self.metadata["render_modes"]:
            self.render_mode = render_mode
        else:
            raise ValueError("Render mode should be 'human'!")
        
        self.size = size
        self.num_obstacles = num_obstacles
        self.observation_space = spaces.MultiDiscrete([20] * 18)
        self.action_space = spaces.MultiBinary(3)
        self.actions = {0: -1, # if zero go left/down/back 
                        1: +1} # if one any axis go front/up/right
        self.seed = 42
        self.max_step = 1000
        
        self.obstacles = [np.random.randint(0, size, size=(1, 3)) for _ in range(num_obstacles)]
        self.start_point = np.array([1, 2, 4])
        self.end_point = np.array([18, 17, 15])
        self.reset()
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.transactions = [self.start_point]
        self.rewards = []
        self.detected_obstacles = []
        self.t_step = 0
        self.terminated = False
        self.truncated = False
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
        self.t_step += 1
        temp_pos = [self.agent_position[0] + self.actions[action[0]],
                    self.agent_position[1] + self.actions[action[1]],
                    self.agent_position[2] + self.actions[action[2]]]
        
        if temp_pos in self.observation_space:
            if temp_pos not in self.detected_obstacles:
                self.agent_position = temp_pos
                
        if np.array_equal(self.start_point, self.end_point):
            self.terminated = True
                    
        elif self.t_step == self.max_step:
            self.truncated = True
        self.transactions.append(self.agent_position)
        self.sensor()
        
        return self._get_obs(), self.reward(), self.terminated, self.truncated, self._get_info()
               # observation, reward, terminated, truncated, info
               
    def render(self):
        self._render_frame()
        self.animation_frame()

    def sensor(self):
        deltas = list(itertools.product([-1, 0, 1], repeat=3))
        neighbors = []
        for delta in deltas:
            neighbor = [self.agent_position[i] + delta[i] for i in range(3)]
            neighbors.append(neighbor)

        for i in range(len(neighbors)):
            if any(np.array_equal(neighbors[i], obstacle) for obstacle in self.obstacles):
                self.detected_obstacles.append(neighbors[i])
                
    def _render_frame(self):

        # Extracting X, Y, Z coordinates
        self.x, self.y, self.z = self.transactions[:,0], self.transactions[:,1], self.transactions[:,2]

        # End point coordinates (last point in the matrix)
        self.start_point = self.start_point
        self.end_point_coords = self.end_point

        # Example reward data (arbitrary values)
        self.total_reward = np.cumsum(self.rewards)

        # List of obstacles, one obstacle per step
        self.obstacle_list = self.detected_obstacles  # Create an obstacle for each step

        # Create a figure and 3D axis
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Initialize the plot with empty data
        self.line, = self.ax.plot([], [], [], 'bo-', lw=2)

        # Mark the start and end points
        self.start_point = self.ax.scatter(self.x[0], self.y[0], self.z[0], color='green', s=100, label='Start')
        self.end_point = self.ax.scatter(self.x[-1], self.y[-1], self.z[-1], color='red', s=100, label='End')

        # Add labels to start and end points
        self.ax.text(self.x[0], self.y[0], self.z[0], ' Start', color='green')
        self.ax.text(self.x[-1], self.y[-1], self.z[-1], ' End', color='red')

        # Set up the axes limits
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)
        self.ax.set_zlim(0, self.size)

        # Setting labels
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')

        # Initialize the obstacle plot (empty at the start)
        self.obstacle_plot = self.ax.scatter([], [], [], color='black', s=50, label='Obstacles')

        # Function to update the plot
    
    def update_frame(self, num, x, y, z, line, end_point_coords, total_reward, obstacle_list):
        line.set_data(x[:num+1], y[:num+1])
        line.set_3d_properties(z[:num+1])
        
        # Calculate the distance from the current point to the end point
        current_point_coords = np.array([x[num], y[num], z[num]])
        distance_to_end = np.linalg.norm(current_point_coords - end_point_coords)
        
        # Update obstacles based on the current step
        if num < len(obstacle_list):
            current_obstacles = np.array(obstacle_list[num])  # Convert to a NumPy array for easier indexing
            if current_obstacles.size > 0:  # Ensure there are obstacles to plot
                obs_x = current_obstacles[:, 0]
                obs_y = current_obstacles[:, 1]
                obs_z = current_obstacles[:, 2]
                self.obstacle_plot._offsets3d = (obs_x, obs_y, obs_z)
            else:
                self.obstacle_plot._offsets3d = ([], [], [])  # No obstacles to show
        
        # Update information texts slightly outside the plot area
        self.ax.text2D(1.05, 0.05, f'Distance to End Point: {distance_to_end:.2f}', 
                transform=self.ax.transAxes, fontsize=10, ha='right', bbox=dict(facecolor='white', edgecolor='black'))
        self.ax.text2D(1.05, 0.02, f'Step: {num + 1}', 
                transform=self.ax.transAxes, fontsize=10, ha='right', bbox=dict(facecolor='white', edgecolor='black'))
        
        # Update the total reward in the left-top corner slightly outside the plot area
        self.ax.text2D(-0.05, 1.05, f'Total Reward: {total_reward[num]}', 
                transform=self.ax.transAxes, fontsize=12, ha='left', bbox=dict(facecolor='white', edgecolor='black'))
        
        if num == len(x) - 1:
            self.ani.event_source.stop()  # Stop the animation when finished
        
        return line, self.obstacle_plot

    def animation_frame(self):
        # Bind the key press event to the on_key function

        interval = 1000/self.metadata["render_fps"]
        # Create an animation object
        self.ani = FuncAnimation(self.fig, self.update_frame, frames=len(self.x), fargs=(self.x, self.y, self.z, self.line, self.end_point_coords, self.total_reward, self.obstacle_list),
                            interval=interval, blit=False) # interval: 500 ms for each frame

        # Show the plot
        plt.legend()
        plt.show()
