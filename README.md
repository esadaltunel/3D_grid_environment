# 3D Grid Environment for Reinforcement Learning

## Overview

This project implements a custom 3D grid environment using `gymnasium`, designed for reinforcement learning (RL) tasks. The environment is a cubic grid where an agent navigates from a start point to an end point while avoiding obstacles. The environment is rendered in 3D using Matplotlib, and the agent's movement is visualized step by step.

### Features

- **3D Grid World**: The environment is represented as a 20x20x20 grid.
- **Customizable Obstacles**: Up to 400 obstacles are randomly placed within the grid.
- **Boundary Conditions**: The environment includes checks for boundaries to prevent the agent from moving out of bounds.
- **Reward System**: A negative reward is given for each step, encouraging the agent to find the shortest path. A reward of zero is given when the agent reaches the end point.
- **3D Rendering and Animation**: The environment provides 3D rendering and real-time animation of the agent's movements.
- **Sensor Functionality**: The agent can detect obstacles in its immediate vicinity.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. The required packages include:
   - `gymnasium`
   - `numpy`
   - `matplotlib`

## Environment Description

### State Space

The observation space is a `MultiDiscrete` space representing the agent's position, boundary conditions, and relative position to the end point. The observation vector contains 18 elements:
- Start position (`x, y, z`)
- Current agent position (`x, y, z`)
- Boundary indicators (`up, down, left, right, front, back`)
- Indicators for the agent's position relative to the end point (`up_end, down_end, left_end, right_end, front_end, back_end`)

### Action Space

The action space is a `MultiBinary(3)` space, where each bit represents movement along one of the three axes (x, y, z):
- `0` indicates movement in the negative direction (left/down/back).
- `1` indicates movement in the positive direction (right/up/front).

### Reward Function

- **Step Reward**: A reward of `-1` is given for each step to encourage efficient pathfinding.
- **Goal Reward**: A reward of `0` is given when the agent reaches the end point.

### Episode Termination

An episode terminates when:
- The agent reaches the end point.
- The maximum number of steps (`1000`) is reached.

### Rendering

The environment provides a 3D rendering of the agent's path, obstacles, start, and end points. The rendering is done using `matplotlib`'s 3D plotting capabilities. The agent's movements are animated, showing its progress through the grid.

### Reset Method

The `reset` method initializes the environment, placing the agent at the start point and resetting the step counter, rewards, and detected obstacles. It returns the initial observation and environment information.

### Step Method

The `step` method processes an action, updating the agent's position and checking for episode termination conditions. It returns the new observation, reward, and episode status.

### Sensor Functionality

The `sensor` method allows the agent to detect obstacles in its immediate vicinity. It checks all neighboring cells for obstacles and updates the list of detected obstacles.

### Rendering Methods

- `_render_frame`: Prepares the 3D plot, including the start and end points, and the agent's path.
- `update_frame`: Updates the plot during the animation, including the agent's position, obstacles, and distance to the end point.
- `animation_frame`: Sets up and runs the animation of the agent's movement.

## Usage

1. **Initialize the Environment**:
   ```python
   env = Env(render_mode="human", size=20, num_obstacles=400)
   ```

2. **Reset the Environment**:
   ```python
   observation, info = env.reset()
   ```

3. **Take a Step**:
   ```python
   action = env.action_space.sample()  # Example action
   observation, reward, terminated, truncated, info = env.step(action)
   ```

4. **Render the Environment**:
   ```python
   env.render()
   ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Warning !

If you try to run env_render_test.py file it can be render diffent test for first time. If you quite the rendering it will render according to current test variables.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## Contact

For any inquiries or issues, please contact "Omer Esad Altunel" at "omeresadaltunel@gmail.com".

---

This README provides an overview of the environment, instructions for installation and usage, and details about the environment's features and methods. Make sure to customize the repository URL, your name, and contact details in the appropriate sections.
