import numpy as np
import random
import matplotlib.pyplot as plt
import time

# Define a 3x3 grid environment
grid_size = 3
Q_table = np.zeros((grid_size, grid_size, 4))  # 4 actions per state

# Reward system: 1 at goal (2,2), 0 otherwise
rewards = np.zeros((grid_size, grid_size))
rewards[2, 2] = 1  # Goal state

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.2  # Exploration rate (ε-greedy)
episodes = 1000  # Training episodes

# Actions: Up, Down, Left, Right
actions = ["up", "down", "left", "right"]

# Function to get next state based on action
def get_next_state(state, action):
    x, y = state
    if action == "up":
        x = max(0, x - 1)  # Move up, but stay in bounds
    elif action == "down":
        x = min(grid_size - 1, x + 1)  # Move down
    elif action == "left":
        y = max(0, y - 1)  # Move left
    elif action == "right":
        y = min(grid_size - 1, y + 1)  # Move right
    return (x, y)

# Train the Q-learning agent
for episode in range(episodes):
    state = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))  # Random start

    while state != (2, 2):  # Until reaching the goal
        # ε-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action_index = np.random.randint(4)  # Explore: random action
        else:
            action_index = np.argmax(Q_table[state])  # Exploit: best action

        action = actions[action_index]
        next_state = get_next_state(state, action)  # Get next position
        reward = rewards[next_state]  # Get reward

        # Q-learning update rule
        best_next_q = np.max(Q_table[next_state])  # Best future Q-value
        Q_table[state][action_index] += alpha * (reward + gamma * best_next_q - Q_table[state][action_index])

        state = next_state  # Move to next state

# Function to visualize agent's movement
def visualize_agent():
    state = (0, 0)  # Start at top-left corner
    path = [state]

    while state != (2, 2):  # Until reaching the goal
        action_index = np.argmax(Q_table[state])  # Choose best action
        action = actions[action_index]
        state = get_next_state(state, action)
        path.append(state)

    # Plot grid and path
    grid = np.zeros((grid_size, grid_size))
    for step in path:
        grid[step] = 0.5  # Mark path

    grid[2, 2] = 1  # Mark goal

    plt.imshow(grid, cmap="coolwarm", origin="upper")
    plt.xticks(range(grid_size))
    plt.yticks(range(grid_size))
    plt.grid(True)
    plt.title("Agent's Path to Goal")
    plt.show()

# Show learned path
visualize_agent()
