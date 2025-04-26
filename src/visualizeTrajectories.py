import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def init():
    with open("trajectories/closed_loop_maze_tests_trajectories.json", "r") as f:
        data = json.load(f)
        return data
    return None
    
def plot_trajectories(data):
    """
    Plot the trajectories from the dataset.
    
    Args:
        data (list): List of dictionaries containing episode data.
    """
    # Create a figure and axis
    fig, ax = plt.subplots()
    
    # Iterate through each episode in the dataset
    for episode in data:
        # Extract the data for each episode
        observations = episode["observations"]
        actions = episode["actions"]
        rewards = episode["rewards"]

        x = [obs[0] for obs in observations]
        y = [obs[1] for obs in observations]
        plt.plot(x, y, marker = 'o', label="Trajectory")
        plt.scatter(x[0], y[0], color='green', label="Start")
        plt.scatter(x[-1], y[-1], color='red', label="End")
        plt.title('Maze2D Closed Loop Trajectory')
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.grid()
        plt.axis('equal')  # to avoid distortion
        plt.show()

def animate_trajectories(data):
    fix, ax = plt.subplots()
    episodes_x = []
    episodes_y = []

    for episode in data:
        observations = episode["observations"]
        x = [obs[0] for obs in observations]
        y = [obs[1] for obs in observations]
        episodes_x.append(x)
        episodes_y.append(y)

    all_x = [xi for x in episodes_x for xi in x]
    all_y = [yi for y in episodes_y for yi in y]

    ax.set_xlim(min(all_x), max(all_x)) 
    ax.set_ylim(min(all_y), max(all_y))
    ax.set_title('Maze2D Closed Loop Trajectory')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')

if __name__ == "__main__":
    data = init()
    if data is not None:
        plot_trajectories(data)
    else:
        print("No data found.")