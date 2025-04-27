import os, sys
import copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.utils import instantiate
from hydra.compose import compose
import json
from src.datasets.sequence_dataset import SequenceDataset
from src.solvers.diffusionpolicy import DiffusionPolicy, cycle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "maze_tests"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    env_id: str = "PointMaze_MediumDense-v3"
    """the environment id of the task"""
    dataset_name: str = 'D4RL/pointmaze/medium-dense-v2'
    """the dataset name of the task"""
    config_name: str = "maze_2d_medium_dense.yaml"
    """the config name of the task"""
    algorithm: str = "diffusion"
    """the algorithm to use"""
    num_runs: int = 10
    """the number of runs for evaluation"""
    action_chunk_size: int = 15
    """the number of actions to take at a time"""
    num_diffusion_segments: int = 1
    """the number of diffusion segments to use"""
    # plan_horizon: int = 64
    # """the number of steps to plan for"""
    episode_length: int = 200
    """the number of steps to run the episode for"""
    plot_diffusion_plan: bool = False
    """whether to plot the diffusion plan"""
    num_diffusion_plans: int = 10
    """the number of diffusion plans to plot"""

def make_env(env_id, seed, idx, capture_video, run_name, max_episode_steps):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=max_episode_steps, continuing_task = False, reset_target = False)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, max_episode_steps=max_episode_steps, continuing_task = False, reset_target = False)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

def is_valid_trajectory(trajectory, maze_map, invalid_counter, xbounds, ybounds):
    xmin, xmax = xbounds[0], xbounds[1]
    ymin, ymax = ybounds[1], ybounds[0]
    
    maze_height = len(maze_map)      # Number of rows (height of the maze)
    maze_width = len(maze_map[0])    # Number of columns (width of the maze)


    wall_positions = set()
    for row in range(maze_height):
        for col in range(maze_width):
            if maze_map[row][col] == 1:
                wall_positions.add((row, col))


    for i in range(len(trajectory)):
        x = trajectory[i][0]
        y = trajectory[i][1]
        # Convert the continuous (x, y) coordinates into grid indices for the maze
        scale_x = (x - xmin) / (xmax - xmin) * (maze_width - 1)
        scale_y = (y - ymin) / (ymax - ymin) * (maze_height - 1)

        # Round the scaled values to get grid indices (since maze_map is a discrete grid)
        col = round(scale_x)  # x-coordinate (column)
        row = round(scale_y)  # y-coordinate (row)

        if (scale_y, scale_x) in wall_positions:

            invalid_counter += 1
            return invalid_counter  # Out of bounds is invalid

        # Ensure the indices are within the bounds of the maze map
        if row < 0 or row >= maze_height or col < 0 or col >= maze_width:
            invalid_counter += 1
            return invalid_counter  # Out of bounds is invalid
    

    return invalid_counter  # If the trajectory passed all checks, it's valid

# Function to plot and save trajectory plan
def plot_trajectory_plan(start_obs, goal, state_plan, run_id, maze_map, xbounds, ybounds):
    """
    Plot and save a figure of the start observation, goal, and state plan.
    
    Args:
        start_obs: The starting observation (position)
        goal: The goal position (if available)
        state_plan: The planned trajectory states
        step: Current global step for filename
    """
    # Create a figure for plotting the plan with an axis object
    fig, ax = plt.subplots(figsize=(10, 8))
    
    maze = np.array(maze_map)

    # Size of each cell
    cell_size = 1

    # Draw the walls
    rows, cols = maze.shape
    for y in range(rows):
        for x in range(cols):
            if maze[y, x] == 1:
                wall = patches.Rectangle((x * cell_size + xbounds[0], (rows - y - 1) * cell_size + ybounds[0]),
                                        width=cell_size, height=cell_size,
                                        facecolor='black')
                ax.add_patch(wall)

    # # Set plot limits and appearance
    # ax.set_xlim(0, cols * cell_size)
    # ax.set_ylim(0, rows * cell_size)
    # ax.set_aspect('equal')

    # Plot state plan (positions only - first two dimensions)
    plt.scatter(state_plan[0, :, 0], state_plan[0, :, 1], s=5, alpha=0.7, label='Planned trajectory', c='blue')
    
    # Plot start position
    plt.scatter(start_obs[0], start_obs[1], s=100, c='green', marker='o', label='Start position')
    
    # Plot goal position if available
    if goal is not None:
        plt.scatter(goal[0], goal[1], s=100, c='red', marker='*', label='Goal position')
    
    # Add labels and legend
    plt.title('Diffusion Policy Trajectory Plan')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    os.makedirs("trajectory_plots", exist_ok=True)
    # plt.savefig(f"trajectory_plots/trajectory_plan_run_{run_id}.png")
    # plt.close()
    return fig

# Function to plot and save trajectory plan
def continue_plot_trajectory_plan(plt_fig, start_obs, goal, state_plan, run_id):
    """
    Plot and save a figure of the start observation, goal, and state plan.
    
    Args:
        start_obs: The starting observation (position)
        goal: The goal position (if available)
        state_plan: The planned trajectory states
        step: Current global step for filename
    """
    # Create a figure for plotting the plan
    if plt_fig is None:
        plt_fig = plt.figure(figsize=(10, 8))
    
    # Plot state plan (positions only - first two dimensions)
    plt_fig.gca().scatter(state_plan[0, :, 0], state_plan[0, :, 1], s=5, alpha=0.7, c='blue')
    
    # plt.close()
    return plt_fig

def plot_closed_loop_trajectory(plt_fig, observations, run_id):
    """
    Plot the actual closed loop trajectory on top of the planned trajectory and save the figure.
    
    Args:
        plt_fig: The matplotlib figure with the planned trajectory
        observations: Array of actual observations from the environment
        run_id: Run identifier for filename
    """
    if plt_fig is None:
        plt_fig = plt.figure(figsize=(10, 8))
    
    # Get the first and last positions for clearer visualization
    start_pos = observations[0][:2]  # First two dimensions are positions
    end_pos = observations[-1][:2]
    
    # Plot the actual trajectory with both a line for clarity and dots for comparison
    plt_fig.gca().plot(observations[:, 0], observations[:, 1], 'r-', linewidth=1, alpha=0.5)
    plt_fig.gca().scatter(observations[:, 0], observations[:, 1], s=10, c='red', alpha=0.7, label='Actual trajectory')
    
    # Highlight start and end of actual trajectory
    # plt_fig.gca().scatter(start_pos[0], start_pos[1], s=100, c='darkgreen', marker='o', label='Actual start')
    plt_fig.gca().scatter(end_pos[0], end_pos[1], s=100, c='darkred', marker='x', label='Actual end')
    
    # Update the title and legend
    plt_fig.gca().set_title('Planned vs Actual Trajectory')
    plt_fig.gca().legend()
    
    # Save the combined figure
    os.makedirs("trajectory_plots", exist_ok=True)
    plt_fig.savefig(f"trajectory_plots/combined_trajectory_run_{run_id}.png")
    # plt_fig.close()

if __name__ == "__main__":

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Load configuration from YAML file using Hydra
    hydra.initialize(config_path="../configs")
    diffuser_args = compose(config_name=args.config_name)
    
    # Rest of your existing configuration can be adjusted or merged with loaded config
    # You can override specific values from the loaded config if needed
    diffuser_args.env_name = args.dataset_name
    diffuser_args.seed = args.seed

    dataset = SequenceDataset(
        env = diffuser_args.env_name,
        max_n_episodes=1000#diffuser_args.max_n_episodes
    )
    
    # Add observation_dim to diffuser_args using open_dict context
    with open_dict(diffuser_args):
        # diffuser_args.horizon = args.plan_horizon # dataset.horizon
        diffuser_args.observation_dim = dataset.observation_dim
        diffuser_args.action_dim = dataset.action_dim
        
    # dataloader_vis = cycle(torch.utils.data.DataLoader(
    #         dataset, batch_size=diffuser_args.eval_batch, num_workers=0, shuffle=True, pin_memory=True
    #     ))

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    diffuser = DiffusionPolicy(diffuser_args, predict_epsilon = False)
    diffuser.load_model()
    sample_shape = (1, diffuser_args.horizon, dataset[0].trajectories.shape[-1])
    # sample_shape = (1,) + dataset[0].trajectories.shape

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, args.episode_length) for i in range(1)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    envs.single_observation_space.dtype = np.float32

    episodic_reward = []
    
    global_step = 0

    invalid_counter = 0

    trajectories_across_episodes = []

    if 'umaze' in args.dataset_name: 
        maze_map =  [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 1, 1, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]]
        xbound = (-2.5,2.5)
        ybound = (-2.5,2.5)
    
    if 'medium' in args.dataset_name: 
        maze_map = [[1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 0, 0, 0, 1], [1, 1, 0, 0, 0, 1, 1, 1], [1, 0, 0, 1, 0, 0, 0, 1], [1, 0, 1, 0, 0, 1, 0, 1], [1, 0, 0, 0, 1, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1]]
        xbound = (-4,4)
        ybound = (-4,4)

    if 'large' in args.dataset_name: 
        maze_map = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1], [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1], [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        xbound = (-6, 6)
        ybound = (-4.5, 4.5)

    for ii in range(args.num_runs):
        obs, _ = envs.reset(seed=args.seed)
        step_along_diffusion_plan = 0
        diffusion_plan = None
        start_obs = copy.deepcopy(obs["observation"].squeeze(0))
        goal = copy.deepcopy(obs["desired_goal"].squeeze(0))

        episode_data = {    #Episode data to be logged for visualizations
            "observations": [],
            "actions": [],
            "rewards": []
        }
        episode_step = 0
        plt_fig = None
        while True:
            obs_tensor = torch.tensor(obs["observation"], device=device, dtype=torch.float32)
            obs_norm = dataset.normalizer.normalize(obs_tensor.cpu(), key='observations')
            conditions = {0: obs_norm.to(device)}

            if step_along_diffusion_plan % args.action_chunk_size == 0:
                diffusion_plan, state_plan, normalize_actions, normalize_obs = diffuser.policy_act(conditions, sample_shape, dataset.action_dim, dataset.normalizer)
                print("normalize_obs", normalize_obs[:, -1, :])
                conditions = {0: normalize_obs[:, -1, :]}
                if args.num_diffusion_segments > 2:
                    for seg_ii in range(args.num_diffusion_segments - 2):
                        print("conditions", conditions)
                        diffusion_plan_, state_plan_, normalize_actions, normalize_obs = diffuser.policy_act(conditions, sample_shape, dataset.action_dim, dataset.normalizer)
                        diffusion_plan = np.concatenate((diffusion_plan, diffusion_plan_), axis=1)
                        state_plan = np.concatenate((state_plan, state_plan_), axis=1)
                        print("normalize_obs", normalize_obs[:, -1, :])
                        conditions = {0: normalize_obs[:, -1, :]}
                if args.num_diffusion_segments > 1:
                    goal_norm = dataset.normalizer.normalize(np.concatenate((goal, np.array([0.1, 0.1])), axis=0), key='observations')
                    print("conditions", conditions)
                    diffusion_plan_, state_plan_, normalize_actions, normalize_obs = diffuser.policy_act_final(conditions, sample_shape, dataset.action_dim, dataset.normalizer, goal = torch.tensor(goal_norm[:2], device=device))
                    diffusion_plan = np.concatenate((diffusion_plan, diffusion_plan_), axis=1)
                    state_plan = np.concatenate((state_plan, state_plan_), axis=1)
                
                if episode_step == 0:
                    plt_fig = plot_trajectory_plan(start_obs, goal, state_plan, ii, maze_map, xbound, ybound)
                else:
                    plt_fig = continue_plot_trajectory_plan(plt_fig, start_obs, goal, state_plan, ii)

                if args.plot_diffusion_plan:
                    diffusion_plans = []
                    for i in range(args.num_diffusion_plans):
                        diffusion_plans.append(diffuser.policy_act(conditions, sample_shape, dataset.action_dim, dataset.normalizer)[0])
                    # TODO: @Harsif: plot diffusion plans and render a video
                step_along_diffusion_plan = 0
                actions = diffusion_plan[0, 0, :][None, :]
                step_along_diffusion_plan += 1
            else:
                actions = diffusion_plan[0, step_along_diffusion_plan, :][None, :]
                step_along_diffusion_plan += 1

            # random actions
            # actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            episode_step += 1
            episode_data["observations"].append(obs["observation"].squeeze(0).tolist())
            episode_data["actions"].append(actions.squeeze(0).tolist())
            episode_data["rewards"].append(float(rewards))
            # Added this to handle both termination and truncation
            ds = np.logical_or(terminations, truncations)
            if ds.any():
                assert len(infos['episode']['r']) == 1, "only one environment is supported"
                print(f"global_step={global_step}, episodic_return={infos['episode']['r'][0]}")
                writer.add_scalar("charts/episodic_return", infos['episode']['r'][0], global_step)
                writer.add_scalar("charts/episodic_length", infos['episode']['l'][0], global_step)
                episodic_reward.append(infos['episode']['r'][0])
                break
            global_step += 1

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
        
        invalid_counter = is_valid_trajectory(episode_data['observations'], maze_map, invalid_counter, xbound, ybound)
        
        trajectories_across_episodes.append(episode_data)
        plot_closed_loop_trajectory(plt_fig, np.stack(episode_data["observations"]), ii)

    perc_valid_traj = ((args.num_runs - invalid_counter) / (args.num_runs)) * 100
    
    print(f"Average episodic reward: {np.mean(episodic_reward)}")
    print(f"Percent valid trajectories: {perc_valid_traj}")
    writer.add_scalar("charts/average_reward", np.mean(episodic_reward), global_step)
    envs.close()
    writer.close()

    results = {
        "experiment_name": args.exp_name,
        "env_id": args.env_id,
        "seed": args.seed,
        "episode_rewards": episodic_reward,
        "average_reward": float(np.mean(episodic_reward)),  # convert from np.float64
        "Percent_valid_trajectories": perc_valid_traj,
        "timestamp": int(time.time())
    }

    # Create output folder if it doesn't exist
    os.makedirs("results", exist_ok=True)

    with open(f"results/{args.exp_name}_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Create trajectories folder if it doesn't exist
    os.makedirs("trajectories", exist_ok=True)
    with open(f"trajectories/{args.exp_name}_trajectories.json", "w") as f:
        json.dump(trajectories_across_episodes, f, indent=4)



        