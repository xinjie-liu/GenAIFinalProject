import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig
from src.datasets.cheetah_sequence_dataset import SequenceDataset
from src.solvers.diffusionpolicy import DiffusionPolicy, cycle

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
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "mujoco_tests"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    env_id: str = "HalfCheetah-v5"
    """the environment id of the task"""
    dataset_name: str = 'mujoco/halfcheetah/medium-v0'
    """the dataset name of the task"""
    algorithm: str = "diffusion"
    """the algorithm to use"""
    num_runs: int = 1
    """the number of runs for evaluation"""
    action_chunk_size: int = 8
    """the number of actions to take at a time"""

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

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

    namespace = {
        'epochs': 100,
        'steps_per_epoch': 1000,
        'train_batch_size': 1024,
        'env_name': args.dataset_name,
        'algorithm': 'DDPM',
        'tag': 'GN',
        'seed': args.seed,
        'eval_batch': 256,
        'max_n_episodes': 5000,
        'scheduler': 'DDPM',
        'beta_schedule':'linear',
        'num_train_steps':1000,
        'lr': 1e-3,
        'clip_sample': False,
        'loss_factor': 0.1,
    }
    diffuser_args = DictConfig(namespace)
    dataset = SequenceDataset(
        env = diffuser_args.env_name,
        max_n_episodes=diffuser_args.max_n_episodes
    )
    diffuser_args.horizon = dataset.horizon
    diffuser_args.observation_dim = dataset.observation_dim
    diffuser_args.action_dim = dataset.action_dim
    dataloader_vis = cycle(torch.utils.data.DataLoader(
            dataset, batch_size=diffuser_args.eval_batch, num_workers=0, shuffle=True, pin_memory=True
        ))

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


    diffuser = DiffusionPolicy(diffuser_args)
    diffuser.net.load_state_dict(torch.load('src/half-cheetah-model'))
    sample_shape = (1,) + dataset[0].trajectories.shape

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(1)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    envs.single_observation_space.dtype = np.float32

    episodic_reward = []
    global_step = 0
    acc_reward = 0

    # Select a random subset of 30 indices from 0 to 500
    random_indices = random.sample(range(501), 30)
    print(f"Selected random indices: {random_indices}")
    
    # Sort the indices for better readability
    sorted_indices = sorted(random_indices)
    print(f"Sorted random indices: {sorted_indices}")

    generated_samples = []
    prediction_error_norm = []

    for ii in sorted_indices:
        print(f"ii: {ii}")
        dataset_cond = {0: torch.tensor(dataset[ii].conditions[0], device=device, dtype=torch.float32)}
        generated_samples.append(diffuser.test_policy_act(dataset_cond, sample_shape, dataset.action_dim, dataset.normalizer))
        prediction_error_norm.append(np.linalg.norm(generated_samples[-1].cpu().numpy() - dataset[ii].trajectories))
    
    # Plot the prediction error norm as a scatter plot
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.scatter(sorted_indices, prediction_error_norm, color='blue', alpha=0.7)
    plt.xlabel('Dataset Index')
    plt.ylabel('Prediction Error Norm')
    plt.title('Prediction Error Norm for Selected Samples')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a trend line
    z = np.polyfit(sorted_indices, prediction_error_norm, 1)
    p = np.poly1d(z)
    plt.plot(sorted_indices, p(sorted_indices), "r--", alpha=0.7)
    
    # Calculate and display average error
    avg_error = np.mean(prediction_error_norm)
    plt.axhline(y=avg_error, color='g', linestyle='--', alpha=0.7)
    plt.text(sorted_indices[-1], avg_error, f'Avg: {avg_error:.4f}', 
             verticalalignment='bottom', horizontalalignment='right')
    
    plt.tight_layout()
    plt.savefig('prediction_error_scatter.png')
    print(f"Scatter plot saved as 'prediction_error_scatter.png'")
    plt.close()