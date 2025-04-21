import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

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
    wandb_project_name: str = "mujoco_tests"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    algorithm: str = "diffusion"
    """the algorithm to use"""
    num_runs: int = 10
    """the number of runs for evaluation"""


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


    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(1)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    envs.single_observation_space.dtype = np.float32

    episodic_reward = []
    global_step = 0

    for ii in range(args.num_runs):
        
        obs, _ = envs.reset(seed=args.seed)
        while True:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
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
    
    print(f"Average episodic reward: {np.mean(episodic_reward)}")
    writer.add_scalar("charts/average_reward", np.mean(episodic_reward), global_step)
    envs.close()
    writer.close()




        