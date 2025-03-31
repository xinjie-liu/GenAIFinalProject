import gymnasium as gym
import gymnasium_robotics
import numpy as np
from minari import DataCollector, StepDataCallback
import minari
import argparse
import os

# === Argument Parser ===
parser = argparse.ArgumentParser(description="Generate Maze2D-style dataset with a direct controller")
parser.add_argument("--maze", type=str, default="PointMaze_Medium-v3", help="Maze environment ID")
parser.add_argument("--dataset-name", type=str, default="custom-maze2d-direct-controller-v1", help="Name for saved Minari dataset")
parser.add_argument("--steps", type=int, default=10000, help="Number of total environment steps")
parser.add_argument("--kp", type=float, default=10.0, help="Proportional gain for controller")
parser.add_argument("--kd", type=float, default=-1.0, help="Derivative gain for controller")
parser.add_argument("--noise", type=float, default=0.2, help="Standard deviation of Gaussian noise added to actions")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing Minari dataset with the same name")
args = parser.parse_args()

# === Custom Controller ===
class DirectController:
    def __init__(self, kp=10.0, kd=-1.0, noise_std=0.2):
        self.kp = kp
        self.kd = kd
        self.noise_std = noise_std

    def compute_action(self, obs):
        pos = obs["observation"][:2]
        vel = obs["observation"][2:]
        goal = obs["desired_goal"]

        action = self.kp * (goal - pos) + self.kd * vel
        action += np.random.normal(scale=self.noise_std, size=action.shape)
        return np.clip(action, -1, 1).astype(np.float32)  # Ensure action matches dtype

# === Step Data Callback ===
class PointMazeStepDataCallback(StepDataCallback):
    def __call__(
        self, env, obs, info, action=None, rew=None, terminated=None, truncated=None
    ):
        qpos = obs["observation"][:2]
        qvel = obs["observation"][2:]
        goal = obs["desired_goal"]

        step_data = super().__call__(env, obs, info, action, rew, terminated, truncated)

        if step_data["info"].get("success", False):
            step_data["truncation"] = True

        step_data["info"]["qpos"] = qpos
        step_data["info"]["qvel"] = qvel
        step_data["info"]["goal"] = goal

        return step_data

# === Main Script ===
gym.register_envs(gymnasium_robotics)

if args.overwrite:
    try:
        minari.delete_dataset(args.dataset_name)
        print(f"⚠️  Existing dataset '{args.dataset_name}' deleted.")
    except Exception as e:
        print(f"Warning: Could not delete dataset (it might not exist): {e}")

env = gym.make(
    args.maze,
    continuing_task=True,
    max_episode_steps=args.steps
)

collector_env = DataCollector(
    env,
    step_data_callback=PointMazeStepDataCallback,
    record_infos=True
)

obs, _ = collector_env.reset(seed=123)
controller = DirectController(
    kp=args.kp,
    kd=args.kd,
    noise_std=args.noise
)

for _ in range(args.steps):
    action = controller.compute_action(obs)
    obs, reward, terminated, truncated, info = collector_env.step(action)

    if terminated or truncated:
        obs, _ = collector_env.reset()

# Ensure dataset_id is valid and versioned
if not args.dataset_name.count("-v"):
    args.dataset_name += "-v1"

# Create dataset
try:
    dataset = collector_env.create_dataset(
        dataset_id=args.dataset_name,
        algorithm_name="DirectController",
        code_permalink="",
        author="Your Name",
        author_email="your@email.com",
    )
    print("✅ Dataset created:", dataset)
except Exception as e:
    print("❌ Failed to create dataset:", e)
