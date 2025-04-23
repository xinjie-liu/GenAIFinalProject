import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
import torch.optim as optim
from datasets.cheetah_sequence_dataset import SequenceDataset
from networks.unets import TemporalUnet
from utils.logger import Logger
from omegaconf import DictConfig
from tqdm import tqdm

import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cycle(dl):
    while True:
        for data in dl:
            yield data

def apply_conditioning(x, conditions, action_dim):
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone().to(x.device)
    return x

def train_diffusion_step(model, scheduler, samples, action_dim, diffusion_steps=1000, loss_factor=0.1):
    sampled_noise = torch.normal(torch.zeros(samples.trajectories.shape), std=torch.tensor(1.0)).to(device)
    time_steps = torch.randint(low=0, high=1000, size=(samples.trajectories.shape[0],)).to(device)
    noised_samples = scheduler.add_noise(samples.trajectories.clone().to(device), sampled_noise, time_steps)
    noised_samples_cond = apply_conditioning(noised_samples, samples.conditions, action_dim)
    pred_noise = model(noised_samples_cond, time_steps)
    return loss_factor * ((pred_noise - sampled_noise) ** 2).mean()

def train(args, net, scheduler, dataset, dataloader, dataloader_viz):
    logger = Logger(args)
    for ep in tqdm(range(args.epochs)):
        total_loss = 0.0
        inner_loop = tqdm(range(args.steps_per_epoch), desc=f"Epoch {ep+1}/{args.epochs}", total=args.steps_per_epoch)
        for _ in inner_loop:
            samples = next(iter(dataloader))
            optimizer.zero_grad()
            loss = train_diffusion_step(net, scheduler, samples, dataset.action_dim)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / args.steps_per_epoch
        logger.log(log={'loss': avg_loss}, step=ep)
        generated_samples = infer_diffusion(net, dataloader_viz, dataset, logger, ep)
    torch.save(net.state_dict(), 'half-cheetah-model')

def infer_diffusion_step(net, sample_t, cond, t, action_dim):
    t_in = torch.tensor([t]*sample_t.shape[0], dtype=torch.float32).to(device)
    print(sample_t.shape[0])
    sample_t = sample_t.to(device)
    sample_t_cond = apply_conditioning(sample_t, cond, action_dim)
    noise = net(sample_t_cond, t_in)
    sample_t1 = scheduler.step(noise, t, sample_t)
    return sample_t1['prev_sample']

def infer_diffusion(net, dataloader_vis, dataset, logger, global_step):
    samples = next(iter(dataloader_vis))
    sampled_noise = torch.normal(torch.zeros(samples.trajectories.shape), std=torch.tensor(1.0)).to(device)
    denoised_samples = sampled_noise.clone()
    image_log = {}
    for step in range(1000):
        if step % 100 == 0:
            plt.figure(figsize=(10, 10))
            print(f"Step {step}")
            unnormalize_obs = dataset.normalizer.unnormalize(
                denoised_samples[:,:,dataset.action_dim:].detach().cpu().numpy(),
                key='observations'
            )
            plt.scatter(unnormalize_obs.reshape((-1,4))[:,0], unnormalize_obs.reshape((-1,4))[:,1], alpha=0.7, s=2)
            plt.scatter(unnormalize_obs[:,0,0], unnormalize_obs[:,0,1], color='red', s=5, label='Start State')
            plt.title(f'Generated Samples - Step {step}/1000')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True, linestyle='--', alpha=0.5)
            image_log[f'denoised_step_{step}'] = plt.gcf()
            plt.close()
        with torch.no_grad():
            denoised_samples = infer_diffusion_step(net, denoised_samples, samples.conditions, 999 - step, dataset.action_dim)
    unnormalize_obs = dataset.normalizer.unnormalize(
        denoised_samples[:,:,dataset.action_dim:].detach().cpu().numpy(),
        key='observations'
    )
    plt.scatter(unnormalize_obs.reshape((-1,4))[:,0], unnormalize_obs.reshape((-1,4))[:,1], alpha=0.7, s=2)
    plt.scatter(unnormalize_obs[:,0,0], unnormalize_obs[:,0,1], color='red', s=5, label='Start State')
    plt.title(f'Generated Samples - Final')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, linestyle='--', alpha=0.5)
    image_log[f'denoised_step_{step}'] = plt.gcf()
    plt.close()
    unnormalize_gt = dataset.normalizer.unnormalize(
        samples.trajectories[:,:,dataset.action_dim:].detach().cpu().numpy(),
        key='observations'
    )
    plt.scatter(unnormalize_gt.reshape((-1,4))[:,0], unnormalize_gt.reshape((-1,4))[:,1], alpha=0.7, s=2)
    plt.scatter(unnormalize_gt[:,0,0], unnormalize_gt[:,0,1], color='red', s=5, label='Start State')
    plt.title(f'Ground Truth')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, linestyle='--', alpha=0.5)
    image_log['ground_truth'] = plt.gcf()
    plt.close()
    logger.log_images(image_log, global_step)




if __name__ == "__main__":

    namespace = {
        'epochs': 100,
        'steps_per_epoch': 1000,
        'train_batch_size': 1024,
        'env_name': 'mujoco/halfcheetah/expert-v0',
        'algorithm': 'DDPM',
        'tag': 'GN',
        'seed': 0,
        'eval_batch': 256,
        'max_n_episodes': 5000
    }
    args = DictConfig(namespace)


    
    dataset = SequenceDataset(env=args.env_name, max_n_episodes=args.max_n_episodes)
    dataloader = cycle(DataLoader(dataset, batch_size=args.train_batch_size, num_workers=1, shuffle=True, pin_memory=True))
    dataloader_vis = cycle(DataLoader(dataset, batch_size=args.eval_batch, num_workers=0, shuffle=True, pin_memory=True))
    net = TemporalUnet(
        horizon=dataset.horizon,
        transition_dim=dataset.observation_dim + dataset.action_dim,
        cond_dim=dataset.observation_dim,
        device=device
    ).to(device)
    net.train()
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear", clip_sample=False)
    #optimizer = optim.Adam(net.parameters(), lr=2e-3)

    optimizer = optim.Adam(net.parameters(), lr=1e-3)


    train(args, net, scheduler, dataset, dataloader, dataloader_vis)
