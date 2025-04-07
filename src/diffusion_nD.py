# Import necessary libraries
import torch
import torch.nn as nn
import numpy as np 
import numpy.random as npr
from matplotlib import pyplot as plt

# Import diffusers components for noise scheduling
from diffusers import DDIMScheduler, DDPMScheduler
from torch.utils.data import DataLoader
import torch.optim as optim

import torch.utils
import torch.utils.data

from src.datasets.sequence_dataset import SequenceDataset
from networks.unets import TemporalUnet
from src.utils.logger import Logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

def cycle(dl):
    while True:
        for data in dl:
            yield data


def apply_conditioning(x, conditions, action_dim):
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()
    return x

def train_diffusion_step(model, scheduler, samples, action_dim, diffusion_steps=1000, loss_factor = 0.1):
    """
    Execute a single training step for the diffusion model.
    
    Args:
        model: The denoising model
        samples: Real data samples
        diffusion_steps: Total number of diffusion steps
        loss_factor: Weight for the loss calculation
    
    Returns:
        Loss tensor
    """
    # Generate random noise
    sampled_noise = torch.normal(torch.zeros(samples.trajectories.shape), std=torch.tensor(1.0)).cuda()
    # Sample random timesteps
    time_steps = torch.randint(
        low=0,
        high=1000,  # Exclusive upper bound (1001 → 1000 max)
        size=(samples.trajectories.shape[0],)
    )

    # Add noise to samples according to the timesteps
    noised_samples = scheduler.add_noise(samples.trajectories.clone().cuda(), sampled_noise, time_steps)
    noised_samples_cond = apply_conditioning(noised_samples, samples.conditions, dataset.action_dim)
    
    # Predict noise using the model
    pred_noise = model(noised_samples_cond.cuda(), time_steps.cuda())
    # Calculate MSE loss between predicted and actual noise
    return loss_factor * (pred_noise - sampled_noise)**2

def train(args, net, scheduler, dataset, dataloader, dataloader_viz):
    """
    Train the diffusion model for the defined number of epochs.
    """
    # # Create a directory for training plots if it doesn't exist
    # import os
    # os.makedirs('/plots/2DMaze/training', exist_ok=True)
    
    # List to store loss values for plotting
    logger = Logger(args)
    for ep in tqdm(range(args.epochs)): 
        total_loss = 0.  
        inner_loop = tqdm(range(args.steps_per_epoch), desc=f"Epoch {ep+1}/{args.epochs}", total=args.steps_per_epoch)
        for j in inner_loop:
            
            # Clear previous gradients
            samples = next(iter(dataloader))
            
            optimizer.zero_grad()
            # Compute loss from diffusion process
            loss = train_diffusion_step(net, scheduler, samples, dataset.action_dim)
            
            # Backpropagate gradients
            loss.mean().backward()
            
            # Update network parameters
            optimizer.step()
            
            total_loss += loss.mean().item()

        
        # Calculate average loss for the epoch
        avg_loss = total_loss / args.steps_per_epoch
        logger.log(log = {'loss': avg_loss}, step = ep)
        generated_samples = infer_diffusion(net, dataloader_viz, dataset, logger, ep)

        # Store loss for plotting
        
        # Print training progress
        # print(f"Epoch {ep+1}/{ep} | Loss: {avg_loss:.4f}")
        
    # Plot and save the training loss curve
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, ep+2), loss_history, linestyle='-')
    # plt.title('Training Loss Over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.savefig('/plots/2DMaze/training/loss_curve.png')
    # plt.close()

def infer_diffusion_step(net, sample_t, cond, t, action_dim):
    """
    Perform a single denoising step during inference.
    
    Args:
        net: The trained denoising model
        sample_t: Current noisy samples
        t: Current timestep
        
    Returns:
        Denoised samples at t-1
    """
    # Prepare timestep tensor
    t_in = torch.tensor([t]*sample_t.shape[0], dtype=torch.float32)
    
    # Predict noise
    sample_t_cond = apply_conditioning(sample_t, cond, action_dim)
    noise = net(sample_t_cond.cuda(), t_in.cuda())
    # Step the scheduler to get previous (less noisy) sample
    sample_t1 = scheduler.step(noise, t, sample_t)
    return sample_t1['prev_sample']

def infer_diffusion(net, dataloader_vis, dataset, logger, global_step):
    """
    Run the full inference/sampling process from noise to data.
    
    Args:
        net: The trained denoising model
        samples: Initial noise samples
    """
    
    samples = next(iter(dataloader_vis))
    sampled_noise = torch.normal(torch.zeros(samples.trajectories.shape), std=torch.tensor(1.0)) 
    
    denoised_samples = sampled_noise.clone().cuda()
    # Create a directory for generation plots if it doesn't exist
    # import os
    # os.makedirs('/plots/2DMaze/generation', exist_ok=True)

    # Iteratively denoise the samples
    image_log = {}
    for step in range(1000):
        # Visualize and save intermediate results every 100 steps
        if step % 100 == 0:
            plt.figure(figsize=(10, 10))
            print(f"Step {step}")
            
            # Unnormalize sample for vis
            unnormalize_obs = dataset.normalizer.unnormalize(
                denoised_samples[:,:,dataset.action_dim:].detach().cpu().numpy(),
                key='observations'
            )
            
            plt.scatter(unnormalize_obs.reshape((-1,4))[:,0],
                        unnormalize_obs.reshape((-1,4))[:,1], 
                        alpha=0.7, s=2)
            plt.scatter(unnormalize_obs[:,0,0], unnormalize_obs[:,0,1], color='red', s=5, label='Start State')

            plt.title(f'Generated Samples - Step {step}/1000')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True, linestyle='--', alpha=0.5)
            
            image = plt.gcf()
            image_log['denoised_step_{}'.format(step)] = image
            plt.close()
        
        # Perform one denoising step
        with torch.no_grad():
            denoised_samples = infer_diffusion_step(net, denoised_samples,
                                                    samples.conditions, 999 - step,
                                                    dataset.action_dim)
            

    # Visualize and save final result
    unnormalize_obs = dataset.normalizer.unnormalize(
        denoised_samples[:,:,dataset.action_dim:].detach().cpu().numpy(),
        key='observations'
    )
    
    plt.scatter(unnormalize_obs.reshape((-1,4))[:,0],
                unnormalize_obs.reshape((-1,4))[:,1], 
                alpha=0.7, s=2)
    plt.scatter(unnormalize_obs[:,0,0], unnormalize_obs[:,0,1], color='red', s=5, label='Start State')

    plt.title(f'Generated Samples - Step {step}/1000')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, linestyle='--', alpha=0.5)
    image = plt.gcf()
    image_log['denoised_step_{}'.format(step)] = image
    plt.close()

    
    
    unnormalize_gt = dataset.normalizer.unnormalize(
        samples.trajectories[:,:,dataset.action_dim:].detach().cpu().numpy(),
        key='observations'
    )
    plt.scatter(unnormalize_gt.reshape((-1,4))[:,0],
                unnormalize_gt.reshape((-1,4))[:,1], 
                alpha=0.7, s=2)
    plt.scatter(unnormalize_gt[:,0,0], unnormalize_gt[:,0,1], color='red', s=5, label='Start State')

    plt.title(f'Ground Truth')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, linestyle='--', alpha=0.5)
    image = plt.gcf()
    image_log['ground_truth'] = image
    plt.close()
    
    logger.log_images(image_log, global_step )
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.savefig('/plots/2DMaze/generation/samples_final.png')

if __name__ == "__main__":


    # Training setup
    namespace = {
        'epochs': 10000,
        'steps_per_epoch': 1000,
        'train_batch_size': 1024,
        'env_name': 'D4RL/pointmaze/umaze-v2',
        'algorithm': 'DDPM',
        'tag': 'GN',
        'seed': 0,
        'eval_batch':256,
        'max_n_episodes': 5000
    }
    

    args = DictConfig(namespace)

    
    # Create dataset and dataloader
    # _, _, _, _, dataset_, dataset_statistics = generate_diffusion_policy_dataset()
    dataset = SequenceDataset(
        env = args.env_name,
        max_n_episodes=args.max_n_episodes
    )
    # noise = torch.normal(torch.zeros(dataset_shape), std=torch.tensor(1.0))

    dataloader = cycle(torch.utils.data.DataLoader(
            dataset, batch_size=args.train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
    dataloader_vis = cycle(torch.utils.data.DataLoader(
            dataset, batch_size=args.eval_batch, num_workers=0, shuffle=True, pin_memory=True
        ))

    # TODO: 
    # 1. modify model architecture + forward passfor trajectory data (Done)
    # 2. normalize trajectory data (Add post processor?)
    # 3. modify training loop (Training Loop stays the same)

    # Initialize model, optimizer and scheduler
    net = TemporalUnet(
        horizon=dataset.horizon,
        transition_dim=dataset.observation_dim + dataset.action_dim,
        cond_dim=4,
        device= torch.device('cuda:0')
    )
    
    net.to(torch.device('cuda:0'))
    net.train()
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear",  clip_sample=False)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)  # Optimizer with learning rate

    # Train the model
    train(args, net, scheduler, dataset, dataloader, dataloader_vis)         
    
    # Generate samples from random noise after training
    with torch.no_grad():
        # samples = torch.normal(torch.zeros(2000,2), std=torch.tensor(1.0))
        generated_samples = infer_diffusion(net, samples)
        
        # Save a comparison of original data vs generated samples
        plt.figure(figsize=(16, 8))
        
        # Original data
        plt.subplot(1, 2, 1)
        
        plt.scatter(original_data[:, 0], original_data[:, 1], alpha=0.7, s=5)
        plt.title('Original Data Distribution')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Generated samples
        plt.subplot(1, 2, 2)
        plt.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.7, s=5)
        plt.title('Diffusion-Generated Samples')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('/plots/2DMaze/generation/comparison.png')
        plt.close()