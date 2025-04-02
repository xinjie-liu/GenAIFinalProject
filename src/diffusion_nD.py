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

from MazeTrajectories import generate_diffusion_policy_dataset
from networks.unets import TemporalUnet

    

def train_diffusion_step(model, samples, diffusion_steps=1000, loss_factor = 0.1):
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
    sampled_noise = torch.normal(torch.zeros(samples.shape), std=torch.tensor(1.0))
    # Sample random timesteps
    time_steps = torch.randint(
        low=0,
        high=1000,  # Exclusive upper bound (1001 → 1000 max)
        size=(samples.shape[0],)
    )

    # Add noise to samples according to the timesteps
    noised_samples = scheduler.add_noise(samples.clone(), sampled_noise, time_steps)
    
    # Predict noise using the model
    pred_noise = model(noised_samples, time_steps)
    # Calculate MSE loss between predicted and actual noise
    return loss_factor * (pred_noise - sampled_noise)**2

def train(dataloader):
    """
    Train the diffusion model for the defined number of epochs.
    """
    # # Create a directory for training plots if it doesn't exist
    # import os
    # os.makedirs('/plots/2DMaze/training', exist_ok=True)
    
    # List to store loss values for plotting
    loss_history = []
    
    for ep in range(epochs): 
        total_loss = 0.   
        for _, samples in enumerate(dataloader):
            # Clear previous gradients
            optimizer.zero_grad()
            # Compute loss from diffusion process
            loss = train_diffusion_step(net, samples[0])
            
            # Backpropagate gradients
            loss.mean().backward()
            
            # Update network parameters
            optimizer.step()
            
            total_loss += loss.mean().item()
            
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        # Store loss for plotting
        loss_history.append(avg_loss)
        
        # Print training progress
        print(f"Epoch {ep+1}/{epochs} | Loss: {avg_loss:.4f}")
        
    # Plot and save the training loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, ep+2), loss_history, linestyle='-')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('/plots/2DMaze/training/loss_curve.png')
    plt.close()

def infer_diffusion_step(net, sample_t, t):
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
    noise = net(sample_t, t_in)
    # Step the scheduler to get previous (less noisy) sample
    sample_t1 = scheduler.step(noise, t, sample_t)
    return sample_t1['prev_sample']

def infer_diffusion(net, samples):
    """
    Run the full inference/sampling process from noise to data.
    
    Args:
        net: The trained denoising model
        samples: Initial noise samples
    """
    denoised_samples = samples.clone()

    # Create a directory for generation plots if it doesn't exist
    import os
    os.makedirs('/plots/2DMaze/generation', exist_ok=True)

    # Iteratively denoise the samples
    for step in range(1000):
        # Visualize and save intermediate results every 100 steps
        if step % 100 == 0:
            plt.figure(figsize=(10, 10))
            print(f"Step {step}")
            plt.scatter(denoised_samples[:,0], denoised_samples[:, 1], alpha=0.7, s=5)
            plt.title(f'Generated Samples - Step {step}/1000')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.savefig(f'/plots/2DMaze/generation/samples_step_{step}.png')
            plt.close()
        
        # Perform one denoising step
        denoised_samples = infer_diffusion_step(net, denoised_samples, 999 - step)
    
    # Visualize and save final result
    plt.figure(figsize=(10, 10))
    print(f"Final Step {step}")
    plt.scatter(denoised_samples[:,0], denoised_samples[:, 1], alpha=0.7, s=5)
    plt.title('Final Generated Samples')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('/plots/2DMaze/generation/samples_final.png')
    plt.close()
    
    return denoised_samples

if __name__ == "__main__":

    # Training setup
    epochs = 1500
    batch_size = 256
    
    # Create dataset and dataloader
    _, _, _, _, dataset = generate_diffusion_policy_dataset()
    noise = torch.normal(torch.zeros(dataset.shape), std=torch.tensor(1.0))
    dataset = torch.utils.data.TensorDataset(torch.tensor(dataset, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # TODO: 
    # 1. modify model architecture + forward passfor trajectory data (Done)
    # 2. normalize trajectory data (Add post processor?)
    # 3. modify training loop (Training Loop stays the same)

    # Initialize model, optimizer and scheduler
    net = TemporalUnet(
        horizon=32,
        transition_dim=4,
        cond_dim=4
    )
    net.train()
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    optimizer = optim.Adam(net.parameters(), lr=1e-3)  # Optimizer with learning rate

    # Train the model
    train()         
    
    # Generate samples from random noise after training
    with torch.no_grad():
        samples = torch.normal(torch.zeros(2000,2), std=torch.tensor(1.0))
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