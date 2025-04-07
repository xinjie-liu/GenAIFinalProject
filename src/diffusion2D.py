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


def sample(num_points=1, n=4, range=[-3,3]):
    """
    Generate points in a checkerboard distribution pattern.
    
    Args:
        num_points: Number of points to generate
        n: Defines the grid size (n×n)
        range: Range of values [min, max]
    
    Returns:
        Array of 2D points in checkerboard pattern
    """
    # Start with random points in unit square
    point = npr.uniform(0,1,size=(num_points,2))
    
    # Randomly choose sectors in a checkerboard pattern
    choice = npr.choice(np.arange(2*n*n), size = num_points)
    grid_sector_row = choice // n
    grid_sector_col = grid_sector_row % 2 + 2 * (choice % n)
    
    # Scale and translate points to their respective grid sectors
    scale_x = (range[1] - range[0])/(2*n)
    scale_y = (range[1] - range[0])/(2*n)
    start_x = range[0] + (range[1] - range[0])/(2*n) * grid_sector_row
    start_y = range[0] + (range[1] - range[0])/(2*n) * grid_sector_col
    point[:, 0] = point[:, 0] * scale_x + start_x  
    point[:, 1] = point[:, 1] * scale_y + start_y  
    return point

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

def train():
    """
    Train the diffusion model for the defined number of epochs.
    """
    # Create a directory for training plots if it doesn't exist
    import os
    os.makedirs('plots/training', exist_ok=True)
    
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
    plt.savefig('plots/training/loss_curve.png')
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
    os.makedirs('plots/generation', exist_ok=True)

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
            plt.savefig(f'plots/generation/samples_step_{step}.png')
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
    plt.savefig('plots/generation/samples_final.png')
    plt.close()
    
    return denoised_samples

if __name__ == "__main__":
    # Initialize model and test a forward pass
    net = denoising()
    net(torch.Tensor([[3,3]]),torch.tensor([1]))

    # Generate and visualize sample data
    ps = sample(num_points=4096, n=4, range=[-8,8])
    plt.scatter(ps[:, 0], ps[:,1])
    plt.savefig('plots/sample_data.png')
    plt.close()
    
    plt.figure()

    # Create dataset for training
    data = sample(num_points=4096, n=4, range=[-1,1])
    data = torch.tensor(data, dtype=torch.float32)
    # Plot the dataset
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.7, s=10)
    plt.title('Training Dataset')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('plots/training_data.png')
    plt.close()
    # Re-initialize the plotting package
    plt.figure()

    # Initialize the noise scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear", clip_sample=False)
    
    # Visualize noise addition at different timesteps
    sample_noise = torch.normal(torch.zeros(data.shape), std=torch.tensor(1.0))
    sample_noise2 = torch.normal(torch.zeros(data.shape), std=torch.tensor(1.0))
    for t in [0, 5, 10, 20, 40, 100, 200, 600, 800]:
        t = torch.tensor(np.array([t]*data.shape[0])).unsqueeze(1)
        noised_samples = scheduler.add_noise(data, sample_noise, t)
        
        plt.figure()
        plt.scatter(noised_samples[:,0], noised_samples[:,1])
        plt.title(f'Noised Samples at t={t[0][0]}')
        plt.savefig(f'plots/noised_samples_t{t[0][0]}.png')
        plt.close()
    
    # Re-initialize the plotting package
    plt.figure()

    # Training setup
    epochs = 1500
    batch_size = 256
    
    # Create dataset and dataloader
    dataset = sample(num_points=4096, n=4, range=[-1,1])
    noise = torch.normal(torch.zeros(dataset.shape), std=torch.tensor(1.0))
    dataset = torch.utils.data.TensorDataset(torch.tensor(dataset, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Initialize model, optimizer and scheduler
    net = denoising()
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
        original_data = sample(num_points=2000, n=4, range=[-1,1])
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
        plt.savefig('plots/generation/comparison.png')
        plt.close()