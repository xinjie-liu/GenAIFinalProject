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

class SinusoidalEmbedding(nn.Module):
    """
    Positional embedding using sinusoidal functions.
    This helps the model understand the position/value of inputs in a continuous space.
    """
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        # Size of the embedding vector
        self.size = size
        # Scaling factor for the input value
        self.scale = scale

    def forward(self, x: torch.Tensor):
        # Scale the input
        x = x * self.scale
        half_size = self.size // 2
        # Calculate embedding frequencies on a log scale
        emb = torch.log(torch.Tensor([10000.0])) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size))
        # Apply sinusoidal embedding
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size
    
class TrajectoryDenoising(nn.Module):
    """
    Neural network model for denoising/noise prediction of maze trajectories.
    """
    def __init__(self, state_dim=4, hidden_dim=256) -> None:
        super().__init__()
        # Timestep embedding with 128 dimensions
        self.t_embedding = SinusoidalEmbedding(size=128, scale=1.0)
        
        # Create embeddings for each dimension of state
        self.state_embeddings = nn.ModuleList([
            SinusoidalEmbedding(size=32, scale=10.0) 
            for _ in range(state_dim)
        ])
        
        # Neural network with multiple linear layers and GELU activations
        input_dim = 128 + 32 * state_dim  # Timestep embedding + state embeddings
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim)  # Output has same dimensions as input state
        )
        
        # Initialize network weights with normal distribution
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.05)
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x, t) -> None:
        # Embed x-coordinate using first embedding
        x_emb1 = self.x_embeddings_1(x[:,0])
        # Embed y-coordinate using second embedding
        x_emb2 = self.x_embeddings_2(x[:,1])
        # Embed timestep
        t_emb = self.t_embedding(t)
        # Concatenate all embeddings
        x = torch.concat((x_emb1,x_emb2,t_emb), dim = -1)
        # Pass through the network
        return self.network(x)
    

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
    # 1. modify model architecture + forward passfor trajectory data
    # 2. normalize trajectory data
    # 3. modify training loop

    # Initialize model, optimizer and scheduler
    net = TrajectoryDenoising()
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
        plt.savefig('/plots/2DMaze/generation/comparison.png')
        plt.close()