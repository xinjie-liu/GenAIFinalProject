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
    
class denoising(nn.Module):
    """
    Neural network model for denoising/noise prediction in diffusion process.
    Takes noisy samples and timesteps as input, predicts the noise component.
    """
    def __init__(self) -> None:
        super().__init__()
        # Timestep embedding with 128 dimensions
        self.t_embedding = SinusoidalEmbedding(size=128, scale=1.0)
        # Spatial embeddings for x and y coordinates
        self.x_embeddings_1 = SinusoidalEmbedding(size=64, scale=20.0)
        self.x_embeddings_2 = SinusoidalEmbedding(size=64, scale=20.0)
        
        # Neural network with multiple linear layers and GELU activations
        self.network = nn.ModuleList([
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 2)  # Output is 2D (x,y) coordinates
        ])
        self.network = nn.Sequential(*self.network)
        
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
        # Print training progress
        print(f"Epoch {ep+1}/{epochs} | Loss: {avg_loss:.4f}")

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

    # Iteratively denoise the samples
    for step in range(1000):
        # Visualize intermediate results every 100 steps
        if step % 100==0:
            plt.figure()
            print("Step {}".format(step))
            plt.scatter(denoised_samples[:,0], denoised_samples[:, 1])
        
        # Perform one denoising step
        denoised_samples = infer_diffusion_step(net, denoised_samples, 999 - step)
    
    # Visualize final result
    plt.figure()
    print("Step {}".format(step))
    plt.scatter(denoised_samples[:,0], denoised_samples[:, 1])

if __name__ == "__main__":
    # Initialize model and test a forward pass
    net = denoising()
    net(torch.Tensor([[3,3]]),torch.tensor([1]))

    # Generate and visualize sample data
    ps = sample(num_points=4096, n=4, range=[-8,8])
    plt.scatter(ps[:, 0], ps[:,1])

    # Create dataset for training
    data = sample(num_points=4096, n=4, range=[-1,1])
    data = torch.tensor(data, dtype=torch.float32)

    # Initialize the noise scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    
    # Visualize noise addition at different timesteps
    sample_noise = torch.normal(torch.zeros(data.shape), std=torch.tensor(1.0))
    sample_noise2 = torch.normal(torch.zeros(data.shape), std=torch.tensor(1.0))
    for t in range(0,1000,200):
        t = torch.tensor(np.array([t]*data.shape[0])).unsqueeze(1)
        noised_samples = scheduler.add_noise(data, sample_noise, t)
        
        plt.figure()
        plt.scatter(noised_samples[:,0], noised_samples[:,1])

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
        infer_diffusion(net, samples)