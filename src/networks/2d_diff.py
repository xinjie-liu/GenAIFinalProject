import torch
import torch.nn as nn

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
    