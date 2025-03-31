import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels, timestep_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.mish = nn.Mish()
        self.fc_timestep = nn.Linear(timestep_dim, channels)

    def forward(self, x, t):
        residual = x
        t_emb = self.fc_timestep(t).unsqueeze(-1)
        
        x = self.norm1(self.conv1(x)) + t_emb
        x = self.mish(x)
        x = self.norm2(self.conv2(x))
        x = self.mish(x)
        
        return x + residual

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64, num_blocks=6, timestep_dim=128):
        super().__init__()
        self.initial = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(base_channels, timestep_dim) for _ in range(num_blocks)
        ])
        
        self.final = nn.Conv1d(base_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t):
        x = self.initial(x)
        for block in self.res_blocks:
            x = block(x, t)
        x = self.final(x)
        return x

class ReturnPredictor(nn.Module):
    def __init__(self, in_channels, base_channels=64, num_blocks=3, timestep_dim=128):
        super().__init__()
        self.initial = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(base_channels, timestep_dim) for _ in range(num_blocks)
        ])
        
        self.final_fc = nn.Linear(base_channels, 1)
    
    def forward(self, x, t):
        x = self.initial(x)
        for block in self.res_blocks:
            x = block(x, t)
        x = x.mean(dim=-1)  # Global average pooling
        x = self.final_fc(x)
        return x

