import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.helpers import *


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



class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        attention=False,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, horizon=horizon),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, cond, time):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x
    
    
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

