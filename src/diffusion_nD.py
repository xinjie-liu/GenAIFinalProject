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


from networks.unets import TemporalUnet
from src.utils.logger import Logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import hydra
from omegaconf import open_dict

from src.solvers import *

def cycle(dl):
    while True:
        for data in dl:
            yield data

@hydra.main(version_base=None, config_path="configs/", config_name="maze_2d_large_dense.yaml")
def train(args:DictConfig):
    
    if 'mujoco' in args.env_name:
        from src.datasets.cheetah_sequence_dataset import SequenceDataset
        
        dataset = SequenceDataset(
            env = args.env_name,
            max_n_episodes=args.max_n_episodes,
            horizon=args.horizon
        )
    else:
        from src.datasets.sequence_dataset import SequenceDataset
        
        dataset = SequenceDataset(
            env = args.env_name,
            max_n_episodes=args.max_n_episodes,
            horizon=args.horizon
        )
    
    
    with open_dict(args):
        args.horizon = dataset.horizon
        args.observation_dim = dataset.observation_dim
        args.action_dim = dataset.action_dim
    
  
    # noise = torch.normal(torch.zeros(dataset_shape), std=torch.tensor(1.0))

    dataloader = cycle(torch.utils.data.DataLoader(
            dataset, batch_size=args.train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
    dataloader_vis = cycle(torch.utils.data.DataLoader(
            dataset, batch_size=args.eval_batch, num_workers=0, shuffle=True, pin_memory=True
        ))

    diffuser = DiffusionPolicy(args)
    diffuser.load_model()
    diffuser.train(dataset, dataloader, dataloader_vis)


if __name__ == "__main__":

    train()
    # # Training setup
    # namespace = {
    #     'epochs': 100,
    #     'steps_per_epoch': 1000,
    #     'train_batch_size': 1024,
    #     'env_name': 'D4RL/pointmaze/large-dense-v2',
    #     'algorithm': 'DDPM',
    #     'tag': 'GN',
    #     'seed': 0,
    #     'eval_batch':256,
    #     'max_n_episodes': 10000
    # }
    

    # args = DictConfig(namespace)

    
    # # Create dataset and dataloader
    # # _, _, _, _, dataset_, dataset_statistics = generate_diffusion_policy_dataset()
    # dataset = SequenceDataset(
    #     env = args.env_name,
    #     max_n_episodes=args.max_n_episodes
    # )
    # # noise = torch.normal(torch.zeros(dataset_shape), std=torch.tensor(1.0))

    # dataloader = cycle(torch.utils.data.DataLoader(
    #         dataset, batch_size=args.train_batch_size, num_workers=1, shuffle=True, pin_memory=True
    #     ))
    # dataloader_vis = cycle(torch.utils.data.DataLoader(
    #         dataset, batch_size=args.eval_batch, num_workers=0, shuffle=True, pin_memory=True
    #     ))

    # # TODO: 
    # # 1. modify model architecture + forward passfor trajectory data (Done)
    # # 2. normalize trajectory data (Add post processor?)
    # # 3. modify training loop (Training Loop stays the same)

    # # Initialize model, optimizer and scheduler
    # net = TemporalUnet(
    #     horizon=dataset.horizon,
    #     transition_dim=dataset.observation_dim + dataset.action_dim,
    #     cond_dim=4,
    #     device= torch.device('cuda:0')
    # )
    
    # net.to(torch.device('cuda:0'))
    # net.train()
    # scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
    # scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear",  clip_sample=False)
    # optimizer = optim.Adam(net.parameters(), lr=1e-3)  # Optimizer with learning rate

    # # Train the model
    # train(args, net, scheduler, dataset, dataloader, dataloader_vis)         
    
    # # Generate samples from random noise after training
    # with torch.no_grad():
    #     # samples = torch.normal(torch.zeros(2000,2), std=torch.tensor(1.0))
    #     generated_samples = infer_diffusion(net, samples)
        
    #     # Save a comparison of original data vs generated samples
    #     plt.figure(figsize=(16, 8))
        
    #     # Original data
    #     plt.subplot(1, 2, 1)
        
    #     plt.scatter(original_data[:, 0], original_data[:, 1], alpha=0.7, s=5)
    #     plt.title('Original Data Distribution')
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.grid(True, linestyle='--', alpha=0.5)
        
    #     # Generated samples
    #     plt.subplot(1, 2, 2)
    #     plt.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.7, s=5)
    #     plt.title('Diffusion-Generated Samples')
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.grid(True, linestyle='--', alpha=0.5)
        
    #     plt.tight_layout()
    #     plt.savefig('/plots/2DMaze/generation/comparison.png')
    #     plt.close()