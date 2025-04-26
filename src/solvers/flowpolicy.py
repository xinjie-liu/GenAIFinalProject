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
from  src.networks.unets import TemporalUnet
from src.networks.attention import DiT
from src.utils.logger import Logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import os 



class RectFlowScheduler:
    def __init__(self, num_train_steps=1000, 
                 clip_sample=False,
                 prediction_type='epsilon'):
        self.num_train_steps = num_train_steps
        self.clip_sample = clip_sample
        self.prediction_type = prediction_type

        
        timesteps = np.linspace(1, num_train_steps, num_train_steps)
        self.sigmas = timesteps/ num_train_steps
        self.sigmas = torch.tensor(self.sigmas, dtype=torch.float32).to(torch.device('cuda:0'))
        print(self.sigmas.shape)
    def add_noise(
        self,
        sample, 
        noise,
        timestep
    ):
        sigmas = self.sigmas[timestep].unsqueeze(-1).unsqueeze(-1)
        noised_sample = sample * (1 - sigmas) + noise * sigmas
        return noised_sample
        
    def step(
        self,
        flow_pred, 
        timestep, 
        sample
    ):
        """
        Step the scheduler to get previous (less noisy) sample.
        
        Args:
            noise_pred: Predicted noise from the model
            timestep: Current timestep
            sample: Current noisy sample
            
        Returns:
            Denoised sample at t-1
        """
        if timestep == 0:
            prev_sample = sample + flow_pred * self.sigmas[timestep]
        else:
            prev_sample = sample + flow_pred * (self.sigmas[timestep] - self.sigmas[timestep - 1])
        return {'prev_sample': prev_sample}
    
    
    
    
class FlowPolicy:
    
    def __init__(self, args, **kwargs):
        
        

        # noise = torch.normal(torch.zeros(dataset_shape), std=torch.tensor(1.0))


        # TODO: Generalize network setting for other networks 
        if args.network == 'unet':
            self.net = TemporalUnet(
                horizon=args.horizon,
                transition_dim=args.observation_dim + args.action_dim,
                cond_dim=args.observation_dim,
                device= torch.device('cuda:0')
            ) 
        elif args.network == 'dit':
            self.net = DiT(
                horizon=args.horizon,
                transition_dim=args.observation_dim + args.action_dim,
                cond_dim=args.observation_dim,
                device= torch.device('cuda:0')
            )
        # Set network
        
        self.args = args
        
        self.net.to(torch.device('cuda:0'))
        self.net.train()
        scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
        if self.args.predict_epsilon:
            prediction_type = 'epsilon'
        else:
            prediction_type = 'sample'
        if args.scheduler == 'Flow':
            self.scheduler = RectFlowScheduler(
                num_train_steps=args.num_train_steps,
                clip_sample=args.clip_sample,
                prediction_type=prediction_type)
        # elif args.scheduler == 'DDIM':
        #     self.scheduler = DDIMScheduler(num_train_timesteps=args.num_train_steps, 
        #                                    beta_schedule=args.beta_schedule, 
        #                                    clip_sample=args.clip_sample,
        #                                    prediction_type=prediction_type
        #                                    )
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr)  # Optimizer with learning rate
        self.epoch = 0
        self.visualization_step = args.num_train_steps // 10  # Visualize every 100 steps
        self.loss_weight = self.get_loss_weights(self.args.action_weight,
                                                 1.0,
                                                 None
                                                 )
        
        self.args.ckpt_path = args.ckpt_path + '_' + args.tag
        
    def apply_conditioning(self, x, conditions, action_dim):
        for t, val in conditions.items():
            x[:, t, action_dim:] = val.clone()
        return x
    
    def store_checkpoint(self):
        if not os.path.exists(self.args.ckpt_path):
            os.makedirs(self.args.ckpt_path, exist_ok=True)
        checkpoint = {
            'model_state_dict':self.net.state_dict(),
            'optimzer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch
        }
        torch.save(checkpoint, os.path.join(self.args.ckpt_path, 'checkpoint.pth'))  # Security measure [4][5]
        
                
    def load_model(self):
        if not os.path.exists(os.path.join(self.args.ckpt_path, 'checkpoint.pth')):
            print("Checkpoint not found. Training from scratch.")
            self.epoch = 0
            return
        checkpoint = torch.load(os.path.join(self.args.ckpt_path, 'checkpoint.pth'), weights_only=True)  # Security measure [4][5]
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimzer_state_dict'])
        self.epoch = checkpoint['epoch']


    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.args.observation_dim + self.args.action_dim, 
                                 dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.args.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.args.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.args.action_dim] = action_weight
        return loss_weights

        
    def train_diffusion_step(self, model, scheduler, samples, action_dim):
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
            high=self.args.num_train_steps,  # Exclusive upper bound (1001 → 1000 max)
            size=(samples.trajectories.shape[0],)
        )

        # Add noise to samples according to the timesteps
        noised_samples = self.scheduler.add_noise(
            samples.trajectories.clone().cuda(), 
            sampled_noise, 
            time_steps
            )
        noised_samples_cond = self.apply_conditioning(noised_samples, samples.conditions, self.args.action_dim)
        
        # Predict noise using the model
        pred_noise = model(noised_samples_cond.cuda(), time_steps.cuda())
        # Calculate MSE loss between predicted and actual noise
        if self.args.predict_epsilon:
            # If predicting epsilon, compute loss directly on predicted noise
            loss = self.args.loss_factor * (pred_noise + sampled_noise -  samples.trajectories.clone().cuda())**2*self.loss_weight.unsqueeze(0).unsqueeze(0)*cuda()
        else:
            loss = self.args.loss_factor * (pred_noise + sampled_noise - samples.trajectories.clone().cuda())**2*self.loss_weight.unsqueeze(0).unsqueeze(0).cuda()
        return loss

    def train(self, dataset, dataloader, dataloader_viz):
        """
        Train the diffusion model for the defined number of epochs.
        """
        # # Create a directory for training plots if it doesn't exist
        # import os
        # os.makedirs('/plots/2DMaze/training', exist_ok=True)
        
        # List to store loss values for plotting
        logger = Logger(self.args)
        epoch_start = self.epoch
        for ep in tqdm(range(epoch_start, self.args.epochs)): 
            total_loss = 0.  
            inner_loop = tqdm(range(self.args.steps_per_epoch), desc=f"Epoch {ep+1}/{self.args.epochs}", total=self.args.steps_per_epoch)
            self.epoch = ep
            for j in inner_loop:
                
                # Clear previous gradients
                samples = next(iter(dataloader))
                
                self.optimizer.zero_grad()
                # Compute loss from diffusion process
                loss = self.train_diffusion_step(self.net, self.scheduler, samples, self.args.action_dim)
                
                # Backpropagate gradients
                loss.mean().backward()
                
                # Update network parameters
                self.optimizer.step()
                
                total_loss += loss.mean().item()

            
            # Calculate average loss for the epoch
            avg_loss = total_loss / self.args.steps_per_epoch
            logger.log(log = {'loss': avg_loss}, step = ep)
            generated_samples = self.infer_diffusion(self.net, dataloader_viz, dataset, logger, ep)
            self.store_checkpoint()
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

    def infer_diffusion_step(self, sample_t, cond, t, action_dim):
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
        sample_t_cond = self.apply_conditioning(sample_t, cond, action_dim)
        noise = self.net(sample_t_cond.cuda(), t_in.cuda())
        # Step the scheduler to get previous (less noisy) sample
        sample_t1 = self.scheduler.step(noise, t, sample_t)
        return sample_t1['prev_sample']

    def infer_diffusion(self, net, dataloader_vis, dataset, logger, global_step , plot = True):
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
        for step in range(self.args.num_train_steps):
            # Visualize and save intermediate results every 100 steps
        
            if plot and step % self.visualization_step== 0:
                plt.figure(figsize=(10, 10))
                print(f"Step {step}")
                
                # Unnormalize sample for vis
                unnormalize_obs = dataset.normalizer.unnormalize(
                    denoised_samples[:,:,self.args.action_dim:].detach().cpu().numpy().copy(),
                    key='observations'
                )
                
                plt.scatter(unnormalize_obs.reshape((-1,4))[:,0],
                            unnormalize_obs.reshape((-1,4))[:,1], 
                            alpha=0.7, s=2)
                plt.scatter(unnormalize_obs[:,0,0], unnormalize_obs[:,0,1], color='red', s=5, label='Start State')
                plt.scatter(unnormalize_obs[:,-1,0], unnormalize_obs[:,-1,1], color='green', s=5, label='End State')
                plt.title(f'Generated Samples - Step {step}/{self.args.num_train_steps}')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.grid(True, linestyle='--', alpha=0.5)
                
                image = plt.gcf()
                image_log['denoised_step_{}'.format(step)] = image
                plt.close()
            
            # Perform one denoising step
            with torch.no_grad():
                denoised_samples = self.infer_diffusion_step( denoised_samples,
                                                        samples.conditions, self.args.num_train_steps - 1 - step,
                                                        dataset.action_dim)
                

        # Visualize and save final result
        unnormalize_obs = dataset.normalizer.unnormalize(
            denoised_samples[:,:,self.args.action_dim:].detach().cpu().numpy().copy(),
            key='observations'
        )
        
        if plot:
            plt.scatter(unnormalize_obs.reshape((-1,4))[:,0],
                        unnormalize_obs.reshape((-1,4))[:,1], 
                        alpha=0.7, s=2)
            plt.scatter(unnormalize_obs[:,0,0], unnormalize_obs[:,0,1], color='red', s=5, label='Start State')
            plt.scatter(unnormalize_obs[:,-1,0], unnormalize_obs[:,-1,1], color='green', s=5, label='End State')
            plt.title(f'Generated Samples - Step {step}/{self.args.num_train_steps}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True, linestyle='--', alpha=0.5)
            image = plt.gcf()
            image_log['denoised_step_{}'.format(step)] = image
            plt.close()

        
        
        unnormalize_gt = dataset.normalizer.unnormalize(
            samples.trajectories[:,:,self.args.action_dim:].detach().cpu().numpy().copy(),
            key='observations'
        )
        if plot:
            plt.scatter(unnormalize_gt.reshape((-1,4))[:,0],
                        unnormalize_gt.reshape((-1,4))[:,1], 
                        alpha=0.7, s=2)
            plt.scatter(unnormalize_gt[:,0,0], unnormalize_gt[:,0,1], color='red', s=5, label='Start State')
            plt.scatter(unnormalize_gt[:,-1,0], unnormalize_gt[:,-1,1], color='green', s=5, label='End State')
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

    def policy_act(self, condition, sample_shape, action_dim, normalizer):
        
        sampled_noise = torch.normal(torch.zeros(sample_shape), std=torch.tensor(1.0)) 
        denoised_samples = sampled_noise.clone().cuda()
        for step in range(self.args.num_train_steps):
            # Perform one denoising step
            with torch.no_grad():
                print(f"Step {step}")
                denoised_samples = self.infer_diffusion_step(denoised_samples,
                                                        condition, self.args.num_train_steps - 1 - step,
                                                        action_dim)
        unnormalize_obs = normalizer.unnormalize(
            denoised_samples[:,:,self.args.action_dim:].detach().cpu().numpy(),
            key='observations'
        )
        unnormalize_actions = normalizer.unnormalize(
            denoised_samples[:,:,:action_dim].detach().cpu().numpy(),
            key='actions'
        )
        return unnormalize_actions

    def test_policy_act(self, condition, sample_shape, action_dim, normalizer):
        
        sampled_noise = torch.normal(torch.zeros(sample_shape), std=torch.tensor(1.0)) 
        denoised_samples = sampled_noise.clone().cuda()
        for step in range(1000):
            # Perform one denoising step
            with torch.no_grad():
                print(f"Step {step}")
                denoised_samples = self.infer_diffusion_step(denoised_samples,
                                                        condition, 999 - step,
                                                        action_dim)
        unnormalize_obs = normalizer.unnormalize(
            denoised_samples[:,:,self.args.action_dim:].detach().cpu().numpy(),
            key='observations'
        )
        unnormalize_actions = normalizer.unnormalize(
            denoised_samples[:,:,:action_dim].detach().cpu().numpy(),
            key='actions'
        )
        return denoised_samples