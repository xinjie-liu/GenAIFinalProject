import minari
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_diffusion_policy_dataset():
    """
    Generates and processes maze trajectory data for training a diffusion policy.
    
    Returns:
        tuple: (states_tensor, actions_tensor, achieved_goals_tensor, desired_goals_tensor, combined_tensor) 
               for different training approaches
    """
    # Need to run 'minari download D4RL/pointmaze/large-dense-v2'
    dataset = minari.load_dataset('D4RL/pointmaze/umaze-v2', download=True)
    env = dataset.recover_environment()

    # Extract all episodes - both observations AND actions
    obs_list, action_list = [], []
    for episode in dataset:
        obs_list.append(episode.observations)   # States
        action_list.append(episode.actions)     # Actions

    # Process observations
    obs_list_fixed = []
    achieved_goals_list = []
    desired_goals_list = []
    for i in range(len(obs_list)):
        obs_seq = np.array([obs_list[i]['observation']], dtype=np.float32)
        obs_list_fixed.append(np.squeeze(obs_seq, axis=0))
        
        # Extract achieved and desired goals
        achieved_goal_seq = np.array([obs_list[i]['achieved_goal']], dtype=np.float32)
        achieved_goals_list.append(np.squeeze(achieved_goal_seq, axis=0))
        
        desired_goal_seq = np.array([obs_list[i]['desired_goal']], dtype=np.float32)
        desired_goals_list.append(np.squeeze(desired_goal_seq, axis=0))
    
    # Process actions similarly
    action_list_fixed = []
    for i in range(len(action_list)):
        action_list_fixed.append(np.array(action_list[i], dtype=np.float32))
    
    # Convert to tensors
    from torch.nn.utils.rnn import pad_sequence
    
    # Convert observations and actions into tensors
    obs_tensors = [torch.tensor(obs_seq, dtype=torch.float32) for obs_seq in obs_list_fixed]
    action_tensors = [torch.tensor(action_seq, dtype=torch.float32) for action_seq in action_list_fixed]
    achieved_goal_tensors = [torch.tensor(goal_seq, dtype=torch.float32) for goal_seq in achieved_goals_list]
    desired_goal_tensors = [torch.tensor(goal_seq, dtype=torch.float32) for goal_seq in desired_goals_list]
    
    # Ensure actions and observations match in sequence length
    # (typically actions are one less than observations since no action after final state)
    for i in range(len(obs_tensors)):
        if len(obs_tensors[i]) > len(action_tensors[i]):
            # Truncate observations to match actions length
            obs_tensors[i] = obs_tensors[i][:len(action_tensors[i])]
        elif len(obs_tensors[i]) < len(action_tensors[i]):
            # Truncate actions to match observations length
            action_tensors[i] = action_tensors[i][:len(obs_tensors[i])]
    
    # Pad sequences to have consistent length
    padded_obs_tensor = pad_sequence(obs_tensors, batch_first=True, padding_value=0)
    padded_action_tensor = pad_sequence(action_tensors, batch_first=True, padding_value=0)
    
    # Also create combined state-action tensor (for direct trajectory modeling)
    combined_tensors = []
    for i in range(len(obs_tensors)):
        # Concatenate state and action along feature dimension
        combined = torch.cat([obs_tensors[i], action_tensors[i]], dim=1)
        combined_tensors.append(combined)
    
    padded_combined_tensor = pad_sequence(combined_tensors, batch_first=True, padding_value=0)
    
    # Normalize the combined tensor along the first dimension
    # Calculate mean and std for normalization
    combined_mean = padded_combined_tensor.mean(dim=0, keepdim=True)
    combined_std = padded_combined_tensor.std(dim=0, keepdim=True)
    # Avoid division by zero
    combined_std = torch.where(combined_std == 0, torch.ones_like(combined_std), combined_std)
    # Normalize the tensor
    normalized_combined_tensor = (padded_combined_tensor - combined_mean) / combined_std

    dataset_statistics = {
        'combined_mean': combined_mean,
        'combined_std': combined_std
    }
    
    # Save all formats
    torch.save({
        'states': padded_obs_tensor,
        'actions': padded_action_tensor,
        'achieved_goals': achieved_goal_tensors,
        'desired_goals': desired_goal_tensors,
        'combined': padded_combined_tensor
    }, 'maze_diffusion_policy_data.pt')
    
    print(f"States shape: {padded_obs_tensor.shape}")
    print(f"Actions shape: {padded_action_tensor.shape}")
    print(f"Achieved goals: {len(achieved_goal_tensors)} tensors")
    print(f"Desired goals: {len(desired_goal_tensors)} tensors")
    print(f"Combined shape: {padded_combined_tensor.shape}")
    
    return padded_obs_tensor, padded_action_tensor, achieved_goal_tensors, desired_goal_tensors, padded_combined_tensor, dataset_statistics



if __name__ == "__main__":

    traj_data, _, _, _, _ = generate_diffusion_policy_dataset()

    # Plot the positions of the trajectories and save the figure

    # Create directory for maze dataset plots if it doesn't exist
    os.makedirs('plots/MazeDataset', exist_ok=True)

    # Get the positions (x,y coordinates) from the dataset
    positions = traj_data[:, :, :2]  # Extract x,y coordinates from all trajectories

    # Plot a subset of trajectories (plotting all might be too cluttered)
    num_trajectories_to_plot = positions.shape[0]#min(100, positions.shape[0])  # Limit to 100 trajectories

    plt.figure(figsize=(10, 10))

    # Plot each trajectory
    for i in range(num_trajectories_to_plot):
        # Get single trajectory
        traj = positions[i]
        
        # Find where padding starts (when both x and y become 0)
        # This assumes zero padding and that valid trajectories don't have points at exactly (0,0)
        zero_mask = torch.all(traj == 0, dim=1)
        
        # Find first zero padding position
        non_zero_indices = torch.where(~zero_mask)[0]
        if len(non_zero_indices) > 0:
            last_valid_idx = non_zero_indices[-1].item() + 1
            valid_traj = traj[:last_valid_idx]
        else:
            # If the trajectory is all zeros (shouldn't happen), skip it
            continue
        
        # Convert to numpy for plotting
        valid_traj_np = valid_traj.numpy()
        
        # Plot the trajectory
        plt.plot(
            valid_traj_np[:, 0], 
            valid_traj_np[:, 1], 
            '-', 
            linewidth=1, 
            alpha=0.5,
            label=f'Trajectory {i}' if i < 10 else None  # Only label first 10 for clarity
        )
        
        # Mark start and end points
        plt.scatter(valid_traj_np[0, 0], valid_traj_np[0, 1], c='green', s=30, marker='o')
        plt.scatter(valid_traj_np[-1, 0], valid_traj_np[-1, 1], c='red', s=30, marker='x')

    # Add labels and title
    plt.title('Maze2D Trajectories')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show legend for first few trajectories
    if num_trajectories_to_plot > 0:
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05), fontsize='small')

    # Add color explanation
    green_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10)
    red_patch = plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='red', markersize=10)
    plt.legend([green_patch, red_patch], ['Start', 'End'], loc='upper left')

    # Save the figure
    plt.tight_layout()
    plt.savefig('plots/MazeDataset/trajectory_visualization.png', dpi=300)
    plt.close()

    # Create a heatmap to visualize the density of trajectory points
    plt.figure(figsize=(12, 10))

    # Prepare data for heatmap
    # Flatten all valid trajectory points
    all_points = []
    for i in range(positions.shape[0]):
        traj = positions[i]
        zero_mask = torch.all(traj == 0, dim=1)
        non_zero_indices = torch.where(~zero_mask)[0]
        if len(non_zero_indices) > 0:
            last_valid_idx = non_zero_indices[-1].item() + 1
            valid_traj = traj[:last_valid_idx]
            all_points.extend(valid_traj.numpy())

    all_points = np.array(all_points)

    # Create heatmap using histogram2d
    heatmap, xedges, yedges = np.histogram2d(
        all_points[:, 0], 
        all_points[:, 1], 
        bins=50,
        range=[[all_points[:, 0].min(), all_points[:, 0].max()], 
            [all_points[:, 1].min(), all_points[:, 1].max()]]
    )

    # Plot heatmap
    plt.imshow(
        heatmap.T, 
        origin='lower', 
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap='viridis',
        aspect='auto'
    )

    # Add colorbar and labels
    plt.colorbar(label='Point Density')
    plt.title('Trajectory Points Density Heatmap')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')

    # Save the heatmap
    plt.tight_layout()
    plt.savefig('plots/MazeDataset/trajectory_heatmap.png', dpi=300)
    plt.close()

    print(f"Plots saved to plots/MazeDataset/")