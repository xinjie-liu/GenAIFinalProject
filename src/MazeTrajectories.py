import minari
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_mazedata():
    """
    Generates and processes maze trajectory data from the D4RL pointmaze dataset.
    
    This function:
    1. Downloads and loads the D4RL/pointmaze/large-dense-v2 dataset using minari
    2. Extracts observation sequences from each episode
    3. Processes the observations to have consistent format
    4. Converts observations to PyTorch tensors
    5. Pads sequences to have consistent length
    6. Saves the processed data to 'mazedata.pt'
    7. Loads and returns the processed trajectory data
    
    Returns:
        torch.Tensor: Padded tensor of maze trajectories with shape [num_episodes, max_seq_length, feature_dim]
    """
    # Need to run 'minari download D4RL/pointmaze/large-dense-v2'
    dataset = minari.load_dataset('D4RL/pointmaze/large-dense-v2', download=True)
    env  = dataset.recover_environment()

    # Extract all episodes
    obs_list, action_list = [], []
    for episode in dataset:
        obs_list.append(episode.observations)   # States
        action_list.append(episode.actions)     # Actions

    obs_list_fixed = []
    for i in range(len(obs_list)):  # Process each episode separately
        obs_seq = np.array([obs_list[i]['observation']], dtype=np.float32)
        obs_list_fixed.append(np.squeeze(obs_seq, axis=0))
        #print(np.squeeze(obs_seq, axis=0).shape)

    # Check if the shapes are consistent
    #for seq in obs_list_fixed:
    #    print(seq.shape)  

    from torch.nn.utils.rnn import pad_sequence

    # Convert observations into tensors (this assumes each observation has shape (X, 4))
    obs_tensors = [torch.tensor(np.array(obs_seq, dtype=np.float32), dtype=torch.float32) for obs_seq in obs_list_fixed]

    # Pad sequences to have consistent length (maximum length of sequences)
    padded_obs_tensor = pad_sequence(obs_tensors, batch_first=True, padding_value=0)  # Padding value can be adjusted

    print(padded_obs_tensor.shape)


    torch.save(padded_obs_tensor, 'mazedata.pt')

    traj_data = torch.load('mazedata.pt')

    print(traj_data.shape)

    #print 1 trajectory - note there is a 0 buffer at the end
    print(traj_data[0])

    return traj_data

if __name__ == "__main__":

    traj_data = generate_mazedata()

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