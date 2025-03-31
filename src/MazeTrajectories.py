import minari
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# Need to run 'minari download D4RL/pointmaze/large-dense-v2'
dataset = minari.load_dataset('D4RL/pointmaze/large-dense-v2')
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