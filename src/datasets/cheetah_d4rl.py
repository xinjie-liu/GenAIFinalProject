import os
import collections
import numpy as np

import pdb
import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import TimeLimit

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)
import minari

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#


def get_dataset_environment(env):
    # dataset = env.get_dataset()
    dataset = minari.load_dataset(env, download=True)
    env = dataset.recover_environment()
    if 'antmaze' in str(env).lower():
        ## the antmaze-v0 environments have a variety of bugs
        ## involving trajectory segmentation, so manually reset
        ## the terminal and timeout fields
        dataset = antmaze_fix_timeouts(dataset)
        dataset = antmaze_scale_rewards(dataset)
        get_max_delta(dataset)

    return dataset

def load_env(dataset):
    return dataset.recover_environment()

def sequence_dataset(env, preprocess_fn):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    dataset = get_dataset_environment(env)
    dataset = preprocess_fn(dataset)

    N = len(dataset)
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts =  True
    episode_step = 0
    # Convert observations and actions into tensors
    
    #Changed for cheetah
    for i in range(N):
        # Get episode
        episode = dataset[i]

        # Termination and truncation info
        done_bool = np.any(episode.terminations)
        final_timestep = np.any(episode.truncations)

        # Core data
        obs_list = np.array(episode.observations)
        action_list = np.array(episode.actions)
        terminals = np.array(episode.terminations)
        timeouts = np.array(episode.truncations)

        # Construct episode dict
        episode_data = {
            'observations': obs_list[:-1],
            'actions': action_list,
            'terminals': terminals,
            'timeouts': timeouts
        }

        # Compute next observations
        next_observations = obs_list[1:]
        episode_data['next_observations'] = next_observations.copy()

        yield episode_data


'''
    for i in range(N):
        done_bool = np.any(dataset[i].terminations)
        final_timestep = np.any(dataset[i].truncations)

        obs_list = dataset[i].observations
        #obs_list = dataset[i].observations['observation']# States
        #ag_list = dataset[i].observations['achieved_goal']
        ag_list = dataset[i].achieved_goal
        #dg_list = dataset[i].observations['desired_goal']
        dg_list = dataset[i].desired_goal
        action_list = np.array(dataset[i].actions)
        episode_data = {
            'observations': np.array(obs_list[:-1]),
            'achieved_goals': np.array(ag_list[:-1]),
            'desired_goals': np.array(dg_list[:-1]),
            'actions':action_list,
            'terminals': dataset[i].terminations,
            'timeouts': dataset[i].truncations
        }

        if 'next_observations' in dataset[i].observations:
            next_observations = dataset[i].observations['next_observations']
        else:
            next_obs_list = obs_list[1:]
            next_observations = np.array(next_obs_list).copy()
        episode_data['next_observations'] = next_observations.copy()
        
        yield episode_data
'''


#-----------------------------------------------------------------------------#
#-------------------------------- maze2d fixes -------------------------------#
#-----------------------------------------------------------------------------#

def process_maze2d_episode(episode):
    '''
        adds in `next_observations` field to episode
    '''
    assert 'next_observations' not in episode
    length = len(episode['observations'])
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode['next_observations'] = next_observations
    return episode