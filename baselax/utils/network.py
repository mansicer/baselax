import numpy as np
import haiku as hk
import gym
from haiku import nets
from typing import List, Callable

def action_space_dim(space: gym.Space) -> int:
    """Returns the dimension of the action space."""
    if isinstance(space, gym.spaces.Box):
        return np.prod(space.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return space.n
    else:
        raise ValueError(f"Unknown action space {space}")

def mlp_network(hidden_dims: List[int], action_space: gym.Space):

    def network(obs):
        network = hk.Sequential([
            hk.Flatten(),
            nets.MLP([*hidden_dims, action_space_dim(action_space)])
        ])
        return network(obs)
    
    return hk.without_apply_rng(hk.transform(network))
