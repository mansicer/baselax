import numpy as np
import haiku as hk
import gym
import jax
import jax.numpy as jnp
from haiku import nets
from typing import List, Callable, Union

def action_space_dim(space: gym.Space) -> int:
    """Returns the dimension of the action space."""
    if isinstance(space, gym.spaces.Box):
        return np.prod(space.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return space.n
    else:
        raise ValueError(f"Unknown action space {space}")

def image_input_preprocess(x):
    return jnp.transpose(x.astype(jnp.float32) / 255., [0, 2, 3, 1])

def mlp_network(hidden_dims: List[int]) -> Callable[[gym.Space], hk.Transformed]:

    class MLPNetwork:
        def __init__(self, action_space: gym.Space):
            self.output_dim = action_space_dim(action_space)
        
        def __call__(self, obs):
            network = hk.Sequential([
                hk.Flatten(),
                nets.MLP([*hidden_dims, self.output_dim])
            ])
            return network(obs)
    
    return lambda action_space: hk.without_apply_rng(hk.transform(MLPNetwork(action_space)))

def cnn_network(
        conv_channels: List[int] = [32, 32], 
        kernel_size: Union[int, List[int]] = 5, 
        stride: Union[int, List[int]] = 2,
        padding: Union[str, List[str]] = "VALID",
        hidden_dims: List[int] = [256],
        preprocess_fn: Callable[[np.ndarray], np.ndarray]=image_input_preprocess,
    ) -> Callable[[gym.Space], hk.Transformed]:
    num_layers = len(conv_channels)
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * num_layers
    if isinstance(stride, int):
        stride = [stride] * num_layers
    if isinstance(padding, str):
        padding = [padding] * num_layers
    
    class CNNNetwork:
        def __init__(self, action_space: gym.Space):
            self.output_dim = action_space_dim(action_space)
        
        def __call__(self, obs):
            x = preprocess_fn(obs)
            for i in range(num_layers):
                x = hk.Conv2D(
                    output_channels=conv_channels[i], 
                    kernel_shape=kernel_size[i], 
                    stride=stride[i], 
                    padding=padding[i],
                )(x)
                x = jax.nn.leaky_relu(x)
            features = hk.Flatten()(x)
            logits = hk.nets.MLP([*hidden_dims, self.output_dim])(features)
            return logits
    
    return lambda action_space: hk.without_apply_rng(hk.transform(CNNNetwork(action_space)))