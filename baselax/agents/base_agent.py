import gym
import haiku
import jax
import jax.numpy as jnp
import optax

from typing import Callable, Mapping, Tuple, Union
from abc import ABC, abstractmethod
from collections import namedtuple


class BaseAgent(ABC):
    """An abstract BaseAgent class.

    Args:
        network (Callable[[gym.Space], haiku.Transformed]): A network function creator that takes the action space as input and return the network function.
        env (gym.Env): a gym environment wrapper with batch input and output.
        learning_rate (Union[float, optax.Schedule]): a learning rate or learning rate schedule for the optimizer.
    """

    Params = namedtuple("Params", ["policy"])
    PolicyState = namedtuple("PolicyState", ["count"])
    OptimState = namedtuple("OptimState", ["count", "opt_state"])
    PolicyOutput = namedtuple("PolicyOutput", ["actions"])
    OptimOutput = namedtuple("OptimOutput", ["loss"])

    def __init__(
        self, 
        network: Callable[[gym.Space], haiku.Transformed], 
        env: gym.Env, 
        learning_rate: Union[float, optax.Schedule]
    ):
        self._network = network(env.action_space)
        self._observation_space = env.observation_space
        self._action_space = env.action_space
        self._optimizer = optax.adam(learning_rate)

    def init_params(self, rng: haiku.PRNGSequence) -> Params:
        """Initialize the parameters for the agent

        Args:
            rng (haiku.PRNGSequence): the random number generator.

        Returns:
            BaseAgentParams: the initialized parameters.
        """
        sample_input = self._observation_space.sample()
        params = self._network.init(rng, sample_input)
        return self.Params(policy=params)

    def init_policy(self, rng: haiku.PRNGSequence) -> PolicyState:
        """Initialize the policy state.

        Args:
            rng (haiku.PRNGSequence): the random number generator.

        Returns:
            BaseAgentPolicyState: the initialized policy state.
        """
        return self.PolicyState(count=0)

    def init_optimizer(self, params: Params, rng: haiku.PRNGSequence) -> OptimState:
        """Initialize the optimizer state.

        Args:
            params (BaseAgentParams): the agent parameters.
            rng (haiku.PRNGSequence): the random number generator.

        Returns:
            BaseAgentOptimState: the initialized optimizer state.
        """
        opt_state = self._optimizer.init(params.policy)
        return self.OptimState(count=0, opt_state=opt_state)

    @abstractmethod
    def predict(
        self, 
        params: Params, 
        policy_state: PolicyState, 
        obs: jnp.DeviceArray, 
        key: haiku.PRNGSequence, 
        evaluation: bool, 
        **kwargs
    ) -> Tuple[PolicyOutput, PolicyState]:
        """Select actions for the agent with given observations.

        Args:
            params (BaseAgentParams): the agent parameters.
            policy_state (BaseAgentPolicyState): the policy state.
            obs (jnp.DeviceArray): observations from the environment.
            key (haiku.PRNGSequence): the random number generator that can be used for exploration.
            evaluation (bool): whether to evaluate the policy or not.

        Returns:
            Tuple[BaseAgentPolicyOutput, BaseAgentPolicyState]: return the policy output and the updated policy state.
        """
        raise NotImplementedError()

    @abstractmethod
    def update(
        self, 
        params: Params, 
        optimizer_state: OptimState, 
        data: Mapping[str, jnp.DeviceArray], 
        **kwargs
    ) -> Tuple[OptimOutput, Params, OptimState]:
        """Update the agent policy with given parameters

        Args:
            params (BaseAgentParams): the agent parameters.
            optimizer_state (BaseAgentOptimState): the current optimizer state.
            data (Mapping[str, jnp.DeviceArray]): the training data dict which may includes the observations, actions, rewards, etc.

        Returns:
            Tuple[BaseAgentOptimOutput, BaseAgentParams, BaseAgentOptimState]: return the optimization output, updated agent parameters, and the updated optimizer state.
        """
        raise NotImplementedError()
    
    def jit(self):
        """Jitting agent `predict` and `update` methods for speeding up
        
        Examples:
            agent = MyAgent(...)
            agent.jit()
        """
        self.predict = jax.jit(self.predict)
        self.update = jax.jit(self.update)
