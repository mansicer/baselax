import gym
import haiku
import jax
import jax.numpy as jnp
import optax
import rlax

from collections import namedtuple
from typing import Mapping, Tuple, Union
from .agent import BaseAgent


class DQN(BaseAgent):
    """The implementation of a DQN agent

    Args:
        network (haiku.Transformed): A haiku network function that takes in the observation and outputs the Q values.
        env (gym.Env): A Gym environment for initializing the observation and action spaces.
        learning_rate (Union[float, optax.Schedule]): A learning rate that can be a float number or an optax.Schedule object.
        discount_factor (float, optional): The discount factor. Defaults to 0.99.
        epsilon_schedule (optax.Schedule, optional): The epsilon-greedy schedule. Defaults to optax.polynomial_schedule(init_value=0.9, end_value=0.05, power=1., transition_steps=50000).
        target_update_interval (int, optional): The update interval of the target network. Defaults to 50.
    """

    Params = namedtuple("Params", ["policy", "target"])
    PolicyState = namedtuple("PolicyState", ["count"])
    OptimState = namedtuple("OptimState", ["count", "opt_state"])
    PolicyOutput = namedtuple("PolicyOutput", ["actions", "q_values", "epsilon"])
    OptimOutput = namedtuple("OptimOutput", ["loss"])

    def __init__(
        self, 
        network: haiku.Transformed, 
        env: gym.Env, 
        learning_rate: Union[float, optax.Schedule],
        discount_factor: float = 0.99,
        epsilon_schedule: optax.Schedule = optax.polynomial_schedule(init_value=0.9, end_value=0.05, power=1., transition_steps=50000),
        target_update_interval: int = 50,
    ): 
        super().__init__(network, env, learning_rate)
        self._gamma = discount_factor
        self._epsilon_schedule = epsilon_schedule
        self._target_update_interval = target_update_interval
    
    def init_params(self, rng: haiku.PRNGSequence) -> Params:
        """Initialize the parameters for the agent

        Args:
            rng (haiku.PRNGSequence): the random number generator.

        Returns:
            Params: the initialized parameters.
        """
        sample_input = self._observation_space.sample()
        sample_input = jnp.expand_dims(sample_input, 0)
        params = self._network.init(rng, sample_input)
        return DQN.Params(params, params)

    def init_policy(self, rng: haiku.PRNGSequence) -> PolicyState:
        """Initialize the policy state.

        Args:
            rng (haiku.PRNGSequence): the random number generator.

        Returns:
            PolicyState: the initialized policy state.
        """
        return super().init_policy(rng)

    def init_optimizer(self, params: Params, rng: haiku.PRNGSequence) -> OptimState:
        """Initialize the optimizer state.

        Args:
            params (Params): the agent parameters.
            rng (haiku.PRNGSequence): the random number generator.

        Returns:
            OptimState: the initialized optimizer state.
        """
        return super().init_optimizer(params, rng)

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
            params (Params): the agent parameters.
            policy_state (PolicyState): the policy state.
            obs (jnp.DeviceArray): observations from the environment.
            key (haiku.PRNGSequence): the random number generator that can be used for exploration.
            evaluation (bool): whether to evaluate the policy or not.

        Returns:
            Tuple[PolicyOutput, PolicyState]: return the policy output and the updated policy state.
        """
        q = self._network.apply(params.policy, obs)
        epsilon = self._epsilon_schedule(policy_state.count)
        train_fn = jax.vmap(lambda x: rlax.epsilon_greedy(epsilon).sample(key, x))
        eval_fn = jax.vmap(lambda x: rlax.greedy().sample(key, x))
        a = jax.lax.select(evaluation, eval_fn(q), train_fn(q))
        return DQN.PolicyOutput(actions=a, q_values=q, epsilon=epsilon), DQN.PolicyState(policy_state.count + 1)
    
    def update(
        self, 
        params: Params, 
        optimizer_state: OptimState, 
        data: Mapping[str, jnp.DeviceArray], **kwargs
    ) -> Tuple[OptimOutput, Params, OptimState]:
        """Update the agent policy with given parameters

        Args:
            params (Params): the agent parameters.
            optimizer_state (OptimState): the current optimizer state.
            data (Mapping[str, jnp.DeviceArray]): the training data dict which may includes the observations, actions, rewards, etc.

        Returns:
            Tuple[OptimOutput, Params, OptimState]: return the optimizer output, updated agent parameters, and the updated optimizer state.
        """
        obs = data['obs']
        action = data['action']
        reward = data['reward']
        next_obs = data['next_obs']
        terminated = data['terminated']
        discounted = (1 - terminated) * self._gamma

        loss = jax.grad(self._loss)(params.policy, params.target, obs, action, reward, discounted, next_obs)
        updates, opt_state = self._optimizer.update(loss, optimizer_state.opt_state)
        online_params = optax.apply_updates(params.policy, updates)
        target_params = optax.periodic_update(params.policy, params.target, optimizer_state.count, self._target_update_interval)

        return (
            DQN.OptimOutput(loss),
            DQN.Params(online_params, target_params),
            DQN.OptimState(optimizer_state.count + 1, opt_state)
        )
    
    def _loss(self, online_params, target_params, obs, action, reward, discount_t, obs_t):
        q_tm1 = self._network.apply(online_params, obs)
        q_t_val = self._network.apply(target_params, obs_t)
        q_t_select = self._network.apply(online_params, obs_t)
        batched_loss = jax.vmap(rlax.double_q_learning)
        td_error = batched_loss(q_tm1, action, reward, discount_t, q_t_val, q_t_select)
        return jnp.mean(rlax.l2_loss(td_error))
