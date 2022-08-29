import jax
import jax.numpy as jnp
import optax
import types
import collections
import gym
import rlax

from baselax.utils.network import build_network


class DQN:
    Params = collections.namedtuple("Params", "online target")
    ActorState = collections.namedtuple("ActorState", "count")
    ActorOutput = collections.namedtuple("ActorOutput", "actions q_values")
    LearnerState = collections.namedtuple("LearnerState", "count opt_state")

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, config: types.SimpleNamespace):
        """Deep Q Network agent implementaion using double Q-learning.

        Args:
            observation_space (gym.Space): the observation space of the environment.
            action_space (gym.Space): the action space of the environment.
            config (types.SimpleNamespace): additional configuration parameters.
        """
        self._observation_space = observation_space
        self._action_space = action_space
        epsilon_cfg = dict(
            init_value=config.epsilon_begin,
            end_value=config.epsilon_end,
            transition_steps=config.epsilon_steps,
            power=1.
        )
        self._target_period = config.target_period
        # Neural net and optimiser.
        self._network = build_network(action_space.n, config.hidden_units)
        self._optimizer = optax.adam(config.learning_rate)
        self._epsilon_by_frame = optax.polynomial_schedule(**epsilon_cfg)

    def jit(self):
        """Jitting agent `actor_step` and `learner_step` methods for speeding up
        
        Examples:
            >>> agent = DQN(...)
            >>> agent.jit()
        """
        self.actor_step = jax.jit(self.actor_step)
        self.learner_step = jax.jit(self.learner_step)
        self.actor_step_batch = jax.jit(self.actor_step_batch)

    def initial_params(self, key):
        sample_input = self._observation_space.sample()
        sample_input = jnp.expand_dims(sample_input, 0)
        online_params = self._network.init(key, sample_input)
        return DQN.Params(online_params, online_params)

    def initial_actor_state(self):
        actor_count = jnp.zeros((), dtype=jnp.float32)
        return DQN.ActorState(actor_count)

    def initial_learner_state(self, params):
        learner_count = jnp.zeros((), dtype=jnp.float32)
        opt_state = self._optimizer.init(params.online)
        return DQN.LearnerState(learner_count, opt_state)

    def actor_step_batch(self, params, obs, actor_state, key, evaluation):
        q = self._network.apply(params.online, obs)
        epsilon = self._epsilon_by_frame(actor_state.count)
        train_fn = jax.vmap(lambda x: rlax.epsilon_greedy(epsilon).sample(key, x))
        eval_fn = jax.vmap(lambda x: rlax.greedy().sample(key, x))
        a = jax.lax.select(evaluation, eval_fn(q), train_fn(q))
        return DQN.ActorOutput(actions=a, q_values=q), DQN.ActorState(actor_state.count + 1)

    def actor_step(self, params, obs, actor_state, key, evaluation):
        obs = jnp.expand_dims(obs, 0)
        q = self._network.apply(params.online, obs)[0]  # remove dummy batch
        epsilon = self._epsilon_by_frame(actor_state.count)
        train_a = rlax.epsilon_greedy(epsilon).sample(key, q)
        eval_a = rlax.greedy().sample(key, q)
        a = jax.lax.select(evaluation, eval_a, train_a)
        return DQN.ActorOutput(actions=a, q_values=q), DQN.ActorState(actor_state.count + 1)

    def learner_step(self, params, data, learner_state, unused_key):
        target_params = optax.periodic_update(params.online, params.target, learner_state.count, self._target_period)
        obs = data['obs']
        action = data['action']
        reward = data['reward']
        next_obs = data['next_obs']
        discounted = data['discounted']
        dloss_dtheta = jax.grad(self._loss)(params.online, target_params, obs, action, reward, discounted, next_obs)
        updates, opt_state = self._optimizer.update(dloss_dtheta, learner_state.opt_state)
        online_params = optax.apply_updates(params.online, updates)
        return (DQN.Params(online_params, target_params),
                DQN.LearnerState(learner_state.count + 1, opt_state))

    def _loss(self, online_params, target_params, obs, action, reward, discount_t, obs_t):
        q_tm1 = self._network.apply(online_params, obs)
        q_t_val = self._network.apply(target_params, obs_t)
        q_t_select = self._network.apply(online_params, obs_t)
        batched_loss = jax.vmap(rlax.double_q_learning)
        td_error = batched_loss(q_tm1, action, reward, discount_t, q_t_val, q_t_select)
        return jnp.mean(rlax.l2_loss(td_error))
