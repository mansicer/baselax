import types
import haiku as hk
import gym
import envpool
import jax
import jax.numpy as jnp
import numpy as np
from absl import app
from absl import flags
from packaging import version
from stable_baselines3.common.env_util import DummyVecEnv
from UtilsRL.rl.buffer import TransitionReplayPool

from baselax.agent.dqn import DQN
from baselax.utils.seeding import global_seed
from baselax.utils.network import mlp_network

config = flags.FLAGS

# experiment configs
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("use_gpu", True, "Whether to use GPU or not.")
flags.DEFINE_bool("jitting", True, "Whether to run without jitting.")

# training configs
flags.DEFINE_integer("training_steps", 1_000_000, "Number of train episodes.")
flags.DEFINE_integer("eval_episodes", 50, "Number of evaluation episodes.")
flags.DEFINE_integer("evaluate_every", 2_000, "Number of episodes between evaluations.")

# optimizer configs
flags.DEFINE_float("learning_rate", 0.0003, "Optimizer learning rate.")

# network configs
flags.DEFINE_integer("batch_size", 64, "Size of the training batch")
flags.DEFINE_integer("replay_capacity", 1_000_000, "Capacity of the replay buffer.")
flags.DEFINE_list("hidden_units", [64, 64], "Number of network hidden units.")

# RL configs
flags.DEFINE_float("target_period", 10, "How often to update the target net.")
flags.DEFINE_float("discount_factor", 0.99, "Q-learning discount factor.")
flags.DEFINE_float("epsilon_begin", .5, "Initial epsilon-greedy exploration.")
flags.DEFINE_float("epsilon_end", 0.05, "Final epsilon-greedy exploration.")
flags.DEFINE_integer("epsilon_steps", 2_000, "Steps over which to anneal eps.")

# env configs
flags.DEFINE_string("env", "CartPole-v1", "Name of the OpenAI Gym environment to use.")
flags.DEFINE_integer("num_envs", 20, "Number of environments to use.")


def evaluate(eval_environment, eval_episode_num, agent, params, rng):
    returns_list = []
    for _ in range(eval_episode_num):
        obs = eval_environment.reset()
        actor_state = agent.init_policy(rng)
        returns = 0.0
        done = False

        while not done:
            actor_output, actor_state = agent.predict(params, actor_state, obs, next(rng), evaluation=True)
            obs, reward, done, info = eval_environment.step(np.array(actor_output.actions))
            returns += reward
        returns_list.append(returns)

    return returns_list



def run_loop(
    agent, train_environment, eval_environment, buffer, config, max_episode_steps=500):
    """A simple run loop for examples of reinforcement learning with rlax."""

    # Init agent.
    rng = hk.PRNGSequence(jax.random.PRNGKey(config.seed))
    params = agent.init_params(next(rng))
    learner_state = agent.init_optimizer(params, next(rng))
    env_max_length = max_episode_steps

    training_state = types.SimpleNamespace(step=0, episode=0, last_eval_t=-1)
    print(f"Training agent for {config.training_steps} timesteps.")
    while training_state.step < config.training_steps:

        # Prepare agent, environment and accumulator for a new episode.
        obs = train_environment.reset()
        actor_state = agent.init_policy(next(rng))
        not_done = np.ones((config.num_envs,), dtype=jnp.bool_)

        while True:

            # Acting.
            actor_output, actor_state = agent.predict(params, actor_state, obs, next(rng), evaluation=False)

            # Agent-environment interaction.
            action = np.array(actor_output.actions)
            next_obs, reward, done, info = train_environment.step(action)
            terminated = done * (actor_state.count < env_max_length)

            # Save data to buffer.
            buffer.add_samples({
                "obs": obs[not_done],
                "action": action[not_done],
                "reward": reward[not_done].reshape(-1, 1),
                "terminated": terminated[not_done].reshape(-1, 1),
                "next_obs": next_obs[not_done]
            })

            obs = next_obs
            sample_step = not_done.sum()
            not_done[done] = False

            # Learning.
            if buffer.size >= config.batch_size:
                for _ in range(sample_step):
                    batch = buffer.random_batch(config.batch_size)
                    batch['reward'] = batch['reward'].reshape(-1)
                    batch['terminated'] = batch['terminated'].reshape(-1)
                    optim_output, params, learner_state = agent.update(params, learner_state, batch)
                training_state.step += sample_step

            if (~not_done).all():
                training_state.episode += config.num_envs
                break

            # Evaluation at: (1) every interval (2) first update (3) last update.
            if training_state.step - training_state.last_eval_t >= config.evaluate_every or training_state.last_eval_t < 0 or training_state.step > config.training_steps:
                training_state.last_eval_t = training_state.step
                returns = evaluate(eval_environment, config.eval_episodes, agent, params, rng)
                avg_returns = np.mean(returns)
                print(f"Training step {training_state.step:10d}, Episode {training_state.episode:4d}: Average returns: {avg_returns:.2f}")


def main(unused_arg):
    """Run the DQN agent on the OpenAI Gym environment."""

    if config.use_gpu:
        jax.config.update('jax_platform_name', 'cuda')
    else:
        jax.config.update('jax_platform_name', 'cpu')

    train_env = envpool.make(config.env, env_type="gym", num_envs=config.num_envs)
    if version.parse(gym.__version__) >= version.parse("0.25.0"):
        eval_env = DummyVecEnv([lambda: gym.make(config.env, new_step_api=False)])
    else:
        eval_env = DummyVecEnv([lambda: gym.make(config.env)])
    
    max_episode_steps = gym.make(config.env).spec.max_episode_steps

    global_seed(config.seed)
    train_env.seed(config.seed)
    eval_env.seed(config.seed)

    agent = DQN(
        network=mlp_network(config.hidden_units, train_env.action_space),
        env=train_env,
        learning_rate=config.learning_rate,
    )

    buffer = TransitionReplayPool(
        train_env.observation_space, 
        train_env.action_space,
        max_size=config.replay_capacity,
        extra_fields={
            "action": {
                "shape": train_env.action_space.shape,
                "dtype": np.int32,
            },
            "terminated": {
                "shape": (1,),
                "dtype": np.bool_,
            }
        }
    )

    if config.jitting:
        agent.jit()

    run_loop(
        agent=agent,
        train_environment=train_env,
        eval_environment=eval_env,
        buffer=buffer,
        config=config,
        max_episode_steps=max_episode_steps
    )

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    app.run(main)
