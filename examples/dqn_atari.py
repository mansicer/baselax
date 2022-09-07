import types
import haiku as hk
import gym
import envpool
import jax
import numpy as np
import nni
import optax
import os
from absl import app
from absl import flags
from packaging import version
from stable_baselines3.common.env_util import SubprocVecEnv
from UtilsRL.rl.buffer import TransitionReplayPool

from baselax.agents import DQN
from baselax.utils.seeding import global_seed
from baselax.utils.network import cnn_network

config = flags.FLAGS

# experiment configs
flags.DEFINE_integer("seed", np.random.randint(0, 1000000), "Random seed.")
flags.DEFINE_bool("use_gpu", True, "Whether to use GPU or not.")
flags.DEFINE_bool("jitting", True, "Whether to run without jitting.")
flags.DEFINE_bool("use_nni", True, "Whether to use NNI experiments or not.")

# training configs
flags.DEFINE_integer("training_steps", 2_000_000, "Number of train episodes.")
flags.DEFINE_integer("eval_episodes", 32, "Number of evaluation episodes.")
flags.DEFINE_integer("evaluate_every", 10_000, "Number of episodes between evaluations.")

# optimizer configs
flags.DEFINE_float("learning_rate", 0.0003, "Optimizer learning rate.")

# network configs
flags.DEFINE_integer("batch_size", 64, "Size of the training batch")
flags.DEFINE_integer("buffer_size", 100_000, "Capacity of the replay buffer.")

# RL configs
flags.DEFINE_float("target_update_interval", 10, "How often to update the target net.")
flags.DEFINE_float("discount_factor", 0.99, "Q-learning discount factor.")
flags.DEFINE_float("epsilon_begin", 1.0, "Initial epsilon-greedy exploration.")
flags.DEFINE_float("epsilon_end", 0.05, "Final epsilon-greedy exploration.")
flags.DEFINE_integer("epsilon_steps", 50000, "Steps over which to anneal eps.")

# env configs
flags.DEFINE_bool("use_envpool", True, "Whether to use envpool or not to create environments.")
flags.DEFINE_string("env", "Pong-v5", "Name of the OpenAI Gym environment to use.")
flags.DEFINE_integer("num_envs", 32, "Number of environments to use.")


def evaluate(eval_environment, eval_episode_num, agent, params, rng):
    returns_list = []
    num_envs = eval_environment.reset().shape[0]
    eval_rounds = int(eval_episode_num // num_envs)
    for _ in range(eval_rounds):
        obs = eval_environment.reset()
        actor_state = agent.init_policy(rng)
        returns = np.zeros(num_envs)
        not_done = np.ones((num_envs,), dtype=np.bool_)

        while True:
            # Acting.
            actor_output, actor_state = agent.predict(params, actor_state, obs, next(rng), evaluation=False)

            # Agent-environment interaction.
            action = np.array(actor_output.actions)
            obs, reward, done, info = eval_environment.step(action)
            returns += reward * not_done

            not_done[done] = False

            if (~not_done).all():
                returns_list.extend(returns.tolist())
                break

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
        not_done = np.ones((config.num_envs,), dtype=np.bool_)

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
            training_state.step += sample_step
            not_done[done] = False

            # Learning.
            if buffer.size >= config.batch_size:
                for _ in range(sample_step):
                    batch = buffer.random_batch(config.batch_size)
                    batch['reward'] = batch['reward'].reshape(-1)
                    batch['terminated'] = batch['terminated'].reshape(-1)
                    optim_output, params, learner_state = agent.update(params, learner_state, batch)

            # Evaluation at: (1) every interval (2) first update (3) last update.
            if training_state.step - training_state.last_eval_t >= config.evaluate_every or training_state.last_eval_t < 0 or training_state.step >= config.training_steps:
                training_state.last_eval_t = training_state.step
                returns = evaluate(eval_environment, config.eval_episodes, agent, params, rng)
                avg_returns = np.mean(returns)
                print(f"Training step {training_state.step:10d}, Episode {training_state.episode:4d}: Average returns: {avg_returns:.2f}")

                if config.use_nni:
                    nni.report_intermediate_result(avg_returns)

            if training_state.step >= config.training_steps or (~not_done).all():
                if (~not_done).all():
                    training_state.episode += config.num_envs
                break
    
    if config.use_nni:
        nni.report_final_result(avg_returns)


def main(unused_arg):
    """Run the DQN agent on the OpenAI Gym environment."""

    if config.use_gpu:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
        jax.config.update('jax_platform_name', 'cuda')
    else:
        jax.config.update('jax_platform_name', 'cpu')

    if config.use_envpool:
        train_env = envpool.make(config.env, env_type="gym", num_envs=config.num_envs)
        eval_env = envpool.make(config.env, env_type="gym", num_envs=config.num_envs)
        max_episode_steps = train_env.config["max_episode_steps"]
    else:
        env_name = str(config.env)
        if version.parse(gym.__version__) >= version.parse("0.25.0"):
            train_env = SubprocVecEnv([lambda: gym.make(env_name, new_step_api=False) for _ in range(config.num_envs)])
            eval_env = SubprocVecEnv([lambda: gym.make(env_name, new_step_api=False) for _ in range(config.num_envs)])
        else:
            train_env = SubprocVecEnv([lambda: gym.make(env_name) for _ in range(config.num_envs)])
            eval_env = SubprocVecEnv([lambda: gym.make(env_name) for _ in range(config.num_envs)])
        max_episode_steps = gym.make(config.env).spec.max_episode_steps

    global_seed(config.seed)
    train_env.seed(config.seed)
    eval_env.seed(config.seed)

    if config.use_nni:
        nni_config = nni.get_next_parameter()
        config.epsilon_steps = nni_config.get("epsilon_steps", config.epsilon_steps)
        config.buffer_size = nni_config.get("target_update_interval", config.target_update_interval)

    agent = DQN(
        network=cnn_network(conv_channels=[32, 64, 64], kernel_size=[5, 5, 5], stride=[2, 2, 2], padding=["VALID", "VALID", "SAME"]),
        env=train_env,
        learning_rate=config.learning_rate,
        discount_factor=config.discount_factor,
        epsilon_schedule=optax.polynomial_schedule(init_value=0.9, end_value=0.05, power=1., transition_steps=config.epsilon_steps),
        target_update_interval=config.target_update_interval,
    )

    buffer = TransitionReplayPool(
        train_env.observation_space, 
        train_env.action_space,
        max_size=config.buffer_size,
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
