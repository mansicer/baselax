import types
import haiku as hk
import gym
import envpool
import jax
import jax.numpy as jnp
import numpy as np
from absl import app
from absl import flags
from agent.dqn import DQN
from UtilsRL.rl.buffer import TransitionReplayPool

from utils.seeding import global_seed

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
        actor_state = agent.initial_actor_state()
        returns = 0.0
        done = False

        while not done:
            actor_output, actor_state = agent.actor_step(params, obs, actor_state, next(rng), evaluation=True)
            obs, reward, done, info = eval_environment.step(int(actor_output.actions))
            returns += reward
        returns_list.append(returns)

    return returns_list



def run_loop(
    agent, train_environment, eval_environment, buffer, config):
    """A simple run loop for examples of reinforcement learning with rlax."""

    # Init agent.
    rng = hk.PRNGSequence(jax.random.PRNGKey(config.seed))
    params = agent.initial_params(next(rng))
    learner_state = agent.initial_learner_state(params)
    env_max_length = eval_environment.spec.max_episode_steps

    training_state = types.SimpleNamespace(step=0, episode=0, last_eval_t=-1)
    print(f"Training agent for {config.training_steps} timesteps.")
    while training_state.step < config.training_steps:

        # Prepare agent, environment and accumulator for a new episode.
        obs = train_environment.reset()
        actor_state = agent.initial_actor_state()
        not_done = np.ones((config.num_envs,), dtype=jnp.bool_)

        while True:

            # Acting.
            actor_output, actor_state = agent.actor_step_batch(params, obs, actor_state, next(rng), evaluation=False)

            # Agent-environment interaction.
            action = np.array(actor_output.actions)
            next_obs, reward, done, info = train_environment.step(action)
            terminated = done * (actor_state.count < env_max_length)
            discounted = (1 - terminated) * config.discount_factor

            # Save data to buffer.
            buffer.add_samples({
                "obs": obs[not_done],
                "action": action[not_done],
                "reward": reward[not_done].reshape(-1, 1),
                "discounted": discounted[not_done].reshape(-1, 1),
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
                    batch['discounted'] = batch['discounted'].reshape(-1)
                    params, learner_state = agent.learner_step(params, batch, learner_state, next(rng))
                training_state.step += sample_step

            if (~not_done).all():
                training_state.episode += config.num_envs
                break

            # Evaluation.
            if training_state.step - training_state.last_eval_t >= config.evaluate_every or training_state.last_eval_t < 0:
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
    eval_env = gym.make(config.env, new_step_api=False)

    global_seed(config.seed)
    train_env.seed(config.seed)
    eval_env.seed(config.seed)

    agent = DQN(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        config=config,
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
            "discounted": {
                "shape": (1,),
                "dtype": np.float32,
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
        config=config
    )

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    app.run(main)
