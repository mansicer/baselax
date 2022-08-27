**Status**: in development, no practical use.

# Baselax (Baselines + JAX)

Baselax (meaning Baselines + JAX) provides stable-baselines-style implementations of reinforcement learning (RL) algorithms with Google JAX framework, supported by the following frameworks:

- [Google JAX](https://github.com/google/jax)
- [DeepMind Optax](https://github.com/deepmind/optax)
- [DeepMind Haiku](https://github.com/deepmind/dm-haiku)
- [DeepMind RLax](https://github.com/deepmind/rlax)

## Installation

Install via `pip`:

```bash
pip install -r requirements.txt
pip install git+https://github.com/deepmind/dm-haiku
```

Additnionally, you should download and install `envpool` wheels from [GitHub releases](https://github.com/sail-sg/envpool/releases).
