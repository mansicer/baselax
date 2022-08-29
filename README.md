**Status**: in development, no practical use.

# Baselax

Baselax (**Basel**ines + j**ax**) provides stable-baselines-style implementations of reinforcement learning (RL) algorithms with Google JAX framework, supported by the following frameworks:

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

Additnionally, you can download and install `envpool` wheels from [GitHub releases](https://github.com/sail-sg/envpool/releases) for better sample efficiency.
