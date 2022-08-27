import random
import numpy as np

def global_seed(seed: int):
    """Seed the random number generator manually."""
    random.seed(seed)
    np.random.seed(seed)
