from __future__ import annotations

import random
from random import Random

DEFAULT_SEED = 0


def normalize_seed(seed: int | None) -> int:
    """Return a stable integer seed value."""
    if seed is None:
        return DEFAULT_SEED
    return int(seed)


def set_seed(seed: int) -> Random:
    """
    Set deterministic seeds and return a seeded Random instance.
    Same seed -> same generated values across env runs.
    """
    stable_seed = normalize_seed(seed)
    random.seed(stable_seed)
    try:
        import numpy as np  # optional dependency

        np.random.seed(stable_seed)
    except Exception:
        pass
    return Random(stable_seed)
