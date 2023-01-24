import contextlib
import random

import torch

__all__ = ["manual_seed"]


def manual_seed(seed: int) -> None:
    """Setup random state from a seed for `torch`, `random` and optionally `numpy` (if can be imported).
    Args:
        seed: Random state seed
    """
    random.seed(seed)
    torch.manual_seed(seed)

    with contextlib.suppress(ImportError):
        import numpy as np

        np.random.seed(seed)
