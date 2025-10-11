from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_all(seed: int, *, deterministic: bool = True) -> None:
    """Seed random generators across Python, NumPy, and torch backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        try:
            torch.use_deterministic_algorithms(False)
        except RuntimeError:
            pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
