"""Boundary metric helpers."""

import numpy as np


def masked_mean(values, mask):
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(mask).astype(bool) & np.isfinite(values)
    if not np.any(mask):
        return None
    return float(np.mean(values[mask]))
