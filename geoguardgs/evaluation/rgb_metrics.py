"""Basic RGB metrics."""

import math
import numpy as np


def mae(pred, target):
    return float(np.mean(np.abs(np.asarray(pred, dtype=np.float32) - np.asarray(target, dtype=np.float32))))


def psnr(pred, target, max_value=1.0):
    err = np.mean((np.asarray(pred, dtype=np.float32) - np.asarray(target, dtype=np.float32)) ** 2)
    if err <= 1e-12:
        return float("inf")
    return float(20.0 * math.log10(float(max_value)) - 10.0 * math.log10(float(err)))
