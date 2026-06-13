"""DA3 structure metric placeholders.

DA3 metrics are structure comparisons, not metric-depth ground truth metrics.
"""

import numpy as np


def edge_mismatch(rendered_edge, da3_edge, mask=None):
    rendered_edge = np.asarray(rendered_edge, dtype=np.float32)
    da3_edge = np.asarray(da3_edge, dtype=np.float32)
    valid = np.isfinite(rendered_edge) & np.isfinite(da3_edge)
    if mask is not None:
        valid &= np.asarray(mask).astype(bool)
    if not np.any(valid):
        return None
    return float(np.mean(np.abs(rendered_edge[valid] - da3_edge[valid])))
