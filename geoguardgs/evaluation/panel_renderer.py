"""Panel rendering placeholder.

The production project can replace this with a richer OpenCV/Pillow contact
sheet builder. This helper intentionally keeps imports light.
"""

from pathlib import Path


def prepare_panel_dir(path):
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
