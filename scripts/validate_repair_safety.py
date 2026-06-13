#!/usr/bin/env python3
"""Validate Gaussian repair safety gates in a config."""

import sys
from pathlib import Path

from check_closed_loop_config import load_config, get


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: validate_repair_safety.py <config.yaml>")
    cfg = load_config(Path(sys.argv[1]))
    mode = get(cfg, "train.gaussian_control.control_mode", "off")
    if get(cfg, "train.gaussian_control.allow_real_prune", False):
        raise SystemExit("Unsafe: real prune is enabled.")
    if get(cfg, "train.gaussian_control.allow_real_split", False):
        raise SystemExit("Unsafe: real split is enabled.")
    if get(cfg, "train.gaussian_control.allow_real_shrink", False):
        raise SystemExit("Unsafe: real shrink is enabled.")
    if mode in {"prune_apply", "split_apply", "shrink_apply", "repair_apply"}:
        raise SystemExit(f"Unsafe: {mode} is not enabled in this release.")
    if mode == "opacity_decay_apply" and not get(cfg, "train.gaussian_control.allow_parameter_modification", False):
        raise SystemExit("Unsafe config: opacity_decay_apply requires allow_parameter_modification=true.")
    print("Repair safety check passed.")


if __name__ == "__main__":
    main()
