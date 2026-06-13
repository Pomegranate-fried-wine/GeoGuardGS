#!/usr/bin/env python3
"""Validate that a DA3-unsupervised config does not use LiDAR supervision."""

import sys
from pathlib import Path

from check_closed_loop_config import load_config, get


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: validate_no_lidar_leakage.py <config.yaml>")
    cfg = load_config(Path(sys.argv[1]))
    supervision = get(cfg, "train.guided_feedback.supervision_mode", "")
    if supervision != "da3_unsupervised":
        print("Not a DA3-unsupervised config; no leakage assertion required.")
        return
    if get(cfg, "train.guided_feedback.use_lidar_depth", False):
        raise SystemExit("LiDAR leakage: guided_feedback.use_lidar_depth is true.")
    if float(get(cfg, "optim.lambda_depth_lidar", 0.0) or 0.0) != 0.0:
        raise SystemExit("LiDAR leakage: optim.lambda_depth_lidar is non-zero.")
    if get(cfg, "train.feedback_controller.risk_source", "") == "lidar_error":
        raise SystemExit("LiDAR leakage: DA3 branch uses lidar_error risk source.")
    print("LiDAR supervision disabled; LiDAR is evaluation only.")


if __name__ == "__main__":
    main()
