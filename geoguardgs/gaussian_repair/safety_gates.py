"""Safety gates for conservative Gaussian control."""


REAL_REPAIR_MODES = {"repair_apply", "prune_apply", "shrink_apply", "split_apply"}


def assert_no_real_structure_repair(control_mode, allow_real_prune=False, allow_real_split=False, allow_real_shrink=False):
    if control_mode in REAL_REPAIR_MODES or allow_real_prune or allow_real_split or allow_real_shrink:
        raise RuntimeError("Real prune/shrink/split is disabled in this release.")


def assert_da3_unsupervised_no_lidar(use_lidar_depth, lambda_depth_lidar, risk_source):
    if use_lidar_depth or float(lambda_depth_lidar) != 0.0 or risk_source == "lidar_error":
        raise RuntimeError("DA3-unsupervised mode must not use LiDAR supervision or LiDAR risk pixels.")
