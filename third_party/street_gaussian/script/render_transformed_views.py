import argparse
import json
import os
import sys

import cv2
import imageio
import numpy as np
import torch
import torchvision

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_script_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--enable-camera-transform", action="store_true")
    parser.add_argument("--cam-shift-x", type=float, default=0.0)
    parser.add_argument("--cam-shift-y", type=float, default=0.0)
    parser.add_argument("--cam-shift-z", type=float, default=0.0)
    parser.add_argument("--cam-yaw-deg", type=float, default=0.0)
    parser.add_argument("--cam-pitch-deg", type=float, default=0.0)
    parser.add_argument("--cam-roll-deg", type=float, default=0.0)
    parser.add_argument("--cam-transform-mode", choices=["camera", "world"], default="camera")
    parser.add_argument("--save-transformed-view", action="store_true")
    parser.add_argument("--output-dir", default="output/local_formal/transformed_view_debug")
    parser.add_argument("--views", nargs="+", default=None)
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument("--save-acc", action="store_true")
    parser.add_argument("--help-transform", action="store_true")
    script_args, remaining = parser.parse_known_args()
    if script_args.help_transform:
        parser.print_help()
        sys.exit(0)
    sys.argv = [sys.argv[0]] + remaining
    return script_args


SCRIPT_ARGS = parse_script_args()

from lib.config import cfg  # noqa: E402
from lib.datasets.dataset import Dataset  # noqa: E402
from lib.models.scene import Scene  # noqa: E402
from lib.models.street_gaussian_model import StreetGaussianModel  # noqa: E402
from lib.models.street_gaussian_renderer import StreetGaussianRenderer  # noqa: E402
from lib.utils.camera_transform_utils import apply_camera_transform_inplace, snapshot_camera_transform, restore_camera_transform  # noqa: E402
from lib.utils.general_utils import safe_state  # noqa: E402
from lib.utils.img_utils import visualize_depth_numpy  # noqa: E402


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def normalize_depth(depth):
    visual = visualize_depth_numpy(depth, cmap=cv2.COLORMAP_JET)[0]
    return visual[..., [2, 1, 0]]


def save_result(out_dir, stem, result, camera, transform_metadata):
    rgb_path = os.path.join(out_dir, f"{stem}_rgb.png")
    torchvision.utils.save_image(result["rgb"], rgb_path)

    paths = {"rgb": rgb_path}
    depth = result.get("depth")
    acc = result.get("acc")
    if depth is not None and SCRIPT_ARGS.save_depth:
        depth_np = depth.detach().cpu().numpy().squeeze()
        if acc is not None:
            acc_np = acc.detach().cpu().numpy().squeeze()
            depth_np = depth_np / (acc_np + 1e-10)
        np.save(os.path.join(out_dir, f"{stem}_depth.npy"), depth_np)
        imageio.imwrite(os.path.join(out_dir, f"{stem}_depth.png"), normalize_depth(depth_np))
        paths["depth_npy"] = os.path.join(out_dir, f"{stem}_depth.npy")
        paths["depth_png"] = os.path.join(out_dir, f"{stem}_depth.png")
    if acc is not None and SCRIPT_ARGS.save_acc:
        acc_np = acc.detach().cpu().numpy().squeeze()
        np.save(os.path.join(out_dir, f"{stem}_acc.npy"), acc_np)
        paths["acc_npy"] = os.path.join(out_dir, f"{stem}_acc.npy")

    metadata = {
        "stem": stem,
        "camera_id": int(camera.meta.get("cam", -1)) if hasattr(camera, "meta") else None,
        "frame_id": int(camera.meta.get("frame", -1)) if hasattr(camera, "meta") else None,
        "transform": transform_metadata,
        "warning": "Transformed views are novel-view renders and must not be evaluated against original LiDAR unless camera/LiDAR geometry is transformed consistently.",
        "paths": paths,
    }
    with open(os.path.join(out_dir, f"{stem}_camera_transform_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


def main():
    cfg.mode = "evaluate"
    cfg.render.save_image = False
    cfg.render.save_video = False
    safe_state(cfg.eval.quiet)

    ensure_dir(SCRIPT_ARGS.output_dir)
    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        renderer = StreetGaussianRenderer()
        cameras = scene.getTrainCameras()
        selected = set(SCRIPT_ARGS.views) if SCRIPT_ARGS.views else None
        summaries = []

        for camera in cameras:
            stem = camera.image_name
            if selected is not None and stem not in selected:
                continue
            snap = snapshot_camera_transform(camera)
            transform_metadata = {"enabled": False}
            if SCRIPT_ARGS.enable_camera_transform:
                transform_metadata = apply_camera_transform_inplace(
                    camera,
                    shift=(SCRIPT_ARGS.cam_shift_x, SCRIPT_ARGS.cam_shift_y, SCRIPT_ARGS.cam_shift_z),
                    rotation_deg=(SCRIPT_ARGS.cam_yaw_deg, SCRIPT_ARGS.cam_pitch_deg, SCRIPT_ARGS.cam_roll_deg),
                    mode=SCRIPT_ARGS.cam_transform_mode,
                )
            result = renderer.render(camera, gaussians)
            summaries.append(save_result(SCRIPT_ARGS.output_dir, stem, result, camera, transform_metadata))
            restore_camera_transform(camera, snap)

    summary = {
        "output_dir": os.path.abspath(SCRIPT_ARGS.output_dir),
        "view_count": len(summaries),
        "enable_camera_transform": SCRIPT_ARGS.enable_camera_transform,
        "shift": [SCRIPT_ARGS.cam_shift_x, SCRIPT_ARGS.cam_shift_y, SCRIPT_ARGS.cam_shift_z],
        "rotation_deg": [SCRIPT_ARGS.cam_yaw_deg, SCRIPT_ARGS.cam_pitch_deg, SCRIPT_ARGS.cam_roll_deg],
        "mode": SCRIPT_ARGS.cam_transform_mode,
        "views": [item["stem"] for item in summaries],
        "warning": "Do not use transformed-view outputs for LiDAR metrics or responsibility unless ground-truth geometry is transformed consistently.",
    }
    with open(os.path.join(SCRIPT_ARGS.output_dir, "camera_transform_run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved transformed-view outputs: {SCRIPT_ARGS.output_dir}")


if __name__ == "__main__":
    main()
