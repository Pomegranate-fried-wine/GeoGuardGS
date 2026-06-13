import argparse
import json
import os
from collections import defaultdict

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize a Waymo frame/camera subset before formal experiments.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=15)
    parser.add_argument("--output", required=True)
    parser.add_argument("--low-lidar-threshold", type=int, default=50)
    return parser.parse_args()


def lidar_info(path):
    if not os.path.exists(path):
        return {"exists": False, "is_dict_mask_value": False, "lidar_valid_count": 0, "low_lidar": True}
    data = np.load(path, allow_pickle=True)
    is_dict = isinstance(data, np.ndarray) and data.shape == () and isinstance(data.item(), dict)
    if is_dict:
        item = data.item()
        ok = "mask" in item and "value" in item
        count = int(np.count_nonzero(np.asarray(item["mask"]).astype(bool))) if ok else 0
        value_count = int(np.asarray(item["value"]).reshape(-1).shape[0]) if ok else 0
        return {
            "exists": True,
            "is_dict_mask_value": bool(ok),
            "lidar_valid_count": count,
            "lidar_value_count": value_count,
            "mask_value_count_match": bool(count == value_count),
        }
    arr = np.asarray(data)
    return {
        "exists": True,
        "is_dict_mask_value": False,
        "lidar_valid_count": int(np.count_nonzero(np.isfinite(arr) & (arr > 0))),
        "lidar_value_count": None,
        "mask_value_count_match": None,
    }


def main():
    args = parse_args()
    image_dir = os.path.join(args.data_dir, "images")
    lidar_dir = os.path.join(args.data_dir, "lidar_depth")
    frames = list(range(args.frame_start, args.frame_start + args.frame_count))
    cameras_by_frame = defaultdict(set)
    for name in os.listdir(image_dir):
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        stem = os.path.splitext(name)[0]
        try:
            frame_s, cam_s = stem.split("_", 1)
            frame, cam = int(frame_s), int(cam_s)
        except ValueError:
            continue
        if frame in frames:
            cameras_by_frame[frame].add(cam)

    cameras = sorted(set().union(*cameras_by_frame.values())) if cameras_by_frame else []
    views = []
    for frame in frames:
        for cam in cameras:
            stem = f"{frame:06d}_{cam}"
            image_path = os.path.join(image_dir, f"{stem}.png")
            lidar_path = os.path.join(lidar_dir, f"{stem}.npy")
            info = lidar_info(lidar_path)
            low_lidar = info["lidar_valid_count"] < args.low_lidar_threshold
            views.append(
                {
                    "stem": stem,
                    "frame": frame,
                    "camera": cam,
                    "image_exists": os.path.exists(image_path),
                    "lidar_exists": info["exists"],
                    "lidar_is_dict_mask_value": info["is_dict_mask_value"],
                    "lidar_valid_count": info["lidar_valid_count"],
                    "lidar_value_count": info.get("lidar_value_count"),
                    "mask_value_count_match": info.get("mask_value_count_match"),
                    "low_lidar": bool(low_lidar),
                    "missing": bool((not os.path.exists(image_path)) or (not info["exists"])),
                }
            )

    summary = {
        "data_dir": os.path.abspath(args.data_dir),
        "frame_start": args.frame_start,
        "frame_count": args.frame_count,
        "frames": frames,
        "cameras": cameras,
        "total_views": len(views),
        "available_image_views": int(sum(v["image_exists"] for v in views)),
        "available_lidar_views": int(sum(v["lidar_exists"] for v in views)),
        "dict_mask_value_lidar_views": int(sum(v["lidar_is_dict_mask_value"] for v in views)),
        "missing_views": [v["stem"] for v in views if v["missing"]],
        "low_lidar_views": [v["stem"] for v in views if v["low_lidar"]],
        "min_lidar_valid_count": int(min([v["lidar_valid_count"] for v in views], default=0)),
        "max_lidar_valid_count": int(max([v["lidar_valid_count"] for v in views], default=0)),
        "views": views,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"frames={len(frames)} cameras={cameras} views={len(views)}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
