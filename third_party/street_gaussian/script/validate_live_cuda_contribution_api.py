import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("PWD", PROJECT_ROOT)


def parse_script_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cached-summary", default="output/local_feedback/da3_boundary_contribution_debug_A5000_top30_regionpixels/contribution_responsibility_all_views_summary.json")
    parser.add_argument("--output", default="output/local_feedback/live_cuda_contribution_api_validation/live_cuda_contribution_api_validation.json")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--frame-index", type=int, default=0)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


SCRIPT_ARGS = parse_script_args()

from lib.config import cfg  # noqa: E402
from lib.datasets.dataset import Dataset  # noqa: E402
from lib.models.scene import Scene  # noqa: E402
from lib.models.street_gaussian_model import StreetGaussianModel  # noqa: E402
from lib.utils.cuda_contribution_utils import capture_contributions_cuda_live  # noqa: E402
from lib.utils.general_utils import safe_state  # noqa: E402


def read_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def cached_stable_ids(npz):
    ids = np.asarray(npz["cuda_contribution_ids"], dtype=np.int64)
    if "stable_gaussian_ids" in npz:
        return np.asarray(npz["stable_gaussian_ids"], dtype=np.int64)
    stable = np.full_like(ids, -1, dtype=np.int64)
    if "candidate_view_local_indices" in npz and "candidate_gaussian_ids" in npz:
        rows = np.asarray(npz["candidate_view_local_indices"], dtype=np.int64).reshape(-1)
        gids = np.asarray(npz["candidate_gaussian_ids"], dtype=np.int64).reshape(-1)
        mapping = {int(r): int(g) for r, g in zip(rows, gids)}
        for index, row in np.ndenumerate(ids):
            stable[index] = mapping.get(int(row), -1)
    return stable


def compare(cached_npz, live):
    cached_ids = cached_stable_ids(cached_npz)
    live_ids = np.asarray(live["stable_gaussian_ids"], dtype=np.int64)
    cached_w = np.asarray(cached_npz["contribution_weights"], dtype=np.float32)
    live_w = np.asarray(live["contribution_weight"], dtype=np.float32)
    cached_order = np.asarray(cached_npz["cuda_depth_order"], dtype=np.int32)
    live_order = np.asarray(live["depth_order"], dtype=np.int32)
    rows = min(cached_ids.shape[0], live_ids.shape[0])
    shared = 0
    total_cached = 0
    weight_diffs = []
    order_diffs = []
    for p in range(rows):
        c = {int(g): k for k, g in enumerate(cached_ids[p]) if int(g) >= 0}
        l = {int(g): k for k, g in enumerate(live_ids[p]) if int(g) >= 0}
        total_cached += len(c)
        for gid, ck in c.items():
            lk = l.get(gid)
            if lk is None:
                continue
            shared += 1
            weight_diffs.append(abs(float(cached_w[p, ck]) - float(live_w[p, lk])))
            order_diffs.append(abs(int(cached_order[p, ck]) - int(live_order[p, lk])))
    return {
        "pixel_count": int(rows),
        "cached_contributor_count": int(total_cached),
        "shared_contributor_count": int(shared),
        "shared_ratio": float(shared / max(total_cached, 1)),
        "talpha_abs_diff_mean": float(np.mean(weight_diffs)) if weight_diffs else None,
        "talpha_abs_diff_max": float(np.max(weight_diffs)) if weight_diffs else None,
        "depth_order_abs_diff_mean": float(np.mean(order_diffs)) if order_diffs else None,
        "unmapped_live_id_count": int(live.get("unmapped_id_count", 0)),
    }


def main():
    safe_state(False)
    summary = read_json(SCRIPT_ARGS.cached_summary)
    frames = [f for f in summary.get("frames", []) if f.get("status") == "ok"]
    frame = frames[min(SCRIPT_ARGS.frame_index, len(frames) - 1)]
    cached_npz_path = frame.get("paths", {}).get("npz")
    cached_npz = np.load(cached_npz_path, allow_pickle=True)
    selected_pixels = np.asarray(cached_npz["selected_pixels"], dtype=np.int64)

    with torch.no_grad():
        dataset = Dataset()
        gaussians = StreetGaussianModel(dataset.scene_info.metadata)
        scene = Scene(gaussians=gaussians, dataset=dataset)
        cameras = {cam.image_name: cam for cam in scene.getTrainCameras()}
        ckpt_path = os.path.join(cfg.trained_model_dir, f"iteration_{cfg.loaded_iter if cfg.loaded_iter != -1 else 5000}.pth")
        state_dict = torch.load(ckpt_path)
        gaussians.load_state_dict(state_dict)
        camera = cameras[str(frame.get("stem"))]
        live = capture_contributions_cuda_live(
            model=gaussians,
            camera=camera,
            renderer=None,
            selected_pixels=selected_pixels,
            top_k=SCRIPT_ARGS.top_k,
        )
    payload = {
        "view_id": str(frame.get("stem")),
        "region_id": str(frame.get("region_id")),
        "cached_npz": cached_npz_path,
        "live_status": live.get("status"),
        "live_cuda_contribution": bool(live.get("live_cuda_contribution", False)),
        "stable_id_map_available": bool(live.get("stable_id_map_available", False)),
        "comparison": compare(cached_npz, live),
        "uses_current_in_memory_model": True,
        "gaussian_parameters_modified": False,
    }
    write_json(SCRIPT_ARGS.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
