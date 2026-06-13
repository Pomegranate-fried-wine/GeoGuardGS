import math

import torch


def snapshot_camera_transform(camera):
    return {
        "world_view_transform": camera.world_view_transform.detach().clone(),
        "full_proj_transform": camera.full_proj_transform.detach().clone(),
        "camera_center": camera.camera_center.detach().clone(),
    }


def restore_camera_transform(camera, snapshot):
    camera.world_view_transform.data.copy_(snapshot["world_view_transform"])
    camera.full_proj_transform.data.copy_(snapshot["full_proj_transform"])
    camera.camera_center.data.copy_(snapshot["camera_center"])


def _rotation_matrix_xyz(yaw_deg, pitch_deg, roll_deg, device, dtype):
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    roll = math.radians(float(roll_deg))

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    rz = torch.tensor([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
    ry = torch.tensor([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], device=device, dtype=dtype)
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], device=device, dtype=dtype)
    return rz @ ry @ rx


def apply_camera_transform_inplace(
    camera,
    shift=(0.0, 0.0, 0.0),
    rotation_deg=(0.0, 0.0, 0.0),
    mode="camera",
):
    """Apply an explicit novel-view transform to a camera.

    This helper is intentionally not used by train/evaluate paths. Callers must
    opt in from a novel-view script and should restore the camera afterwards if
    the original view will be reused.
    """
    device = camera.world_view_transform.device
    dtype = camera.world_view_transform.dtype
    shift = torch.tensor(shift, device=device, dtype=dtype)
    rotation_deg = tuple(float(x) for x in rotation_deg)

    if mode not in {"camera", "world"}:
        raise ValueError(f"Unsupported camera transform mode: {mode}")

    if mode == "camera":
        right = camera.world_view_transform[:3, 0].detach()
        up = camera.world_view_transform[:3, 1].detach()
        forward = camera.world_view_transform[:3, 2].detach()
        world_shift = right * shift[0] + up * shift[1] + forward * shift[2]
    else:
        world_shift = shift

    if torch.any(world_shift != 0):
        rotation = camera.world_view_transform[:3, :3].detach()
        shift_in_view = torch.matmul(rotation, world_shift)
        camera.world_view_transform.data[:3, 3] -= shift_in_view
        camera.camera_center.data += world_shift

    if any(abs(v) > 1e-12 for v in rotation_deg):
        local_rotation = _rotation_matrix_xyz(*rotation_deg, device=device, dtype=dtype)
        center_before = camera.camera_center.detach().clone()
        camera.world_view_transform.data[:3, :3] = local_rotation @ camera.world_view_transform.data[:3, :3]
        center_after = camera.world_view_transform.inverse()[3, :3]
        center_delta = center_before - center_after
        correction = torch.matmul(camera.world_view_transform[:3, :3].detach(), center_delta)
        camera.world_view_transform.data[:3, 3] -= correction
        camera.camera_center.data.copy_(center_before)

    camera.full_proj_transform.data = torch.matmul(
        camera.world_view_transform,
        camera.projection_matrix,
    ).data

    return {
        "enabled": True,
        "mode": mode,
        "shift": [float(x) for x in shift.detach().cpu().tolist()],
        "rotation_deg": [float(x) for x in rotation_deg],
        "camera_center": [float(x) for x in camera.camera_center.detach().cpu().tolist()],
    }
