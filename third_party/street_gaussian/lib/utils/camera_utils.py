import numpy as np
import torch
import copy
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from lib.utils.general_utils import PILtoTorch, NumpytoTorch
from lib.utils.graphics_utils import fov2focal, getProjectionMatrix, getWorld2View2, getProjectionMatrixK
from lib.datasets.base_readers import CameraInfo
from lib.config import cfg

class Camera(nn.Module):
    def __init__(
        self, 
        id,
        R, T, 
        FoVx, FoVy, K,
        image, image_name, 
        trans = np.array([0.0, 0.0, 0.0]), 
        scale = 1.0,
        metadata = dict(),
        guidance=dict(),
    ):
        super(Camera, self).__init__()

        self.id = id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.K_orig = K # 备份原始K
        self.image_name = image_name
        self.trans, self.scale = trans, scale

        # metadata & guidance
        self.meta = metadata
        self.guidance = guidance
        self.original_image = image.clamp(0.0, 1.0)
        
        self.image_height, self.image_width = self.original_image.shape[1], self.original_image.shape[2]
        self.zfar = 1000.0
        self.znear = 0.001

        # 核心修改：使用 .to("cuda") 代替 .cuda()，确保动态适配指定的显卡
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to("cuda")
        
        if K is not None:
            self.projection_matrix = getProjectionMatrixK(
                znear=self.znear, zfar=self.zfar, K=K, H=self.image_height, W=self.image_width
            ).transpose(0, 1).to("cuda")
            self.K = torch.from_numpy(K).float().to("cuda")
        else:
            self.projection_matrix = getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            ).transpose(0, 1).to("cuda")

        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        # 处理位姿元数据
        if 'ego_pose' in self.meta:
            self.ego_pose = torch.from_numpy(self.meta['ego_pose']).float().to("cuda")
        if 'extrinsic' in self.meta:
            self.extrinsic = torch.from_numpy(self.meta['extrinsic']).float().to("cuda")

    def set_device(self, device):
        self.original_image = self.original_image.to(device)
        for k, v in self.guidance.items():
            self.guidance[k] = v.to(device, non_blocking=True)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

def loadguidance(guidance, resolution):
    new_guidance = dict()
    for k, v in guidance.items():
        # 统一处理各类 Mask 和 Depth
        if k in ['mask', 'acc_mask', 'sky_mask', 'obj_bound']:
            new_guidance[k] = PILtoTorch(v, resolution, resize_mode=Image.NEAREST).bool()
        elif k == 'lidar_depth':
            new_guidance[k] = NumpytoTorch(v, resolution, resize_mode=Image.NEAREST).float()
    return new_guidance

def loadCam(cam_info: CameraInfo, resolution_scale, scale=1.0):
    orig_w, orig_h = cam_info.width, cam_info.height
    scale = min(scale, 1600 / orig_w) / resolution_scale
    resolution = (int(orig_w * scale), int(orig_h * scale))

    K = copy.deepcopy(cam_info.K)
    K[:2] *= scale

    image = PILtoTorch(cam_info.image, resolution, resize_mode=Image.BILINEAR)[:3, ...]
    guidance = loadguidance(cam_info.guidance, resolution)

    return Camera(
        id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
        FoVx=cam_info.FovX, FoVy=cam_info.FovY, K=K,
        image=image, image_name=cam_info.image_name,
        metadata=cam_info.metadata, guidance=guidance
    )

def cameraList_from_camInfos(cam_infos, resolution_scale):
    camera_list = []
    for i, cam_info in tqdm(enumerate(cam_infos), desc="Loading Cameras"):
        camera_list.append(loadCam(cam_info, resolution_scale))
    return camera_list

def make_rasterizer(viewpoint_camera: Camera, active_sh_degree=0, bg_color=None, scaling_modifier=None):
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    import math

    if bg_color is None:
        bg_color = [1, 1, 1] if cfg.data.white_background else [0, 0, 0]
    
    # 关键：动态获取 device，避免硬编码 GPU 0
    device = viewpoint_camera.world_view_transform.device
    bg_color = torch.tensor(bg_color).float().to(device)
    
    if scaling_modifier is None:
        scaling_modifier = cfg.render.scaling_modifier

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=cfg.render.debug,
    )
    return GaussianRasterizer(raster_settings=raster_settings)
def camera_to_JSON(id, camera: CameraInfo):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry