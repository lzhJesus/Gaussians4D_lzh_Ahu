#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.external import get_average_depth

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", time = 0,
                 mask = None, depth=None,light_position=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.time = time
        self.light_position = light_position
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        self.original_image = image.clamp(0.0, 1.0)[:3,:,:]
        # breakpoint()
        # .to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask
            # .to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width))
                                                #   , device=self.data_device)
        self.depth = depth
        self.mask = mask
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        # .cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        # .cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    def generate_ray_from_pixel(self, pixel_coords):
        """
        根据像素坐标生成射线。
        Args:
            pixel_coords (tuple): 图像中的 (row, col) 坐标。
        Returns:
            ray_origin (torch.Tensor): 射线的起点（相机中心）。
            ray_direction (torch.Tensor): 射线的方向。
        """
        # 提取像素坐标 (i, j)
        i, j = pixel_coords

        # 计算像素在归一化设备坐标 (NDC) 中的位置
        x_ndc = (j + 0.5) / self.image_width * 2 - 1  # 列 j 对应的 x 坐标 [-1, 1]
        y_ndc = 1 - (i + 0.5) / self.image_height * 2  # 行 i 对应的 y 坐标 [-1, 1]

        # NDC 坐标的 z 值可以设为 -1，表示靠近投影平面的距离
        ndc_coords = torch.tensor([x_ndc, y_ndc, -1.0, 1.0], dtype=torch.float32, device=self.data_device)  # 确保在同一设备上

        # 将投影矩阵移到正确的设备
        inv_proj_matrix = torch.inverse(self.projection_matrix).to(self.data_device)

        # 将 NDC 坐标转换为相机坐标 (通过逆投影矩阵)
        cam_coords = inv_proj_matrix.matmul(ndc_coords)

        # 在相机坐标中进行归一化处理
        cam_coords = cam_coords / cam_coords[-1]  # 确保最后一个分量为 1

        # 将相机坐标转换为世界坐标
        ray_direction = torch.inverse(self.world_view_transform).to(self.data_device).matmul(cam_coords)
        ray_direction = ray_direction[:3]  # 提取方向的 x, y, z 分量
        ray_direction = torch.nn.functional.normalize(ray_direction, dim=0)  # 归一化射线方向

        # 射线起点就是相机的中心坐标
        ray_origin = self.camera_center.to(self.data_device)

        return ray_origin, ray_direction
    
    def depth_to_world(self, pixel_x, pixel_y, depth_map, window_size=3):
        """
        将深度图中的像素坐标转换为世界坐标系下的 3D 坐标点。

        Args:
            pixel_x (int): 像素的 x 坐标。
            pixel_y (int): 像素的 y 坐标。
            depth_map (torch.Tensor): 深度图，形状为 (H, W)。

        Returns:
            torch.Tensor: 对应的世界坐标系下的 3D 坐标点。
        """
        # 获取像素的深度值
        depth_value = get_average_depth(pixel_x, pixel_y, depth_map, window_size)

        if depth_value > 0:
            # 将像素坐标转换为 NDC
            x_ndc = (pixel_x + 0.5) / self.image_width * 2 - 1  # x 坐标从左到右
            y_ndc = 1 - (pixel_y + 0.5) / self.image_height * 2  # y 坐标从上到下，翻转


            # 计算相机坐标
            x_cam = x_ndc * depth_value
            y_cam = y_ndc * depth_value
            z_cam = depth_value

            cam_coords = torch.tensor([x_cam, y_cam, z_cam], dtype=torch.float32, device=self.data_device)

            # 确保 R 和 T 都是张量
            R_tensor = torch.tensor(self.R, dtype=torch.float32, device=self.data_device)
            T_tensor = torch.tensor(self.T, dtype=torch.float32, device=self.data_device)

            # 从相机坐标转换到世界坐标
            world_coordinates = R_tensor @ cam_coords + T_tensor
            
            return world_coordinates
        else:
            return None



class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, time):
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
        self.time = time

