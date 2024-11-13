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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import lpips
from utils.external import o3d_knn_loss,compute_euclidean_distance
def lpips_loss(img1, img2, lpips_model):
    loss = lpips_model(img1,img2)
    return loss.mean()
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def KNN_loss(origin_vector_part=None, k=None, lambda_knn_weight=0, cluster_centers_indices=None):
    if origin_vector_part is None or k is None or cluster_centers_indices is None:
        return None

    # Step 1: 从原始数据中提取采样点
    sampled_points = origin_vector_part[cluster_centers_indices].to('cuda')  # 转换为Tensor并传递到GPU
    xyz_coordinates = origin_vector_part.cpu().detach().numpy()
    xyz_coordinates_sample = sampled_points.cpu().detach().numpy()
    
    # Step 2: 使用 Open3D 或其他库计算 KNN
    knn_indices = o3d_knn_loss(xyz_coordinates, xyz_coordinates_sample, k)

    # Convert knn_indices to tensor and move to GPU
    knn_indices = torch.tensor(knn_indices, dtype=torch.long, device='cuda')

    # Step 3: 计算欧式距离差异
    differences = compute_euclidean_distance(sampled_points, origin_vector_part, knn_indices)

    # 计算欧式距离的平方
    o3d_dist_sqrd = torch.sum(differences ** 2, dim=2)  # 形状: (n_clusters, k)

    # Step 4: 计算损失权重
    knn_weights = torch.exp(-lambda_knn_weight * o3d_dist_sqrd)  # 权重通过距离平方的衰减计算

    # Step 5: 计算加权后的 KNN 损失
    knn_loss = torch.norm(differences, dim=2) * knn_weights  # 使用权重调整损失
    total_knn_loss = knn_loss.sum()

    return total_knn_loss

def entropy_loss(opacities):
    """
    Compute entropy loss for opacity values.

    Args:
        opacities (Tensor): A tensor of opacity values of shape (N,).
                            These values should be between 0 and 1.

    Returns:
        Tensor: The entropy loss.
    """
    # Avoid log(0) by clamping the opacities to a minimum value
    opacities = torch.clamp(opacities, min=1e-8)
    
    # Compute entropy loss
    entropy = -opacities * torch.log(opacities)
    
    # Return the mean entropy loss
    return entropy.mean()

def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()