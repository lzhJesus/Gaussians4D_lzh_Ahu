from sklearn.cluster import MeanShift
import torch
import open3d as o3d
import numpy as np 
from scipy.optimize import least_squares
import dgl
from scipy.spatial import cKDTree
def norm_quat(q):
    
    return q / torch.norm(q, dim=-1, keepdim=True)

def o3d_knn(pts, num_knn):
    indices = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for p in pcd.points:
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
        indices.append(i[1:])
        sq_dists.append(d[1:])

    return np.array(sq_dists), np.array(indices)

def o3d_knn_loss(pts, query_pts, num_knn):
    indices = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))  # 原始点云
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    # 针对每个查询点（特征点）在原始点云中寻找最近邻
    for query_p in query_pts:
        [_, i, d] = pcd_tree.search_knn_vector_3d(query_p, num_knn + 1)  # +1 是因为第一个最近邻是点本身
        indices.append(i[1:])  # 忽略第一个，因为它是点自己
    
    return np.array(indices)

def find_neighbors_o3d_kdtree(points, threshold):
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 构建 KDTree
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    neighbors = []
    for point in points:
        # 查找每个点在距离阈值内的邻居
        [_, idx, _] = kdtree.search_radius_vector_3d(point, threshold)
        neighbors.append(idx)
    
    return neighbors

def compute_euclidean_distance(query_tensor, origin_vector_part, knn_indices):
    # 提取邻居点的坐标 (通过索引找到 k 近邻的 3D 坐标)
    neighbors = origin_vector_part[knn_indices]  # 形状: (n_clusters, k, 3)

    # 将 query_tensor 的维度扩展，以便与邻居点匹配
    query_tensor_expanded = query_tensor.unsqueeze(1)  # 形状: (n_clusters, 1, 3)

    # 计算 query_tensor 与邻居点之间的差异
    differences = query_tensor_expanded - neighbors  # 形状: (n_clusters, k, 3)

    return differences


def mean_shift_filtering(points, bandwidth, min_cluster_size):
    """
    使用 Mean Shift 聚类进行点云的噪声过滤。
    
    Args:
        points (np.ndarray): 输入点云 (N, 3)。
        bandwidth (float): Mean Shift 算法中的带宽参数，控制搜索窗口大小。
    
    Returns:
        filtered_points (np.ndarray): 过滤后的点云，去除离群点。
        cluster_centers (np.ndarray): 聚类中心。
    """
    # 创建 Mean Shift 聚类器
    mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    
    # 对点云进行聚类
    mean_shift.fit(points)
    
    # 获取聚类标签
    labels = mean_shift.labels_
    cluster_centers = mean_shift.cluster_centers_

    # 统计每个簇中的点数量
    cluster_sizes = np.bincount(labels)
    
    large_clusters = cluster_sizes >= min_cluster_size
    
    # 生成位置掩码
    mask = np.isin(labels, np.where(large_clusters)[0])
    
    return mask,cluster_centers

def unravel_index(index, shape):
    """
    Args:
        index (int): 一维索引。
        shape (tuple): 张量的形状 (H, W)。

    Returns:
        tuple: 对应于二维张量中的 (row, col) 坐标。
    """
    rows = index // shape[1]  # 计算行坐标
    cols = index % shape[1]   # 计算列坐标
    return rows, cols

def distance_point_to_ray(point, ray_origin, ray_direction):
    """
    Calculate the shortest distance between a point and a ray.

    Parameters:
        point (torch.Tensor): A point in 3D space.
        ray_origin (torch.Tensor): The origin of the ray.
        ray_direction (torch.Tensor): The direction of the ray.

    Returns:
        float: The shortest distance from the point to the ray.
    """
    point_to_origin = point - ray_origin
    proj_length = torch.dot(point_to_origin, ray_direction)
    closest_point = ray_origin + proj_length * ray_direction
    distance = torch.norm(point - closest_point)
    return distance

def find_nearby_points(point_cloud, ray_origin, ray_direction, threshold):
    """
    Find points in the point cloud that are close to the given ray.

    Parameters:
        point_cloud (torch.Tensor): The point cloud, shape (N, 3).
        ray_origin (torch.Tensor): The origin of the ray.
        ray_direction (torch.Tensor): The direction of the ray.
        threshold (float): The distance threshold to consider a point as 'nearby'.

    Returns:
        torch.Tensor: Points in the point cloud that are close to the ray.
    """
    # Compute distances for all points
    point_to_origin = point_cloud - ray_origin
    proj_lengths = torch.matmul(point_to_origin, ray_direction)
    closest_points = ray_origin + proj_lengths.unsqueeze(1) * ray_direction

    distances = torch.norm(point_cloud - closest_points, dim=1)
    
    # Apply threshold to find nearby points
    nearby_points_mask = distances < threshold
    return nearby_points_mask

# 定义一个函数，用于计算给定像素周围的平均深度
def get_average_depth(pixel_x, pixel_y, depth_map, window_size):
    """
    从深度图中获取指定像素周围的平均深度值。
    Args:
        depth_map (torch.Tensor): 深度图，形状为 (H, W)
        pixel_x (int): 像素的x坐标
        pixel_y (int): 像素的y坐标
        window_size (int): 计算平均值的窗口大小 (例如 3 表示3x3窗口)
    Returns:
        float: 周围像素的平均深度值
    """
    # 确定窗口的范围
    if len(depth_map.shape) == 3:
        depth_map = depth_map.squeeze(0)
        
    half_window = window_size // 2
    x_min = max(pixel_x - half_window, 0)
    x_max = min(pixel_x + half_window + 1, depth_map.shape[1])
    y_min = max(pixel_y - half_window, 0)
    y_max = min(pixel_y + half_window + 1, depth_map.shape[0])
    
    # 提取窗口内的深度值
    depth_window = depth_map[y_min:y_max, x_min:x_max]

    # 计算窗口内的平均深度值（排除0或无效的深度值）
    valid_depths = depth_window[depth_window > 0]  # 排除无效的深度值
    if valid_depths.numel() > 0:
        return valid_depths.mean().item()  # 返回有效深度值的平均值
    else:
        return 0  # 如果没有有效的深度值，则返回0 
    

def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def compute_normals(pcd, radius=5, max_nn=30):
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    return np.asarray(pcd.normals)


def estimate_brightness(pcd):
    if not pcd.has_colors():
        raise ValueError("Point cloud has no color information.")
    colors = np.asarray(pcd.colors)
    # 将RGB颜色转换为亮度
    brightness = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]
    return brightness

def residuals(light_direction, normals, brightness):
    light_direction = light_direction / np.linalg.norm(light_direction)
    predicted_brightness = np.dot(normals, light_direction)
    return predicted_brightness - brightness

def estimate_light_direction(normals, brightness):
    initial_guess = np.array([0.0, 0.0, 1.0])  # 初始猜测光源方向
    result = least_squares(residuals, initial_guess, args=(normals, brightness))
    light_direction = result.x / np.linalg.norm(result.x)
    return light_direction

def estimate_light_position(points, normals, brightness):
    weighted_normals = normals * brightness[:, np.newaxis]
    light_position, _, _, _ = np.linalg.lstsq(weighted_normals, points, rcond=None)
    return light_position

def visualize_light_sources(pcd, light_positions, radius=0.5):
    light_spheres = []
    for pos in light_positions:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)  # 创建球体
        sphere.paint_uniform_color([0, 0, 1])  # 蓝色
        sphere.translate(pos)  # 移动到光源位置
        light_spheres.append(sphere)

    return light_spheres

def calculate_weighted_average(light_positions, bounding_box):
    """
    计算光源位置的加权平均。
    
    :param light_positions: 光源位置的数组 (N, 3)
    :param bounding_box: 点云范围的边界盒，格式为 (min_x, max_x, min_y, max_y, min_z, max_z)
    :return: 加权平均后的光源位置
    """
    min_x, max_x, min_y, max_y, min_z, max_z = bounding_box
    weights = []
    
    # 计算每个光源位置到边界盒中心的距离
    center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2])
    for position in light_positions:
        distance = np.linalg.norm(position - center)
        weight = 1 / (distance + 1e-6)  # 防止除以0，加1e-6以确保权重正常
        weights.append(weight)
    
    weights = np.array(weights)
    weighted_sum = np.dot(weights, light_positions)  # 加权和
    weighted_average = weighted_sum / np.sum(weights)  # 加权平均
    
    return weighted_average

def light_estimation(file_path):
    
    pcd = load_point_cloud(file_path)
    normals = compute_normals(pcd)
    brightness = estimate_brightness(pcd)
    points = np.asarray(pcd.points)
    light_position = estimate_light_position(points, normals, brightness)
    bounding_box = (-25, 20, -20, 20, 10, 20)
    weighted_average_position = calculate_weighted_average(light_position, bounding_box)
    return weighted_average_position

def lifecycle_function(G_i, t, Do):
    input_data = torch.cat([G_i, t], dim=-1)
    delta_o = Do(input_data)
    lifecycle = torch.sigmoid(delta_o)
    return lifecycle

def compute_angle_between_vectors(v1, v2):
    dot_product = torch.sum(v1 * v2, dim=-1)  # 点积
    norms = torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1)  # 向量的模
    cos_theta = dot_product / norms
    return torch.acos(torch.clamp(cos_theta, min=-1.0, max=1.0))

def create_keypoint(point_tensor1, point_tensor2, point_tensor3, coordinates_input_tensor):
    # 计算速度（假设时间间隔为1秒，简单计算速度差）
    velocity = (point_tensor2 - point_tensor1) / 0.11  # 速度 = 位移 / 时间间隔
    velocity2 = (point_tensor3 - point_tensor2) / 0.11  # 第二个时间点到第三个时间点的速度
    acceleration = velocity2 - velocity  # 加速度 = 速度变化

    # 计算相邻速度向量之间的角度变化
    direction_change = compute_angle_between_vectors(velocity[:-1], velocity[1:])

    # 将速度（magnitude）和加速度存储在一个(N, 3)形状的张量中
    result = torch.zeros((point_tensor1.shape[0], 3))

    # 计算速度的大小并保存在result[:, 0]中
    result[:, 0] = torch.norm(velocity, dim=-1)

    # 计算加速度的大小并保存在result[:, 1]中
    result[:, 1] = torch.norm(acceleration, dim=-1)

    # 将方向变化存储在result的第三列中
    # 对于方向变化，结果只有N-1个数据，因此需要对result进行填充
    result[:-1, 2] = direction_change

    # 计算速度、加速度和方向变化的均值和标准差
    velocity_mean = result[:, 0].mean().item()
    velocity_std = result[:, 0].std().item()
    acceleration_mean = result[:, 1].mean().item()
    acceleration_std = result[:, 1].std().item()
    direction_change_mean = result[:, 2].mean().item()
    direction_change_std = result[:, 2].std().item()

    # 设置阈值，标记速度、加速度或方向变化超出阈值的点为关键点
    velocity_threshold = velocity_mean + 2 * velocity_std  # 设定阈值：速度大于均值+2倍标准差
    acceleration_threshold = acceleration_mean + 2 * acceleration_std  # 加速度阈值
    direction_change_threshold = direction_change_mean + 2 * direction_change_std  # 方向变化阈值

    # 检测关键点
    key_points = torch.zeros(point_tensor1.shape[0], dtype=torch.bool)

    # 根据速度、加速度和方向变化来标记关键点
    key_points |= result[:, 0] > velocity_threshold  # 标记速度超过阈值的点
    key_points |= result[:, 1] > acceleration_threshold  # 标记加速度超过阈值的点
    key_points[:-1] |= result[:-1, 2] > direction_change_threshold  # 标记方向变化超过阈值的点

    # 打印关键点的索引
    key_point_indices = torch.nonzero(key_points).squeeze()
    key_point_coordinates = coordinates_input_tensor[key_point_indices]
    # 创建 Open3D 点云对象
    point_cloud = o3d.geometry.PointCloud()
    key_point_coordinates_np = key_point_coordinates.cpu().numpy()
    # 将关键点坐标添加到点云
    point_cloud.points = o3d.utility.Vector3dVector(key_point_coordinates_np)
    # 保存点云为 .ply 文件
    o3d.io.write_point_cloud("key_points.ply", point_cloud)
    print("key_points.ply created")
    # 剩余的普通点
    non_key_point_indices = torch.nonzero(~key_points).squeeze()
    return key_point_indices, non_key_point_indices

def guide_error_prune(guide_cams, error_images, opt):
    nearby_points_masks = []
    final_cam_nearby = []
    final_error_points_mask = None
    while len(guide_cams) > 0:
        all_error_coords = []
        error_image_name = error_images.pop()
        image, viewpoint_cam, means_3D_deform_tensor= guide_cams.pop(error_image_name)
        gt_image = viewpoint_cam.original_image.to('cuda:0')
        # 计算调整后的图像和差异
        imageadjust = image / (torch.mean(image) + 0.01)
        gtadjust = gt_image / (torch.mean(gt_image) + 0.01)
        diff = torch.abs(imageadjust - gtadjust)  # 差异计算

        # 计算每个像素的绝对差异
        diff_sum = torch.sum(diff, dim=0)  # h, w

        # 获取前128个差异最大的像素
        diff_flat = diff_sum.view(-1)
        top_k_values, top_k_indices = torch.topk(diff_flat, 128)

        # 计算全局坐标
        error_coords = [(index // diff_sum.shape[0], index % diff_sum.shape[1]) for index in top_k_indices.cpu().numpy()]

        # 记录所有差异坐标
        all_error_coords.extend(error_coords)
        for coords in all_error_coords:
            ray_origin, ray_direction = viewpoint_cam.generate_ray_from_pixel(coords)
            error_points_mask = find_nearby_points(means_3D_deform_tensor, ray_origin, ray_direction, opt.error_threshold)
            if error_points_mask is not None:
                nearby_points_masks.append(error_points_mask)
        distances = torch.norm(means_3D_deform_tensor - viewpoint_cam.camera_center.to('cuda:0'), dim=1)
        cam_nearby_mask = distances < opt.cam_distance_threshold
        final_cam_nearby.append(cam_nearby_mask)

    if final_cam_nearby and nearby_points_masks :
        final_cam_nearby_mask = torch.any(torch.stack(final_cam_nearby), dim=0)
        final_error_points_mask = torch.any(torch.stack(nearby_points_masks), dim=0)
        final_error_points_mask = final_error_points_mask | final_cam_nearby_mask
    return final_error_points_mask