#!/usr/bin/env python3
"""从spconv2 LiDAR backbone末端构建初始3D Gaussian锚点。

该脚本负责验证固定初始Gaussian的构建及高度感知Gaussian-to-BEV：

1. 按正式训练pipeline加载六路增强图像和多帧聚合点云；
2. 点云经过Spconv2Voxelization和SparseEncoderSpconv2，保留conv_out；
3. 将conv_out非空稀疏坐标转换为增强后LiDAR坐标系中的物理中心；
4. 使用归一化XYZ空间中的最远点采样，从全部候选中选择N个代表性锚点；
5. 将锚点投影到六路相机，从Camera FPN特征图grid_sample图像特征；
6. 多相机重叠区域对全部有效相机特征取均值；尺度采用conv_out输出格点间距；
7. 旋转初始化为单位四元数，不透明度初始化为指定的固定值。
8. 在DAOcc的10×180×180三维网格中执行局部Gaussian splatting，并按原模型
   的高度层顺序折叠为1280×180×180相机BEV特征。

脚本输出JSON统计和可视化图片，不保存锚点PTH；当前用途是验证初始化方法、
相机可见性、空间分布和各阶段耗时。

示例：
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python \
    tools/build_initial_gaussian_anchors.py \
    configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_inin_spconv2_voxel.yaml \
    --gpu 0 --num-samples 20 --num-gaussians 1200 --history-sweeps 9 \
    --checkpoint /path/to/epoch_1.pth --visualize \
    --output-dir /tmp/initial_gaussian_anchors
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import spconv.pytorch as spconv

# 复用现有独立脚本中已经验证过的配置递归加载、正式Dataset pipeline、
# checkpoint提取以及新旧spconv卷积核布局转换，避免出现两套不同的数据路径。
import visualize_camera_bev_pipeline as camera_pipeline
import visualize_spconv2_lidar_pipeline as lidar_pipeline

from mmdet3d.datasets import build_dataset
from mmdet3d.models.builder import build_fuser
from mmdet3d.ops import furthest_point_sample
from ops.gaussian_voxel_splat import gaussian_voxel_splat


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="同时包含camera和spconv2 lidar encoder的训练配置")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="从基础训练数据集中均匀抽取的样本数")
    parser.add_argument("--num-gaussians", type=int, default=1200, help="选取的Gaussian数量")
    parser.add_argument("--history-sweeps", type=int, help="覆盖训练pipeline中的历史帧数")
    parser.add_argument("--warmup-iters", type=int, default=3,
                        help="正式计时前完整预热次数，默认3")
    parser.add_argument("--scale-factor", type=float, default=1.0,
                        help="固定初始尺度相对conv_out格点间距的倍率")
    parser.add_argument("--initial-opacity", type=float, default=1.0,
                        help="初始Gaussian不透明度，取值范围(0,1]，默认1.0")
    parser.add_argument("--splat-sigma", type=float, default=3.0,
                        help="每个Gaussian参与splatting的标准差截断半径，默认3")
    parser.add_argument("--splat-chunk-size", type=int, default=64,
                        help="Gaussian-to-BEV分块大小，用于限制临时显存，默认64")
    parser.add_argument(
        "--splat-accumulation", choices=("normalized", "sum"),
        default="normalized",
        help="BEV网格使用归一化加权或直接加权求和，默认normalized",
    )
    parser.add_argument(
        "--splat-implementation", choices=("cuda", "pytorch"), default="cuda",
        help="Gaussian体素聚合实现；默认使用localagg框架改写的CUDA算子",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path,
                        help="可选完整模型权重，同时提取camera与lidar encoder参数")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visualize", action="store_true", help="保存锚点及相机投影可视化")
    args, overrides = parser.parse_known_args()
    if args.num_samples <= 0:
        parser.error("--num-samples必须为正整数")
    if args.num_gaussians <= 0:
        parser.error("--num-gaussians必须为正整数")
    if args.history_sweeps is not None and args.history_sweeps < 0:
        parser.error("--history-sweeps必须为非负整数")
    if args.warmup_iters < 0:
        parser.error("--warmup-iters必须为非负整数")
    if args.scale_factor <= 0:
        parser.error("--scale-factor必须大于0")
    if not 0.0 < args.initial_opacity <= 1.0:
        parser.error("--initial-opacity必须位于(0,1]范围内")
    if args.splat_sigma <= 0:
        parser.error("--splat-sigma必须大于0")
    if args.splat_chunk_size <= 0:
        parser.error("--splat-chunk-size必须为正整数")
    return args, overrides


def build_dataset_for_uniform_sampling(cfg, args):
    """构建训练集并去除CBGS等外层包装，返回基础数据集及历史帧配置。"""
    history_sweeps = camera_pipeline.configure_history_sweeps(
        cfg.data.train, args.history_sweeps
    )
    wrapped = build_dataset(cfg.data.train)
    dataset = camera_pipeline.unwrap_dataset_for_uniform_sampling(wrapped)
    if args.num_samples > len(dataset):
        raise ValueError(
            f"num-samples={args.num_samples}超过基础训练数据集长度{len(dataset)}"
        )
    selected_indices = camera_pipeline.uniform_dataset_indices(
        len(dataset), args.num_samples
    )
    return wrapped, dataset, history_sweeps, selected_indices


@torch.no_grad()
def run_camera_fpn(images, modules):
    """只运行图像backbone和neck，保留逐相机透视特征图，不生成原Camera BEV。"""
    batch, cameras, channels, height, width = images.shape
    if batch != 1:
        raise ValueError("当前构建脚本一次只处理一个样本")
    feature = modules["backbone"](
        images.reshape(batch * cameras, channels, height, width)
    )
    feature = modules["neck"](feature)
    if not isinstance(feature, torch.Tensor):
        feature = feature[0]
    _, feature_channels, feature_height, feature_width = feature.shape
    return feature.reshape(
        batch, cameras, feature_channels, feature_height, feature_width
    ).contiguous()


@torch.no_grad()
def run_original_camera_bev(camera_features, points, matrices, metas, modules):
    """从已提取的FPN特征执行原BEVTransformV2，避免重复运行图像backbone。"""
    feature = modules["vtransform"](
        camera_features,
        [points],
        matrices["camera2ego"],
        matrices["lidar2ego"],
        matrices["lidar2camera"],
        matrices["lidar2image"],
        matrices["camera_intrinsics"],
        matrices["camera2lidar"],
        matrices["img_aug_matrix"],
        matrices["lidar_aug_matrix"],
        [metas],
    )
    if "bev_backbone" in modules:
        feature = modules["bev_backbone"](feature)
    if "bev_neck" in modules:
        feature = modules["bev_neck"](feature)
    return feature


@torch.no_grad()
def run_lidar_conv_out(points, voxelizer, backbone):
    """复现正式LiDAR分支，返回conv_out稀疏结果及最终二维LiDAR BEV。

    记号：N为输入点数，V为初始非空体素数，P为单体素最多保存点数，
    F为输入点特征维数，M为conv_out非空位置数，C为conv_out通道数。
    本独立脚本一次只处理一个样本，所以这里的batch size固定为B=1。
    """
    #* ==================== 1. spconv2点云体素化与体素内均值聚合 ====================
    # points：[N,F]，每行通常为[x,y,z,intensity,time_lag]。
    # PointToVoxel只处理当前一个样本，不会自动增加batch维或batch坐标。
    voxels, coordinates_xyz, num_points = voxelizer(points)
    # voxels：[V,P,F]，每个非空体素最多保存P个点，不足的位置补0。
    # coordinates_xyz：[V,3]，每个非空体素的整数网格索引[x,y,z]。
    # num_points：[V]，各体素实际有效点数，用于排除padding对均值的影响。
    voxel_features = (
        voxels.sum(dim=1) / num_points.type_as(voxels).view(-1, 1)
    ).contiguous()  # [V,F]：每个非空体素对应一个平均点特征。

    #* ==================== 2. 补batch索引并构造spconv2稀疏输入 ====================
    # DAOcc包装器输出[x,y,z]，spconv2 SparseConvTensor要求[batch,z,y,x]。
    # 当前函数只处理一个样本，因此新增的batch索引全部为0。
    batch = torch.zeros(
        (len(coordinates_xyz), 1), dtype=torch.int32, device=points.device
    )  # [V,1]：当前所有体素均属于batch 0。
    coordinates_bxyz = torch.cat([batch, coordinates_xyz.int()], dim=1)  # [V,4]：[batch,x,y,z]。
    coordinates_bzyx = coordinates_bxyz[:, [0, 3, 2, 1]].contiguous()  # [V,4]：[batch,z,y,x]，spconv2内部坐标顺序。
    sparse = spconv.SparseConvTensor(voxel_features, coordinates_bzyx, backbone.sparse_shape_spconv2, 1)
    # sparse.features：[V,F]；sparse.indices：[V,4]；
    # sparse.spatial_shape：[Z,Y,X]，最后一个参数1表示batch size为1。

    #* ==================== 3. 执行spconv2稀疏卷积backbone ====================
    # 稀疏卷积只在非空位置保存特征和坐标；下采样过程中非空位置数及通道数
    # 会逐层变化。conv_out之后仍是SparseConvTensor，并非点云或稠密BEV。
    x = backbone.conv_input(sparse)  # SparseConvTensor：初始稀疏卷积特征。
    for layer in backbone.encoder_layers:
        x = layer(x)  # SparseConvTensor：逐阶段编码、增加通道并下采样。
    output = backbone.conv_out(x)  # SparseConvTensor：backbone末端稀疏输出。
    # output.features：[M,C]，M个末端非空位置各自对应一个C维LiDAR特征。
    # output.indices：[M,4]，对应坐标为[batch,z,y,x]。
    # output.spatial_shape：[D,H,W]，当前配置通常为[2,180,180]。

    #* ==================== 4. 提取末端稀疏坐标用于Gaussian初始化 ====================
    # 去掉batch列，并将spconv2的[z,y,x]恢复为便于换算物理中心的[x,y,z]。
    coordinates_output_xyz = output.indices[
        :, [3, 2, 1]
    ].int().contiguous()  # [M,3]：conv_out非空位置的整数[x,y,z]索引。

    #* ==================== 5. 稠密化、高度折叠并生成LiDAR BEV ====================
    # dense()按照output.spatial_shape把未保存的空位置补0，输出标准spconv2排列。
    spatial_features = output.dense()  # [B,C,D,H,W]=[B,C,Z,Y,X]。
    n, c, d, h, w = spatial_features.shape  # B、C、Z、Y、X的实际尺寸。
    lidar_bev_features = spatial_features.reshape(
        n, c * d, h, w
    )  # [B,C*Z,Y,X]：将高度Z拼入通道，不对高度求和或平均。
    # DAOcc旧SparseEncoder、BEVTransformV2和ConvFuser内部沿用[X,Y]空间排列，
    # 因此交换最后两维，使spconv2输出与原模型保持一致。
    lidar_bev_features = lidar_bev_features.permute(
        0, 1, 3, 2
    ).contiguous()  # [B,C*Z,X,Y]，当前配置通常为[1,256,180,180]。

    #* ==================== 6. 返回稀疏锚点信息及稠密BEV ====================
    return {
        "input_voxel_count": int(len(coordinates_xyz)),  # 标量：初始非空体素数V。
        "coordinates_xyz": coordinates_output_xyz,  # [M,3]：末端非空位置XYZ索引。
        "features": output.features,  # [M,C]：末端非空位置LiDAR特征。
        "spatial_shape_zyx": [
            int(value) for value in output.spatial_shape
        ],  # 长度3的列表：[Z,Y,X]稀疏空间尺寸。
        "dense_3d_features": spatial_features,  # [B,C,Z,Y,X]稠密三维特征。
        "bev_features": lidar_bev_features,  # [B,C*Z,X,Y]，供后续融合使用。
    }


def load_fuser_checkpoint(fuser, checkpoint_path):
    """从完整DAOcc checkpoint中只提取fuser参数。"""
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    prefixes = ("module.fuser.", "fuser.")
    extracted = {}
    for name, value in state_dict.items():
        for prefix in prefixes:
            if name.startswith(prefix):
                extracted[name[len(prefix):]] = value
                break
    if not extracted:
        raise KeyError("checkpoint中未找到fuser参数")
    incompatible = fuser.load_state_dict(extracted, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "fuser权重不匹配：missing=%s, unexpected=%s"
            % (incompatible.missing_keys, incompatible.unexpected_keys)
        )


def build_fuser_module(cfg, device, checkpoint_path=None):
    """构建与正式DAOcc相同的ConvFuser，并可加载训练权重。"""
    if cfg.model.get("fuser") is None:
        raise ValueError("当前配置没有fuser，无法比较多模态BEV融合")
    fuser = build_fuser(cfg.model.fuser).to(device).eval()
    if checkpoint_path is not None:
        load_fuser_checkpoint(fuser, checkpoint_path.expanduser().resolve())
    return fuser


def _as_xyz(value):
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError(f"三维卷积参数长度应为3，实际为{value}")
        return np.asarray(value, dtype=np.float64)
    return np.asarray([value, value, value], dtype=np.float64)


def conv_out_lattice_geometry(backbone_cfg, voxel_cfg):
    """计算conv_out索引对应的物理中心起点和相邻格点间距。

    这里跟踪旧DAOcc配置采用的[X,Y,Z]卷积参数。offset以输入体素索引为单位，
    初始0.5表示原体素中心；每次普通稀疏下采样按kernel/stride/padding更新。
    SubMConv和残差块不改变输出坐标格点，因此不改变offset与step。
    """
    offset = np.full(3, 0.5, dtype=np.float64)
    step = np.ones(3, dtype=np.float64)

    def apply_conv(kernel, stride, padding):
        nonlocal offset, step
        kernel = _as_xyz(kernel)
        stride = _as_xyz(stride)
        padding = _as_xyz(padding)
        offset += ((kernel - 1.0) / 2.0 - padding) * step
        step *= stride

    block_type = backbone_cfg.get("block_type", "conv_module")
    channels = backbone_cfg.encoder_channels
    paddings = backbone_cfg.encoder_paddings
    for stage_index, blocks in enumerate(channels):
        if block_type == "basicblock":
            # SparseEncoder basicblock在除最后stage外的最后一个block执行stride=2。
            if stage_index != len(channels) - 1:
                apply_conv(3, 2, paddings[stage_index][-1])
        elif stage_index != 0:
            # conv_module在除第一stage外的第一个block执行stride=2。
            apply_conv(3, 2, paddings[stage_index][0])

    # SparseEncoder默认conv_out：旧配置XYZ顺序下kernel=(1,1,3)。
    conv_out_stride = backbone_cfg.get("conv_out_stride", (1, 1, 2))
    apply_conv((1, 1, 3), conv_out_stride, 0)

    voxel_size = np.asarray(voxel_cfg.voxel_size, dtype=np.float64)
    minimum = np.asarray(voxel_cfg.point_cloud_range[:3], dtype=np.float64)
    return {
        "origin_xyz": minimum + offset * voxel_size,
        "step_xyz": step * voxel_size,
        "offset_in_input_voxels_xyz": offset,
        "stride_in_input_voxels_xyz": step,
    }


def sparse_indices_to_centers(indices_xyz, lattice):
    origin = torch.as_tensor(
        lattice["origin_xyz"], dtype=torch.float32, device=indices_xyz.device
    )
    step = torch.as_tensor(
        lattice["step_xyz"], dtype=torch.float32, device=indices_xyz.device
    )
    return origin + indices_xyz.float() * step


def farthest_point_sample_xyz(centers_xyz, count, point_range):
    """使用mmdet3d已编译CUDA算子，在归一化XYZ空间执行FPS。"""
    total = len(centers_xyz)
    if count > total:
        raise ValueError(f"请求{count}个Gaussian，但可见候选只有{total}个")
    if count == total:
        return torch.arange(total, device=centers_xyz.device, dtype=torch.long)

    lower = centers_xyz.new_tensor(point_range[:3], dtype=torch.float32)
    upper = centers_xyz.new_tensor(point_range[3:], dtype=torch.float32)
    normalized = (
        (centers_xyz.detach().float() - lower) / (upper - lower).clamp_min(1e-6)
    )
    # CUDA接口输入为[B,N,3]，输出为int32的[B,count]候选局部索引。
    return furthest_point_sample(
        normalized[None].contiguous(), count
    )[0].long()


@torch.no_grad()
def project_and_sample_image_features(
    centers_xyz, camera_features, matrices, input_height, input_width
):
    """把增强后LiDAR坐标中的锚点投影到相机并采样FPN特征。

    几何步骤与BEVTransformV2.point_sampling一致：先撤销lidar_aug_matrix，
    再由camera2lidar反变换到相机坐标，应用相机内参和img_aug_matrix。

    记号：G为候选Gaussian数量，N为相机数量，C为FPN通道数，Hf/Wf为
    FPN特征图尺寸；input_height/input_width是增强后输入图像尺寸而非Hf/Wf。
    """

    #* ==================== 1. 检查输入并确定相机数量 ====================
    # centers_xyz：[G,3]，候选中心位于“经过3D数据增强后的LiDAR坐标系”。
    # camera_features：[1,N,C,Hf,Wf]，当前独立脚本一次只处理一个样本；
    # N通常为6，C通常为256，Hf/Wf通常为32/88。
    if camera_features.shape[0] != 1:
        raise ValueError("当前脚本一次只支持一个样本")
    device = centers_xyz.device
    cameras = camera_features.shape[1]  # N：环视相机数量。

    #* ==================== 2. 撤销LiDAR三维数据增强 ====================
    # 训练pipeline可能对点云及3D标注施加旋转、缩放和平移，候选中心继承了
    # 该增强。相机外参描述的是增强前几何关系，因此投影前先求解：
    # p_original = R_aug^{-1} * (p_augmented - t_aug)。
    bda = matrices["lidar_aug_matrix"][0]  # [4,4]：当前样本的3D增强矩阵。
    centers_unaugmented = torch.linalg.solve(
        bda[:3, :3].float(),
        (centers_xyz - bda[:3, 3].float()).T,
    ).T  # [G,3]：恢复到增强前LiDAR坐标系的候选中心。

    #* ==================== 3. 从LiDAR坐标转换到每路相机坐标 ====================
    # camera2lidar给出p_lidar = R * p_camera + t。这里不显式求完整4×4逆矩阵，
    # 而利用刚体旋转R^{-1}=R^T计算p_camera = R^T * (p_lidar - t)。
    camera2lidar = matrices["camera2lidar"][0].float()  # [N,4,4]。
    rotation = camera2lidar[:, :3, :3]  # [N,3,3]：camera到LiDAR旋转。
    translation = camera2lidar[:, :3, 3]  # [N,3]：camera在LiDAR系中的平移。
    relative = (
        centers_unaugmented[None, :, :] - translation[:, None, :]
    )  # [N,G,3]：每个候选相对各相机原点的位置。
    points_camera = torch.matmul(
        rotation.transpose(-1, -2)[:, None, :, :], relative[..., None]
    ).squeeze(-1)  # [N,G,3]：候选中心在各相机坐标系中的XYZ。
    depths = points_camera[..., 2]  # [N,G]：相机光轴深度，必须为正才在相机前方。

    #* ==================== 4. 透视投影到原始图像像素平面 ====================
    # 使用相机内参得到齐次图像坐标[u*d,v*d,d]，再除以深度得到像素(u,v)。
    # 深度为负的点随后会被valid剔除；clamp只用于避免此处发生除零。
    intrinsics = matrices[
        "camera_intrinsics"
    ][0, :, :3, :3].float()  # [N,3,3]：每路相机内参K。
    projected = torch.matmul(
        intrinsics[:, None], points_camera[..., None]
    ).squeeze(-1)  # [N,G,3]：齐次图像坐标[u*d,v*d,d]。
    uv = (
        projected[..., :2] / projected[..., 2:3].clamp_min(1e-5)
    )  # [N,G,2]：增强前原始图像上的像素坐标(u,v)。

    #* ==================== 5. 应用二维图像数据增强 ====================
    # Camera FPN处理的是resize/crop/flip等增强后的图像，因此还需用
    # img_aug_matrix把原始像素坐标同步变换到增强后输入图像坐标系。
    image_aug = matrices["img_aug_matrix"][0].float()  # [N,4,4]。
    uv = torch.matmul(
        image_aug[:, None, :2, :2], uv[..., None]
    ).squeeze(-1) + image_aug[:, None, :2, 3]  # [N,G,2]：增强后像素坐标。

    #* ==================== 6. 判断候选在各相机中是否可见 ====================
    # 这里只判断“位于相机前方且投影落在图像边界内”，属于视锥可见性；
    # 没有利用深度图检查候选是否被车辆、建筑等物体遮挡。
    valid = (
        (depths > 1e-5)
        & (uv[..., 0] > 0)
        & (uv[..., 0] < input_width)
        & (uv[..., 1] > 0)
        & (uv[..., 1] < input_height)
    )  # [N,G]：相机-候选可见性布尔掩码。

    #* ==================== 7. 从每路相机FPN特征图进行双线性采样 ====================
    # grid_sample要求采样坐标位于[-1,1]。此处沿用BEVTransformV2的归一化
    # 和align_corners设置，使同一增强后像素位置映射到Camera FPN特征图。
    grid = uv.clone()  # [N,G,2]，最后一维顺序必须是[x(u),y(v)]。
    grid[..., 0] = 2.0 * grid[..., 0] / float(input_width) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / float(input_height) - 1.0
    feature = camera_features[0]  # [N,C,Hf,Wf]：把相机维当作grid_sample的batch。
    sampled = F.grid_sample(
        feature,
        grid.view(cameras, -1, 1, 2),  # [N,G,1,2]：每路相机采样G个位置。
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # [N,C,G,1]：每路相机中各候选对应的C维图像特征。
    sampled = (
        sampled.squeeze(-1).permute(0, 2, 1).contiguous()
    )  # [N,G,C]：整理为[相机,Gaussian,特征]。

    #* ==================== 8. 聚合多相机特征并整理输出 ====================
    # 第一版没有学习相机权重：先把无效视角特征清零，再对每个候选的全部
    # 有效相机特征取算术平均。注意：BEVTransformV2在重叠区域选第一个有效
    # 相机，而这里使用多视角均值，两者几何投影一致但跨相机聚合策略不同。
    sampled = sampled * valid[..., None].type_as(sampled)  # [N,G,C]。
    visibility_count = valid.sum(dim=0)  # [G]：每个候选可见相机数量。
    gaussian_features = (
        sampled.sum(dim=0) / visibility_count.clamp_min(1)[:, None]
    )  # [G,C]：每个候选最终聚合得到的图像特征；完全不可见候选保持全0。
    return {
        "features": gaussian_features,  # [G,C]：多视角均值Gaussian图像特征。
        "visibility_mask": valid.T.contiguous(),  # [G,N]：各相机是否可见。
        "visibility_count": visibility_count,  # [G]：可见相机数量。
        "camera_depths": depths.T.contiguous(),  # [G,N]：各相机坐标系深度。
        "image_uv": uv.permute(
            1, 0, 2
        ).contiguous(),  # [G,N,2]：增强后图像像素坐标(u,v)。
    }


def quaternion_wxyz_to_rotation_matrix(quaternions):
    """把[w,x,y,z]四元数转换为旋转矩阵；输入会先归一化。"""
    q = F.normalize(quaternions.float(), dim=-1, eps=1e-8)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
            2 * (x * z + w * y), 2 * (x * y + w * z),
            1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


@torch.no_grad()
def map_gaussian_features_to_bev_channels(features, vtransform):
    """复用BEVTransformV2训练好的1×1 transfer_conv，将256维映射到128维。"""
    transfer = vtransform.transfer_conv
    if transfer is None:
        return features
    weight = transfer.weight[:, :, 0, 0]
    return F.linear(features, weight, transfer.bias)


@torch.no_grad()
def _splat_gaussians_to_height_aware_bev_pytorch(
    centers, scales, rotations, opacities, features, vtransform,
    sigma_limit=3.0, chunk_size=64, accumulation="normalized",
):
    """把稀疏3D Gaussian局部splat到规则体素，再把高度层拼入BEV通道。

    先得到物理排列[B,C,Z,Y,X]，高度折叠后为[B,Z*C,Y,X]，最后转换为
    DAOcc内部融合使用的[B,Z*C,X,Y]。这里只枚举每个Gaussian的局部邻域，
    避免直接构造“Gaussian数量×全部体素数量”的巨大稠密矩阵。
    """

    #* ==================== 1. 建立目标三维规则网格 ====================
    # volume_size按[X,Y,Z]记录网格数量，pc_range按
    # [xmin,ymin,zmin,xmax,ymax,zmax]记录物理范围。由此计算每个体素的
    # XYZ尺寸及第一个体素中心；当前配置对应180×180×10个三维体素。
    x_size, y_size, z_size = [int(value) for value in vtransform.volume_size]
    minimum = centers.new_tensor(vtransform.pc_range[:3])
    maximum = centers.new_tensor(vtransform.pc_range[3:])
    grid_shape_xyz = torch.tensor(
        [x_size, y_size, z_size], dtype=torch.long, device=centers.device
    )
    grid_step = (maximum - minimum) / grid_shape_xyz.type_as(centers)
    grid_first_center = minimum + 0.5 * grid_step

    #* ==================== 2. 准备全局累加缓冲区和旋转矩阵 ====================
    # 为每个三维体素保存“Gaussian权重×特征”的累加结果及总权重。
    # 体素暂时展平成一维，便于后面使用index_add_把不同Gaussian的贡献
    # 汇聚到同一个体素。四元数则预先转换为3×3旋转矩阵，避免分块中重复计算。
    channels = features.shape[1]
    total_voxels = x_size * y_size * z_size
    feature_sum = features.new_zeros((total_voxels, channels))
    weight_sum = features.new_zeros((total_voxels, 1))
    rotations_matrix = quaternion_wxyz_to_rotation_matrix(rotations)

    #* ==================== 3. 分块处理Gaussian以控制临时显存 ====================
    # 每个Gaussian只影响自身附近的体素，但一次展开全部Gaussian邻域仍会产生
    # 大量indices、距离和128维特征临时张量，因此按chunk_size分批计算。
    # 该Python循环及每个分块内多次CUDA kernel启动也是当前实现耗时较高的原因。
    for chunk_start in range(0, len(centers), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(centers))
        chunk_centers = centers[chunk_start:chunk_end].float()
        chunk_scales = scales[chunk_start:chunk_end].float().clamp_min(1e-4)
        chunk_rotations = rotations_matrix[chunk_start:chunk_end]

        #* ==================== 4. 构造各Gaussian的局部候选体素邻域 ====================
        # scales描述Gaussian局部坐标系三个主轴的标准差。旋转后先计算它在
        # 世界XYZ轴上的标准差，再乘sigma_limit并换算成体素半径，从而得到
        # 包住椭球有效范围的轴对齐包围盒，不必检查整个324000体素网格。
        covariance_diagonal = torch.sum(
            chunk_rotations.square() * chunk_scales[:, None, :].square(), dim=-1
        )
        world_std = covariance_diagonal.clamp_min(1e-8).sqrt()
        radii = torch.ceil(sigma_limit * world_std / grid_step).long()
        radii = torch.minimum(radii, grid_shape_xyz[None] - 1)

        # 同一分块先按最大的XYZ半径构造一份公共offset网格；每个Gaussian
        # 再通过inside_radius删掉超出自身实际半径的offset。这样代码易于并行，
        # 但分块中若存在大尺度Gaussian，其他Gaussian也会检查更多候选位置。
        maximum_radius = radii.max(dim=0).values

        offsets_x = torch.arange(
            -maximum_radius[0], maximum_radius[0] + 1, device=centers.device
        )
        offsets_y = torch.arange(
            -maximum_radius[1], maximum_radius[1] + 1, device=centers.device
        )
        offsets_z = torch.arange(
            -maximum_radius[2], maximum_radius[2] + 1, device=centers.device
        )
        offsets = torch.stack(
            torch.meshgrid(offsets_x, offsets_y, offsets_z, indexing="ij"), dim=-1
        ).reshape(1, -1, 3)

        # 将Gaussian物理中心映射到最近的规则体素索引，再加局部offset得到
        # 候选体素XYZ索引；同时排除超出自身半径和越出全局网格边界的位置。
        base_indices = torch.round(
            (chunk_centers - grid_first_center) / grid_step
        ).long()
        indices_xyz = base_indices[:, None, :] + offsets
        inside_radius = (
            offsets.abs() <= radii[:, None, :]
        ).all(dim=-1)
        inside_grid = (
            (indices_xyz >= 0) & (indices_xyz < grid_shape_xyz[None, None])
        ).all(dim=-1)

        #* ==================== 5. 计算各候选体素的三维Gaussian权重 ====================
        # 先把候选体素索引恢复为物理中心，再计算它相对Gaussian中心的位移。
        # 将世界坐标位移旋转到Gaussian局部坐标系，用各主轴scale归一化后，
        # 得到平方Mahalanobis距离；只保留sigma_limit椭球范围内的位置。
        voxel_centers = (
            grid_first_center[None, None] + indices_xyz.type_as(centers) * grid_step
        )
        delta_world = voxel_centers - chunk_centers[:, None]
        # 对行向量执行delta_local = delta_world @ R，相当于R^T作用于列向量。
        delta_local = torch.einsum("bni,bij->bnj", delta_world, chunk_rotations)
        mahalanobis = torch.sum(
            (delta_local / chunk_scales[:, None]).square(), dim=-1
        )
        valid = inside_radius & inside_grid & (mahalanobis <= sigma_limit ** 2)
        if not valid.any():
            continue

        # 标准Gaussian核权重为exp(-0.5*d²)，再乘不透明度alpha。
        # 当前opacities通常初始化为1，后续正式模型可改为网络预测值。
        weights = (
            torch.exp(-0.5 * mahalanobis)
            * opacities[chunk_start:chunk_end, 0].float()[:, None]
        )
        gaussian_indices, neighbor_indices = torch.nonzero(valid, as_tuple=True)
        chosen_xyz = indices_xyz[gaussian_indices, neighbor_indices]

        #* ==================== 6. 将Gaussian特征累加到公共体素 ====================
        # 按[z,y,x]顺序把三维索引展平。每条有效“Gaussian-体素”关系都会生成
        # 一份“权重×128维特征”，再用index_add_汇聚到目标体素；多个Gaussian
        # 写入同一体素时存在原子累加竞争，这是当前splatting的主要耗时之一。
        flat_indices = (
            chosen_xyz[:, 2] * (y_size * x_size)
            + chosen_xyz[:, 1] * x_size
            + chosen_xyz[:, 0]
        )
        chosen_weights = weights[gaussian_indices, neighbor_indices]
        contributions = (
            chosen_weights[:, None]
            * features[chunk_start:chunk_end][gaussian_indices]
        )
        feature_sum.index_add_(0, flat_indices, contributions)
        weight_sum.index_add_(0, flat_indices, chosen_weights[:, None])

    #* ==================== 7. 得到每个三维体素的最终特征 ====================
    # normalized：用总权重归一化，得到覆盖该体素的Gaussian加权平均特征，
    # 可避免Gaussian数量不同直接改变特征幅值；sum：保留未归一化加权和。
    # 没有Gaussian覆盖的体素feature_sum和weight_sum均为0，归一化后仍为0。
    if accumulation == "normalized":
        voxel_features = feature_sum / weight_sum.clamp_min(1e-6)
    elif accumulation == "sum":
        voxel_features = feature_sum
    else:
        raise ValueError(f"不支持的splat accumulation：{accumulation}")

    #* ==================== 8. 恢复3D体积并转换为DAOcc BEV ====================
    # flat索引按[z,y,x]展开，首先恢复为[B,C,Z,Y,X]稠密体积；随后不对高度
    # Z求和或平均，而是把Z层和通道C拼接，保留不同高度层的信息。
    volume_features = voxel_features.reshape(z_size, y_size, x_size, channels)
    volume_features = volume_features.permute(3, 0, 1, 2).unsqueeze(0).contiguous()

    # Gaussian栅格自然得到[B,C*Z,Y,X]，但DAOcc旧SparseEncoder、
    # BEVTransformV2及ConvFuser沿用[B,C,X,Y]内部排列。融合前必须交换X/Y，
    # 否则数组中同一位置会对应两个模态的不同物理位置。
    bev_features_yx = volume_features.permute(0, 2, 1, 3, 4).reshape(
        1, z_size * channels, y_size, x_size
    ).contiguous()
    bev_features = bev_features_yx.permute(0, 1, 3, 2).contiguous()
    weight_volume = weight_sum.reshape(1, 1, z_size, y_size, x_size)
    return {
        "volume_features": volume_features,
        "bev_features_yx": bev_features_yx,
        "bev_features": bev_features,
        "weight_volume": weight_volume,
        "grid_step_xyz": grid_step,
        "occupied_voxel_count": int((weight_sum[:, 0] > 0).sum()),
        "total_voxel_count": int(total_voxels),
    }


def splat_gaussians_to_height_aware_bev(
    centers, scales, rotations, opacities, features, vtransform,
    sigma_limit=3.0, chunk_size=64, accumulation="normalized",
    implementation="cuda",
):
    """选择CUDA或原PyTorch实现，输出统一的体素及BEV特征字典。

    CUDA实现沿用GaussianFormer localagg的“包围盒分桶、CUB排序、按网格点
    聚合候选Gaussian”框架，但计算语义与本脚本原实现一致。PyTorch实现保留
    为数值对照及无扩展环境下的显式回退；chunk_size仅对PyTorch实现生效。

    张量记号：G为高斯数量，C为特征通道数，X/Y/Z为目标网格各轴格数。
    当前CUDA接口仅支持单样本B=1、C=128；典型配置G=1200、X=Y=180、Z=10。
    输入高斯没有batch维，不能直接把多个样本的高斯拼在一起传入。

    输入：
        centers：[G,3]，增强后LiDAR物理坐标系中的中心[x,y,z]，单位米。
        scales：[G,3]，高斯局部三主轴的标准差，单位米，并非完整直径。
        rotations：[G,4]，按[w,x,y,z]排列的四元数，转换时会归一化。
        opacities：[G,1]，实际alpha权重，并非sigmoid前的logit。
        features：[G,C]，已从图像采样并映射到128通道的高斯特征。
        vtransform：提供pc_range（6个物理边界值）和volume_size（[X,Y,Z]）。
        sigma_limit：标准差截断倍数，只聚合椭球范围内的体素中心。
        accumulation：normalized为除以总权重的加权平均，sum为加权和。
    """
    #* ==================== 1. 选择体素聚合实现 ====================
    # 两条路径均返回相同字段和空间排列；PyTorch路径内部完成四元数转换、
    # 体素聚合和高度折叠。下方继续展开CUDA路径的各阶段。
    if implementation == "pytorch":
        return _splat_gaussians_to_height_aware_bev_pytorch(
            centers, scales, rotations, opacities, features, vtransform,
            sigma_limit, chunk_size, accumulation,
        )
    if implementation != "cuda":
        raise ValueError(f"不支持的splat implementation：{implementation}")

    #* ==================== 2. 把四元数转换为旋转矩阵 ====================
    # R的列向量是高斯局部主轴在LiDAR坐标系中的方向；底层用它与scales
    # 构造协方差Sigma=R diag(scales²) R^T，从而计算三维椭球距离。
    rotation_matrices = quaternion_wxyz_to_rotation_matrix(rotations)  # [G,3,3]。

    #* ==================== 3. CUDA聚合到三维体素网格 ====================
    # float()将连续属性统一为FP32；contiguous()保证底层CUDA所需的连续内存。
    # 二者不会主动切断梯度；是否记录梯度由调用处的no_grad上下文等决定。
    # 每个体素按opacity*exp(-0.5*d²)聚合覆盖它的高斯特征；没有覆盖则为0。
    # 此时保留Z轴，尚未变成二维BEV，也不是沿相机射线的遮挡合成。
    volume_features, weight_volume = gaussian_voxel_splat(
        centers.float().contiguous(),           # [G,3]：物理中心。
        scales.float().contiguous(),            # [G,3]：三主轴标准差。
        rotation_matrices.float().contiguous(), # [G,3,3]：旋转矩阵。
        opacities.float().contiguous(),         # [G,1]：高斯权重系数。
        features.float().contiguous(),          # [G,128]：待聚合图像特征。
        pc_range=vtransform.pc_range,           # [6]：xmin,ymin,zmin,xmax,ymax,zmax。
        grid_size=vtransform.volume_size,       # [3]：[X,Y,Z]网格数量。
        sigma_limit=sigma_limit,
        accumulation=accumulation,
    )
    # volume_features：[1,C,Z,Y,X]，典型[1,128,10,180,180]，可反向求导。
    # weight_volume：[1,1,Z,Y,X]，每个体素的总权重；当前仅作统计，不参与求导。

    #* ==================== 4. 将高度层拼入BEV通道 ====================
    # permute：[1,C,Z,Y,X] -> [1,Z,C,Y,X]，再reshape为[1,Z*C,Y,X]。
    # 高度没有求和或取平均；通道顺序为z0的全部C通道、z1的全部C通道……
    # 最终通道索引为z*C+c，不能直接在原[C,Z]排列上reshape，否则顺序不同。
    _, channels, z_size, y_size, x_size = volume_features.shape
    bev_features_yx = volume_features.permute(0, 2, 1, 3, 4).reshape(
        1, z_size * channels, y_size, x_size
    ).contiguous()  # [1,Z*C,Y,X]，典型[1,1280,180,180]。
    # DAOcc融合模块沿用[X,Y]空间排列，交换最后两轴以对齐LiDAR BEV。
    # X=Y=180时shape看似不变，但每个数组位置对应的物理位置已改变。
    bev_features = bev_features_yx.permute(0, 1, 3, 2).contiguous()  # [1,Z*C,X,Y]。

    #* ==================== 5. 计算体素物理尺寸并整理输出 ====================
    # 各轴物理跨度除以格数，得到[dx,dy,dz]；当前为[0.6,0.6,0.8]米。
    grid_step = (
        centers.new_tensor(vtransform.pc_range[3:])
        - centers.new_tensor(vtransform.pc_range[:3])
    ) / centers.new_tensor(vtransform.volume_size)  # [3]，XYZ顺序。
    return {
        "volume_features": volume_features,  # [1,C,Z,Y,X]：完整三维特征。
        "bev_features_yx": bev_features_yx,  # [1,Z*C,Y,X]：物理平面排列。
        "bev_features": bev_features,       # [1,Z*C,X,Y]：送入DAOcc融合模块。
        "weight_volume": weight_volume,     # [1,1,Z,Y,X]：未归一化总权重。
        "grid_step_xyz": grid_step,         # [3]：体素物理尺寸，单位米。
        # 以下为Python int标量；occupied按正权重计数，不是非零特征元素数。
        "occupied_voxel_count": int((weight_volume > 0).sum()),
        "total_voxel_count": int(x_size * y_size * z_size),
    }


def tensor_stats(tensor):
    value = tensor.detach().float()
    return {
        "shape": list(value.shape),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "abs_max": float(value.abs().max()),
        "finite": bool(torch.isfinite(value).all()),
    }


def daocc_bev_to_physical_heatmap(bev_features):
    """把DAOcc内部[B,C,X,Y]特征转换为供imshow显示的[Y,X]热力图。

    DAOcc为兼容旧SparseEncoder，在融合、decoder和head之间沿用X在前、Y在后
    的空间排列；Matplotlib imshow则把二维数组解释为[纵轴Y,横轴X]。因此先在
    通道维取绝对值均值，再交换最后两维。该转换只用于绘图，不参与模型计算。
    """
    heatmap_xy = bev_features.detach().float().abs().mean(dim=1)[0]
    return heatmap_xy.transpose(0, 1).contiguous().cpu()


def timed_cuda_call(function, *args, **kwargs):
    """同步CUDA后测量一次阶段的实际墙钟时间，并返回函数结果和毫秒数。

    CUDA算子默认异步执行；若不在计时边界同步，后续阶段会替前一阶段承担时间，
    得到的分项结果没有意义。CUDA FPS同样在边界同步后再记录耗时。
    """
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = function(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return output, elapsed_ms


def save_visualization(
    images, points, metas, source, centers_all, anchors, path, point_range
):
    """用单行紧凑布局展示相机投影、输入点云及Gaussian的BEV分布。"""
    restored = camera_pipeline.denormalize_images(images, metas)
    points = points.detach().float().cpu().numpy()
    centers_all = centers_all.detach().cpu().numpy()
    centers = anchors["centers_xyz"].numpy()
    visibility = anchors["camera_visibility"].numpy()
    uv = anchors["image_uv"].numpy()
    camera_depths = anchors["camera_depths"].numpy()
    feature_norm = np.linalg.norm(anchors["features"].numpy(), axis=1)
    gaussian_bev_heatmap = anchors["gaussian_bev_heatmap"].numpy()

    #* 六路相机仍采用2x3排列，但把它作为整张图最左侧的一个紧凑区域；
    # LiDAR输入、Gaussian选择结果和图像特征强度依次放在右侧，整图只有一行。
    fig = plt.figure(figsize=(44, 4.15), facecolor="white")
    outer = fig.add_gridspec(
        1, 9, width_ratios=[3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        wspace=0.045,
    )
    camera_grid = outer[0].subgridspec(2, 3, wspace=0.008, hspace=0.025)
    camera_names = source["camera_names"]
    valid_depths = camera_depths[visibility]
    if len(valid_depths):
        depth_min, depth_max = np.percentile(valid_depths, [2.0, 98.0])
    else:
        depth_min, depth_max = 0.0, 1.0
    if depth_max <= depth_min:
        depth_max = depth_min + 1.0
    depth_artist = None
    for camera_index in range(min(6, len(restored))):
        axis = fig.add_subplot(
            camera_grid[camera_index // 3, camera_index % 3]
        )
        axis.imshow(restored[camera_index], aspect="auto")
        visible = visibility[:, camera_index]
        depth_artist = axis.scatter(
            uv[visible, camera_index, 0], uv[visible, camera_index, 1],
            s=2.0, c=camera_depths[visible, camera_index], cmap="turbo",
            vmin=depth_min, vmax=depth_max, alpha=0.72,
        )
        axis.text(
            0.5, 0.995, camera_names[camera_index], transform=axis.transAxes,
            ha="center", va="top", fontsize=7, color="white",
            bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=0.8),
        )
        axis.set_xlim(0, restored.shape[2])
        axis.set_ylim(restored.shape[1], 0)
        axis.axis("off")

    # 深度色条嵌在相机区域内部，避免额外留出大片白边。
    if depth_artist is not None:
        colorbar = fig.colorbar(
            depth_artist, ax=fig.axes[:min(6, len(restored))],
            fraction=0.008, pad=0.004, aspect=30,
        )
        colorbar.set_label("camera depth (m)", fontsize=7)
        colorbar.ax.tick_params(labelsize=6, length=2)

    #* LiDAR输入点云俯视图。第5维通常是历史帧时间差：0为当前帧，非0为历史帧；
    # 若数据只有XYZ/强度，则统一画为输入点云，不虚构帧来源。
    lidar_axis = fig.add_subplot(outer[1])
    if points.shape[1] >= 5:
        current = np.isclose(points[:, 4], 0.0, atol=1e-6)
        history = ~current
        lidar_axis.scatter(
            points[history, 0], points[history, 1], s=0.18, c="#e58b2a",
            alpha=0.30, rasterized=True, label=f"history ({history.sum():,})",
        )
        lidar_axis.scatter(
            points[current, 0], points[current, 1], s=0.25, c="#2468b4",
            alpha=0.42, rasterized=True, label=f"current ({current.sum():,})",
        )
        lidar_axis.legend(loc="upper right", fontsize=6, markerscale=5,
                          framealpha=0.72, borderpad=0.25)
    else:
        lidar_axis.scatter(
            points[:, 0], points[:, 1], s=0.2, c="#2468b4",
            alpha=0.35, rasterized=True,
        )
    lidar_axis.set_title(f"LiDAR input ({len(points):,} points)", fontsize=9, pad=2)
    lidar_axis.set_xlim(point_range[0], point_range[3])
    lidar_axis.set_ylim(point_range[1], point_range[4])
    lidar_axis.set_aspect("equal")
    lidar_axis.set_xticks([])
    lidar_axis.set_yticks([])

    bev_axis = fig.add_subplot(outer[2])
    bev_axis.scatter(
        centers_all[:, 0], centers_all[:, 1], s=0.25, c="0.78",
        label=f"conv_out candidates ({len(centers_all):,})",
    )
    shown = bev_axis.scatter(
        centers[:, 0], centers[:, 1], s=3.0,
        c=anchors["visible_camera_count"].numpy(), cmap="viridis",
        label=f"selected Gaussians ({len(centers):,})",
    )
    visible_bar = fig.colorbar(shown, ax=bev_axis, fraction=0.04, pad=0.012)
    visible_bar.set_label("visible cameras", fontsize=7)
    visible_bar.ax.tick_params(labelsize=6, length=2)
    bev_axis.set_xlim(point_range[0], point_range[3])
    bev_axis.set_ylim(point_range[1], point_range[4])
    bev_axis.set_aspect("equal")
    bev_axis.set_title("LiDAR candidates / Gaussians", fontsize=9, pad=2)
    bev_axis.set_xticks([])
    bev_axis.set_yticks([])
    bev_axis.legend(loc="upper right", fontsize=6, markerscale=3,
                    framealpha=0.72, borderpad=0.25)

    feature_axis = fig.add_subplot(outer[3])
    colored = feature_axis.scatter(
        centers[:, 0], centers[:, 1], s=3.0, c=feature_norm,
        cmap="magma",
    )
    feature_bar = fig.colorbar(colored, ax=feature_axis, fraction=0.04, pad=0.012)
    feature_bar.set_label("image feature L2", fontsize=7)
    feature_bar.ax.tick_params(labelsize=6, length=2)
    feature_axis.set_xlim(point_range[0], point_range[3])
    feature_axis.set_ylim(point_range[1], point_range[4])
    feature_axis.set_aspect("equal")
    feature_axis.set_title("Gaussian image features", fontsize=9, pad=2)
    feature_axis.set_xticks([])
    feature_axis.set_yticks([])

    #* 高度感知Gaussian-to-BEV结果：模型内部[X,Y]热力图已在整理锚点时
    # 转为imshow所需的物理[Y,X]排列。这里只观察覆盖和响应，不代表语义概率。
    gaussian_bev_axis = fig.add_subplot(outer[4])
    bev_image = gaussian_bev_axis.imshow(
        gaussian_bev_heatmap,
        origin="lower",
        extent=[point_range[0], point_range[3], point_range[1], point_range[4]],
        cmap="magma",
        aspect="equal",
    )
    bev_bar = fig.colorbar(bev_image, ax=gaussian_bev_axis, fraction=0.04, pad=0.012)
    bev_bar.set_label("mean |Gaussian BEV feature|", fontsize=7)
    bev_bar.ax.tick_params(labelsize=6, length=2)
    gaussian_bev_axis.set_title("Height-aware Gaussian BEV", fontsize=9, pad=2)
    gaussian_bev_axis.set_xticks([])
    gaussian_bev_axis.set_yticks([])

    #* 使用同一物理[Y,X]排列比较LiDAR、Gaussian融合、原视角变换和原融合BEV。
    # 各图独立色标用于观察自身空间结构；颜色值不宜直接跨图比较绝对大小，
    # 绝对数值差异应以每帧JSON中的mean/std/abs_max为准。
    comparison_maps = (
        ("lidar_bev_heatmap", "LiDAR BEV", 5),
        ("gaussian_fused_bev_heatmap", "Gaussian BEV + LiDAR", 6),
        ("original_camera_bev_heatmap", "BEVTransformV2 camera BEV", 7),
        ("original_fused_bev_heatmap", "Original camera BEV + LiDAR", 8),
    )
    for key, title, column in comparison_maps:
        axis = fig.add_subplot(outer[column])
        image = axis.imshow(
            anchors[key].numpy(), origin="lower",
            extent=[point_range[0], point_range[3], point_range[1], point_range[4]],
            cmap="magma", aspect="equal",
        )
        colorbar = fig.colorbar(image, ax=axis, fraction=0.04, pad=0.012)
        colorbar.ax.tick_params(labelsize=6, length=2)
        axis.set_title(title, fontsize=9, pad=2)
        axis.set_xticks([])
        axis.set_yticks([])

    fig.subplots_adjust(left=0.001, right=0.998, bottom=0.004, top=0.988)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.005)
    plt.close(fig)


def summarize_values(values):
    """汇总多样本标量，便于判断典型值和慢样本尾部。"""
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def warmup_pipeline(
    sample, args, modules, voxelizer, lidar_backbone, fuser, lattice, lidar_cfg
):
    """仅用第一个均匀样本预热完整初始化链路，不保存预热结果。"""
    images, points, matrices, _, _ = sample
    with torch.no_grad():
        for _ in range(args.warmup_iters):
            camera_features = run_camera_fpn(images, modules)
            lidar_output = run_lidar_conv_out(points, voxelizer, lidar_backbone)
            candidates = sparse_indices_to_centers(lidar_output["coordinates_xyz"], lattice)
            projected = project_and_sample_image_features(
                candidates, camera_features, matrices,
                input_height=images.shape[-2], input_width=images.shape[-1],
            )
            visible = torch.nonzero(
                projected["visibility_count"] > 0, as_tuple=False
            ).squeeze(1)
            selected_visible = farthest_point_sample_xyz(
                candidates[visible], args.num_gaussians,
                lidar_cfg.voxelize.point_cloud_range,
            )
            selected = visible[selected_visible]
            centers = candidates[selected]
            scales = centers.new_tensor(lattice["step_xyz"])[None].expand(
                args.num_gaussians, -1
            ) * args.scale_factor
            rotations = centers.new_zeros((args.num_gaussians, 4))
            rotations[:, 0] = 1.0
            opacities = centers.new_full((args.num_gaussians, 1), args.initial_opacity)
            bev_features = map_gaussian_features_to_bev_channels(
                projected["features"][selected], modules["vtransform"]
            )
            gaussian_bev = splat_gaussians_to_height_aware_bev(
                centers, scales, rotations, opacities, bev_features,
                modules["vtransform"], args.splat_sigma, args.splat_chunk_size,
                args.splat_accumulation, args.splat_implementation,
            )
            original_camera_bev = run_original_camera_bev(
                camera_features, points, matrices, sample[3], modules
            )
            _ = fuser([gaussian_bev["bev_features"], lidar_output["bev_features"]])
            _ = fuser([original_camera_bev, lidar_output["bev_features"]])
        torch.cuda.synchronize()


def process_sample(
    sample, sample_order, dataset_index, args, device, camera_modules,
    voxelizer, lidar_backbone, fuser, lidar_cfg, lattice, root,
):
    """处理一个均匀抽取的样本，保存该帧JSON及可选可视化。"""

    #* ==================== 1. 解包当前样本 ==============================
    # images：训练pipeline处理后的六路图像，形状为[1,6,3,H,W]；
    # points：当前帧与历史帧聚合后的LiDAR点；
    # matrices：相机内外参及图像、LiDAR数据增强矩阵；
    # metas/source：图像归一化信息、场景token和帧token等记录信息。
    images, points, matrices, metas, source = sample

    #* ==================== 2. 提取相机和LiDAR分支特征 ====================
    # 计时从模型输入开始。timed_cuda_call会在阶段前后同步CUDA，因此各阶段
    # 耗时不会因CUDA异步执行而错误地计入后续阶段。
    generation_started = time.perf_counter()
    with torch.no_grad():
        #* 相机分支
        # 六路图像经过camera backbone和neck，得到逐相机FPN特征；
        # 此处不执行原DAOcc的图像到BEV视角变换。
        camera_features, camera_fpn_ms = timed_cuda_call(
            run_camera_fpn, images, camera_modules
        ) # (1 6 256 32 88) 1/8下采样特征图
        #* lidar分支
        # 聚合点云依次经过spconv2体素化和稀疏backbone，保留conv_out的
        # 非空稀疏坐标与对应LiDAR特征，不执行dense()和高度维折叠。
        lidar_output, lidar_conv_out_ms = timed_cuda_call(
            run_lidar_conv_out, points, voxelizer, lidar_backbone
        )

    #* ==================== 3. 构建全部Gaussian候选中心 ====================
    # conv_out只提供离散[x,y,z]稀疏索引。根据已计算的格点原点和步长，
    # 将每个非空位置转换为增强后LiDAR坐标系中的真实物理中心，单位为米。
    candidate_centers, coordinate_conversion_ms = timed_cuda_call(sparse_indices_to_centers, lidar_output["coordinates_xyz"], lattice)

    #* ==================== 4. 投影候选并剔除相机不可见位置 ====================
    # 将全部候选中心投影到六路相机，同时从各相机FPN特征图双线性采样特征。
    # 多路相机都可见时对有效特征取均值；visibility_count记录可见相机数。
    projected_candidates, visibility_filter_ms = timed_cuda_call(
        project_and_sample_image_features, candidate_centers, camera_features,
        matrices, input_height=images.shape[-2], input_width=images.shape[-1],
    )
    # 仅保留位于至少一路相机前方、且投影点落在增强后图像范围内的候选，
    # 从源头保证后续选出的Gaussian具有有效图像特征。
    visible_candidate_indices = torch.nonzero(projected_candidates["visibility_count"] > 0, as_tuple=False).squeeze(1)
    # 若当前帧可见候选不足，无法无重复地构建指定数量的Gaussian，直接报错并
    # 给出数据集索引，避免静默重复候选或产生缺少图像特征的锚点。
    if len(visible_candidate_indices) < args.num_gaussians:
        raise RuntimeError(
            f"dataset index={dataset_index}只有{len(visible_candidate_indices)}个"
            f"相机可见候选，无法采样{args.num_gaussians}个Gaussian"
        )

    #* ==================== 5. CUDA FPS选择代表性Gaussian ==========================
    # 先把XYZ按完整感知范围归一化，再使用mmdet3d CUDA FPS，从相机可见
    # 候选中选择空间分布较均匀的固定数量锚点。selected_visible是可见候选
    # 子集内的局部索引，selected再映射回全部conv_out候选的全局索引。
    selected_visible, fps_sampling_ms = timed_cuda_call(
        farthest_point_sample_xyz, candidate_centers[visible_candidate_indices],
        args.num_gaussians, lidar_cfg.voxelize.point_cloud_range,
    )
    selected = visible_candidate_indices[selected_visible]
    centers = candidate_centers[selected]
    # 投影和图像特征采样已经对全部候选执行过，此处直接索引选中结果，
    # 不再对1200个锚点重复执行投影与grid_sample。
    sampled = {key: value[selected] for key, value in projected_candidates.items()} 

    #* ==================== 6. 初始化Gaussian尺度、旋转和不透明度 ====================
    # 当前版本使用conv_out相邻格点的物理间距乘scale_factor作为固定初始尺度，
    # 尚未通过网络预测每个Gaussian各自的尺度修正量。
    scale = torch.as_tensor(
        lattice["step_xyz"], dtype=torch.float32, device=device
    ) * args.scale_factor
    scales = scale[None].expand(args.num_gaussians, -1).contiguous()

    # 旋转使用[w,x,y,z]顺序的单位四元数[1,0,0,0]，表示初始椭球主轴与
    # LiDAR坐标系XYZ轴对齐。后续可由更新网络预测四元数并重新归一化。
    rotations = torch.zeros(args.num_gaussians, 4, device=device)
    rotations[:, 0] = 1.0

    # opacities保存实际alpha值而非sigmoid之前的logit。默认1表示完全不透明；
    # 可以通过--initial-opacity设置其他统一初值，后续再由网络预测或更新。
    opacities = torch.full(
        (args.num_gaussians, 1), args.initial_opacity,
        dtype=torch.float32, device=device,
    )

    #* ==================== 7. 映射特征并生成高度感知Gaussian BEV ====================
    # 复用原BEVTransformV2的transfer_conv权重，将Camera FPN采样得到的256维
    # Gaussian特征映射为128维，保证后续高度折叠后仍为原DAOcc需要的1280通道。
    gaussian_bev_features, feature_projection_ms = timed_cuda_call(
        map_gaussian_features_to_bev_channels,
        sampled["features"], camera_modules["vtransform"],
    )
    # 生成高斯BEV
    # 在10×180×180网格中按3D位置、尺度、旋转和opacity计算局部高斯权重；
    # 输出volume为[1,128,10,180,180]；先按物理网格得到[Y,X]，随后转换为
    # DAOcc内部统一使用的[X,Y]排列，最终形状仍为[1,1280,180,180]。
    gaussian_bev, gaussian_splat_ms = timed_cuda_call(
        splat_gaussians_to_height_aware_bev,
        centers, scales, rotations, opacities, gaussian_bev_features,
        camera_modules["vtransform"], args.splat_sigma, args.splat_chunk_size,
        args.splat_accumulation, args.splat_implementation,
    )

    #* ==================== 8. 生成原相机BEV并分别执行多模态融合 ====================
    # 原分支直接复用同一camera FPN特征执行BEVTransformV2，其内部空间排列为
    # [X,Y]；Gaussian BEV已在splat函数末尾转成相同排列，因此两条路径可以
    # 使用完全相同的LiDAR BEV和ConvFuser。
    original_camera_bev, original_vtransform_ms = timed_cuda_call(
        run_original_camera_bev,
        camera_features, points, matrices, metas, camera_modules,
    )
    gaussian_fused_bev, gaussian_fusion_ms = timed_cuda_call(
        fuser, [gaussian_bev["bev_features"], lidar_output["bev_features"]]
    )
    original_fused_bev, original_fusion_ms = timed_cuda_call(
        fuser, [original_camera_bev, lidar_output["bev_features"]]
    )
    torch.cuda.synchronize()

    #* ==================== 9. 整理分阶段耗时 ====================
    # 总耗时还包括索引映射、属性初始化和少量Python调度；不包含Dataset读取、
    # 模型/权重加载、JSON写盘和Matplotlib可视化。
    timings_ms = {
        "camera_fpn": camera_fpn_ms,
        "lidar_voxelization_and_conv_out": lidar_conv_out_ms,
        "sparse_coordinate_to_physical_center": coordinate_conversion_ms,
        "visibility_filter_and_sample_all_candidate_image_features": visibility_filter_ms,
        "cuda_farthest_point_sampling": fps_sampling_ms,
        "gaussian_feature_projection_256_to_128": feature_projection_ms,
        "height_aware_gaussian_splatting": gaussian_splat_ms,
        "original_bevtransform_v2": original_vtransform_ms,
        "gaussian_bev_lidar_fusion": gaussian_fusion_ms,
        "original_camera_bev_lidar_fusion": original_fusion_ms,
        # Gaussian路径包含双分支特征、锚点构建、splatting和一次融合。
        "gaussian_path_total": (
            camera_fpn_ms + lidar_conv_out_ms + coordinate_conversion_ms
            + visibility_filter_ms + fps_sampling_ms + feature_projection_ms
            + gaussian_splat_ms + gaussian_fusion_ms
        ),
        # 原路径不需要Gaussian候选、投影采样和FPS，只使用原视角变换及融合。
        "original_bevtransform_path_total": (
            camera_fpn_ms + lidar_conv_out_ms
            + original_vtransform_ms + original_fusion_ms
        ),
        # 本独立脚本同时计算两条路径用于对照，墙钟总时间不能代表单一模型。
        "all_comparison_paths_wall_total": (
            time.perf_counter() - generation_started
        ) * 1000.0,
    }

    #* ==================== 10. 整理锚点属性用于统计和可视化 ====================
    # 当前脚本不保存PTH，转到CPU只为后续计算统计值和绘图：
    # - centers/scales/rotations/opacities：Gaussian基础属性；
    # - features：由有效多视角相机FPN特征取均值得到；
    # - lidar_seed_features：同一稀疏位置的LiDAR特征，供后续属性更新设计参考；
    # - visibility/depth/uv：用于验证投影关系及绘制六路相机散点。
    anchors = {
        "centers_xyz": centers.detach().cpu(),
        "scales_xyz": scales.detach().cpu(),
        "rotations_wxyz": rotations.detach().cpu(),
        "opacities": opacities.detach().cpu(),
        "features": sampled["features"].detach().cpu(),
        "lidar_seed_features": lidar_output["features"][selected].detach().cpu(),
        "source_sparse_indices_xyz": lidar_output["coordinates_xyz"][selected].detach().cpu(),
        "camera_visibility": sampled["visibility_mask"].detach().cpu(),
        "visible_camera_count": sampled["visibility_count"].detach().cpu(),
        "camera_depths": sampled["camera_depths"].detach().cpu(),
        "image_uv": sampled["image_uv"].detach().cpu(),
        # 高度和通道取绝对值均值，只保存轻量二维热力图用于可视化；
        # 完整1280通道BEV不写盘，避免每个样本产生大型特征文件。
        #* 以下模型特征均为DAOcc内部[X,Y]排列。绘图前统一转成Matplotlib
        # 使用的[Y,X]，确保热力图朝向能与物理点云及Gaussian中心直接比较。
        "gaussian_bev_heatmap": daocc_bev_to_physical_heatmap(
            gaussian_bev["bev_features"]
        ),
        "lidar_bev_heatmap": daocc_bev_to_physical_heatmap(
            lidar_output["bev_features"]
        ),
        "gaussian_fused_bev_heatmap": daocc_bev_to_physical_heatmap(
            gaussian_fused_bev
        ),
        "original_camera_bev_heatmap": daocc_bev_to_physical_heatmap(
            original_camera_bev
        ),
        "original_fused_bev_heatmap": daocc_bev_to_physical_heatmap(
            original_fused_bev
        ),
    }

    #* ==================== 11. 创建当前帧输出目录 ====================
    # 每帧采用“场景token_帧token”作为目录名；token缺失时回退到数据集索引，
    # 防止多个均匀样本的JSON和可视化文件相互覆盖。
    scene = source["scene_token"] or "scene_unknown"
    frame = source["frame_token"] or f"frame_index_{dataset_index}"
    sample_dir = root / f"{scene}_{frame}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    json_path = sample_dir / "initial_gaussian_anchors.json"
    image_path = sample_dir / "initial_gaussian_anchors.png"
    visible_counts = anchors["visible_camera_count"].numpy()

    #* ==================== 12. 生成并保存当前帧JSON摘要 ====================
    # JSON只保存方法、数量、特征统计、可见性和耗时等轻量结果，不保存完整
    # Gaussian张量。每帧结果稍后还会被main()收集到根目录汇总JSON中。
    result = {
        "sample_order": sample_order,
        "source": {**source, "output_directory": str(sample_dir)},
        "checkpoint": str(args.checkpoint.expanduser().resolve()) if args.checkpoint else None,
        "initialization": {
            "candidate_source": "SparseEncoderSpconv2.conv_out non-empty outputs",
            "candidate_filter": "positive camera depth and inside at least one augmented image",
            "selection": "mmdet3d CUDA furthest_point_sample in normalized XYZ",
            "gaussian_feature_source": "mean of valid multi-view Camera FPN samples",
            "gaussian_scale": "fixed conv_out lattice spacing * scale_factor",
            "gaussian_rotation": "identity quaternion [w,x,y,z] = [1,0,0,0]",
            "gaussian_opacity": f"constant alpha = {args.initial_opacity}",
            "gaussian_to_bev": (
                "local anisotropic 3D splatting, then concatenate height into channels"
            ),
            "learnable_update": False,
            "splatting": True,
        },
        "num_input_points": int(len(points)),
        "num_input_voxels": lidar_output["input_voxel_count"],
        "num_conv_out_candidates": int(len(candidate_centers)),
        "num_visible_conv_out_candidates": int(len(visible_candidate_indices)),
        "num_discarded_invisible_candidates": int(
            len(candidate_centers) - len(visible_candidate_indices)
        ),
        "num_selected_gaussians": int(args.num_gaussians),
        "conv_out_spatial_shape_zyx": lidar_output["spatial_shape_zyx"],
        "conv_out_lattice": {key: np.asarray(value).tolist() for key, value in lattice.items()},
        "scale_factor": args.scale_factor,
        "initial_opacity": args.initial_opacity,
        "splatting": {
            "implementation": args.splat_implementation,
            "sigma_limit": args.splat_sigma,
            "chunk_size": args.splat_chunk_size,
            "accumulation": args.splat_accumulation,
            "grid_step_xyz": gaussian_bev["grid_step_xyz"].detach().cpu().tolist(),
            "volume_feature_shape": list(gaussian_bev["volume_features"].shape),
            "bev_feature_shape": list(gaussian_bev["bev_features"].shape),
            "volume_layout": "[B,C,Z,Y,X]",
            "bev_layout_before_daocc_conversion": "[B,C*Z,Y,X]",
            "bev_layout_for_fusion": "[B,C*Z,X,Y]",
            "occupied_voxel_count": gaussian_bev["occupied_voxel_count"],
            "total_voxel_count": gaussian_bev["total_voxel_count"],
            "occupied_voxel_ratio": (
                gaussian_bev["occupied_voxel_count"]
                / gaussian_bev["total_voxel_count"]
            ),
            "bev_feature_stats": tensor_stats(gaussian_bev["bev_features"]),
        },
        "bev_comparison": {
            "model_internal_spatial_layout": "[X,Y]",
            "visualization_spatial_layout": "[Y,X]",
            "lidar_bev_shape": list(lidar_output["bev_features"].shape),
            "lidar_bev_stats": tensor_stats(lidar_output["bev_features"]),
            "gaussian_camera_bev_shape": list(gaussian_bev["bev_features"].shape),
            "gaussian_camera_bev_stats": tensor_stats(gaussian_bev["bev_features"]),
            "gaussian_fused_bev_shape": list(gaussian_fused_bev.shape),
            "gaussian_fused_bev_stats": tensor_stats(gaussian_fused_bev),
            "original_camera_bev_shape": list(original_camera_bev.shape),
            "original_camera_bev_stats": tensor_stats(original_camera_bev),
            "original_fused_bev_shape": list(original_fused_bev.shape),
            "original_fused_bev_stats": tensor_stats(original_fused_bev),
            "fuser": "shared checkpoint ConvFuser([camera_bev, lidar_bev])",
        },
        "camera_fpn_feature_shape": list(camera_features.shape),
        "gaussian_image_feature_stats": tensor_stats(anchors["features"]),
        "lidar_seed_feature_stats": tensor_stats(anchors["lidar_seed_features"]),
        "visibility": {
            "fully_invisible_count": int((visible_counts == 0).sum()),
            "visible_at_least_one_count": int((visible_counts > 0).sum()),
            "mean_visible_cameras": float(visible_counts.mean()),
            "histogram": {
                str(value): int((visible_counts == value).sum())
                for value in range(len(source["camera_names"]) + 1)
            },
        },
        "timings_ms": timings_ms,
        "timing_warmup_iters": args.warmup_iters,
        "timing_scope": (
            "single sample; CUDA synchronized; excludes dataset loading, module/checkpoint "
            "loading, serialization and visualization"
        ),
        "visualization_file": str(image_path) if args.visualize else None,
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    #* ==================== 13. 可选组合可视化 ====================
    # --visualize开启时，绘制六路相机投影、LiDAR输入、全部conv_out候选、
    # FPS选中Gaussian及其图像特征强度；可视化耗时不计入生成耗时。
    if args.visualize:
        save_visualization(
            images, points, metas, source, candidate_centers, anchors, image_path,
            list(lidar_cfg.voxelize.point_cloud_range),
        )

    #* ==================== 14. 输出当前样本进度并返回摘要 ====================
    print(
        f"[{sample_order + 1}/{args.num_samples}] index={dataset_index} "
        f"points={len(points):,} candidates={len(candidate_centers):,} "
        f"visible={len(visible_candidate_indices):,} "
        f"gaussian_path={timings_ms['gaussian_path_total']:.3f} ms "
        f"original_path={timings_ms['original_bevtransform_path_total']:.3f} ms"
    )
    return result


def main():
    args, overrides = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("相机特征采样和spconv2需要CUDA GPU")
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = camera_pipeline.load_config(args.config, overrides)
    if cfg.model.encoders.get("camera") is None or cfg.model.encoders.get("lidar") is None:
        raise ValueError("该脚本要求配置同时包含camera和lidar encoder")
    wrapped_dataset, dataset, history_sweeps, selected_indices = (
        build_dataset_for_uniform_sampling(cfg, args)
    )
    camera_modules = camera_pipeline.build_camera_modules(cfg, device, args.checkpoint)
    voxelizer, lidar_backbone = lidar_pipeline.build_spconv2_modules(
        cfg, device, args.checkpoint
    )
    fuser = build_fuser_module(cfg, device, args.checkpoint)
    lidar_cfg = cfg.model.encoders.lidar
    lattice = conv_out_lattice_geometry(lidar_cfg.backbone, lidar_cfg.voxelize)
    root = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    # 模型只构建一次；首个均匀样本既用于预热，也作为正式处理的第一个样本。
    first_sample = camera_pipeline.load_training_sample(
        dataset, selected_indices[0], history_sweeps, args, device
    )
    warmup_pipeline(
        first_sample, args, camera_modules, voxelizer, lidar_backbone, fuser,
        lattice, lidar_cfg
    )

    sample_results = []
    for order, dataset_index in enumerate(selected_indices):
        sample = first_sample if order == 0 else camera_pipeline.load_training_sample(
            dataset, dataset_index, history_sweeps, args, device
        )
        result = process_sample(
            sample, order, dataset_index, args, device, camera_modules, voxelizer,
            lidar_backbone, fuser, lidar_cfg, lattice, root,
        )
        sample_results.append(result)
        del sample

    timing_keys = sample_results[0]["timings_ms"].keys()
    timing_summary = {
        key: summarize_values([item["timings_ms"][key] for item in sample_results])
        for key in timing_keys
    }
    count_summary = {
        key: summarize_values([item[key] for item in sample_results])
        for key in (
            "num_input_points", "num_input_voxels", "num_conv_out_candidates",
            "num_visible_conv_out_candidates", "num_discarded_invisible_candidates",
        )
    }
    summary = {
        "config": str(Path(args.config).expanduser().resolve()),
        "checkpoint": str(args.checkpoint.expanduser().resolve()) if args.checkpoint else None,
        "wrapped_training_dataset_length": len(wrapped_dataset),
        "dataset_length": len(dataset),
        "num_samples": len(selected_indices),
        "uniform_dataset_indices": selected_indices,
        "num_gaussians_per_sample": args.num_gaussians,
        "configured_history_sweeps": history_sweeps,
        "warmup_iters": args.warmup_iters,
        "visualization_enabled": args.visualize,
        "timing_summary_ms": timing_summary,
        "count_summary": count_summary,
        "samples": sample_results,
    }
    summary_path = root / "initial_gaussian_anchors_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\nGaussian锚点生成耗时汇总（ms）：")
    for name, stats in timing_summary.items():
        print(
            f"  {name}: mean={stats['mean']:.3f}, p50={stats['p50']:.3f}, "
            f"p90={stats['p90']:.3f}"
        )
    print(f"汇总JSON：{summary_path}")


if __name__ == "__main__":
    main()
