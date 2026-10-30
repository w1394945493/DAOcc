from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F
import einops

from mmdet3d.models.builder import (
    build_backbone,
    build_neck,
    build_vtransform,
    build_fuser,
    build_custom,
    build_head,
)
from mmdet3d.ops import DynamicScatter, Spconv2Voxelization, Voxelization
from mmdet3d.models import FUSIONMODELS


from .base import Base3DFusionModel

__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
            if encoders["camera"].get("bev_backbone") is not None:
                self.encoders["camera"]["bev_backbone"] = build_backbone(encoders["camera"]["bev_backbone"])
            if encoders["camera"].get("bev_neck") is not None:
                self.encoders["camera"]["bev_neck"] = build_neck(encoders["camera"]["bev_neck"])

        if encoders.get("lidar") is not None:
            # 未配置 type 时保持原行为：硬体素仍使用仓库原 Voxelization，
            # 因而原配置、原 SparseEncoder 及已有 checkpoint 均不受影响。
            voxelize_cfg = dict(encoders["lidar"]["voxelize"])
            voxelize_type = voxelize_cfg.pop("type", "Voxelization")
            if voxelize_type == "Spconv2Voxelization":
                voxelize_module = Spconv2Voxelization(**voxelize_cfg)
            elif voxelize_type == "Voxelization" and voxelize_cfg.get(
                "max_num_points", -1
            ) > 0:
                voxelize_module = Voxelization(**voxelize_cfg)
            elif voxelize_type in ("Voxelization", "DynamicScatter"):
                voxelize_module = DynamicScatter(**voxelize_cfg)
            else:
                raise ValueError(f"不支持的 LiDAR 体素化类型: {voxelize_type}")
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)
            if encoders["lidar"].get("bev_backbone") is not None:
                self.encoders["lidar"]["bev_backbone"] = build_backbone(encoders["lidar"]["bev_backbone"])
            if encoders["lidar"].get("bev_neck") is not None:
                self.encoders["lidar"]["bev_neck"] = build_neck(encoders["lidar"]["bev_neck"])

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        if decoder is not None:
            self.decoder = nn.ModuleDict(
                {
                    "backbone": build_backbone(decoder["backbone"]),
                    "neck": build_neck(decoder["neck"]),
                }
            )
        else:
            self.decoder = None

        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        # model.loss_scale 是 object/occ 等任务的损失权重，会改变各任务对总 loss 的贡献。
        # 它不是顶层 fp16.loss_scale（梯度缩放），不能用来切换 FP16/FP32。
        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        **kwargs,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
            **kwargs,
        )

        if "bev_backbone" in self.encoders["camera"]:
            x = self.encoders["camera"]["bev_backbone"](x)
        if "bev_neck" in self.encoders["camera"]:
            x = self.encoders["camera"]["bev_neck"](x)

        return x

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)

        if "bev_backbone" in self.encoders["lidar"]:
            x = self.encoders["lidar"]["bev_backbone"](x)
        if "bev_neck" in self.encoders["lidar"]:
            x = self.encoders["lidar"]["bev_neck"](x)

        return x

    # 混合精度下将传入的半精度浮点张量转回 FP32，体素化保留单精度计算。
    # no_grad 只控制梯度记录；force_fp32 才是这里的精度控制。
    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    # 仅在 fp16_enabled 开启时，将 img/points 中的 FP32 张量自动转成 FP16。
    # 几何变换等敏感子模块可通过 force_fp32 转回 FP32，因此这是混合精度。
    # 这里逻辑可近似理解为:
    # def forward(self, img, points, metas):
    #   if not self.fp16_enabled:
    #     # FP32训练：直接执行原方法
    #     return original_forward(img, points, metas)

    #   # 混合精度训练：
    #   img = convert_fp32_to_fp16(img)
    #   points = convert_fp32_to_fp16(points)

    #   with torch.cuda.amp.autocast(enabled=True):
    #     return original_forward(img, points, metas)
    #* @auto_fp16是否生效,取决于当前模块是否启用混合精度训练
    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    # 仅在 fp16_enabled 开启时，将 img/points 中的 FP32 张量自动转成 FP16。
    # 几何变换等敏感子模块可通过 force_fp32 转回 FP32，因此这是混合精度。
    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        #* =================== 1. 分别提取相机与LiDAR的BEV ===================
        # 相机经过图像backbone、neck和视角变换；LiDAR经过体素化和稀疏编码。
        # 当前配置输出相机[B,1280,180,180]、LiDAR[B,256,180,180]。
        features = []
        for sensor in (self.encoders if self.training else list(self.encoders.keys())[::-1]):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                    **kwargs,
                )
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature) # 2:[(1 256 180 180) (1 1280 180 180)]

        # 推理时原实现反向遍历encoder以调整显存使用，这里恢复正常模态顺序。
        # 这只是改变列表顺序，不是对BEV进行空间翻转。
        if not self.training:
            # avoid OOM
            features = features[::-1]

        #* =================== 2. Fusion：跨模态BEV融合 ===================
        # ConvFuser：通道拼接1536 → 3×3卷积/BN/ReLU → 512通道。
        # 未配置fuser时仅允许单模态输入；当前双模态配置会执行融合。
        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        #* =================== 3. Decoder：处理共享融合特征 ===================
        # backbone为CustomResNet，生成三层BEV；neck为FPN_LSS，融合第0与第2层。
        # 这里是二维残差卷积与上采样，不是Transformer Decoder。
        if self.decoder is not None:
            x = self.decoder["backbone"](x) # 3:[(1 128 180 180) (1 256 180 180) (1 512 90 90)]
            x = self.decoder["neck"](x)
            # 当前输出[B,512,180,180]，随后同时服务于检测与占据预测。

        #* =================== 4. 训练分支：各预测头计算监督损失 ===================
        # model.train()走这里；完整前向检查使用model.eval()，进入下面推理分支。
        if self.training:
            outputs = {}
            for head_type, head in self.heads.items():
                #*--------------------------------------#
                #* 3D预测Head
                if head_type == "object": #* "object"
                    pred_dict = head(x, metas)
                    '''
                    # 在下一行 losses = head.loss(...) 设置断点（必须在执行 loss 前，此时 pred_dict 已生成）。
                    # 选择当前 forward_single 栈帧，复制下面完整 exec(...) 到 VSCode 调试控制台。
                    # 不要复制外层三单引号。只保存 PNG 图片，保存在 out_dir 下的独立时间戳目录。
                    # 图中坐标为当前训练增强后的 LiDAR 坐标；不是原始相机/世界坐标。
                    # 这是检测框检查，不是 occ 体素检查；单帧图片不能代替验证集评估。
                    exec(__import__("textwrap").dedent("""
                    import os
                    from pathlib import Path
                    from datetime import datetime
                    import numpy as np
                    import torch
                    from matplotlib.figure import Figure
                    from matplotlib.backends.backend_agg import FigureCanvasAgg
                    from matplotlib.lines import Line2D

                    sample_idx = 0
                    zoom_radius = 20.0  # 第二张图片显示原点附近 +/-20 m
                    max_text_labels = 60  # 每幅图最多标注多少个框，预测优先标注高分框
                    score_thr = 0.10  # 额外显示阈值；解码仍使用 head.test_cfg 的阈值和 NMS
                    out_dir = Path('/home/wangyushen/Data/Outputs/daocc/w_mask/train_debug/vis_detection')
                    out_dir = out_dir / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_pid' + str(os.getpid()))
                    out_dir.mkdir(parents=True, exist_ok=False)

                    # clone 避免解码中的原地操作修改训练张量；不重新 forward，不切换模型模式。
                    def clone_tree(value):
                        if torch.is_tensor(value):
                            return value.detach().clone()
                        if isinstance(value, dict):
                            return {k: clone_tree(v) for k, v in value.items()}
                        if isinstance(value, list):
                            return [clone_tree(v) for v in value]
                        if isinstance(value, tuple):
                            return tuple(clone_tree(v) for v in value)
                        return value

                    with torch.no_grad():
                        decoded = head.get_bboxes(clone_tree(pred_dict), metas)
                    pred_boxes, pred_scores, pred_labels = decoded[sample_idx]
                    gt_boxes = gt_bboxes_3d[sample_idx].to('cpu')
                    pred_boxes = pred_boxes.to('cpu')
                    gt_array = gt_boxes.tensor.detach().numpy()
                    pred_array = pred_boxes.tensor.detach().numpy()
                    gt_ids = gt_labels_3d[sample_idx].detach().cpu().numpy()
                    pred_ids = pred_labels.detach().cpu().numpy()
                    scores = pred_scores.detach().cpu().numpy()
                    cloud = points[sample_idx].detach().cpu().numpy()[:, :3]
                    class_names = [name for task in head.class_names for name in task]

                    gt_valid = np.isfinite(gt_array[:, :7]).all(axis=1) & (gt_array[:, 3:6] > 0).all(axis=1)
                    pred_valid = np.isfinite(pred_array[:, :7]).all(axis=1) & (pred_array[:, 3:6] > 0).all(axis=1)
                    keep = pred_valid & np.isfinite(scores) & (scores >= score_thr)
                    def corners(boxes, valid):
                        selected = boxes[torch.from_numpy(valid)]
                        return selected.corners.detach().numpy() if len(selected) else np.empty((0, 8, 3))
                    gt_corners, pred_corners = corners(gt_boxes, gt_valid), corners(pred_boxes, keep)
                    cloud = cloud[np.isfinite(cloud).all(axis=1)]
                    cloud = cloud[::max(1, int(np.ceil(len(cloud) / 40000.0)))]
                    limits = head.train_cfg['point_cloud_range']
                    fig = Figure(figsize=(24, 15))
                    FigureCanvasAgg(fig)  # 无需 DISPLAY、GUI 或 plt.show()
                    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
                    groups = [(gt_corners, gt_ids[gt_valid], None, 'limegreen'),
                              (pred_corners, pred_ids[keep], scores[keep], 'orangered')]
                    for col, title in enumerate(['GT', 'Prediction', 'Overlay: green=GT, red=prediction']):
                        selected_groups = [groups[col]] if col < 2 else groups
                        ax = fig.add_subplot(2, 3, col + 1)
                        ax3 = fig.add_subplot(2, 3, col + 4, projection='3d')
                        ax.scatter(cloud[:, 0], cloud[:, 1], s=0.2, c='gray', alpha=0.4)
                        ax3.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=0.2, c='gray', alpha=0.15)
                        for box_corners, ids, conf, color in selected_groups:
                            label_order = np.arange(len(ids)) if conf is None else np.argsort(-conf)
                            label_indices = set(label_order[:max_text_labels].tolist())
                            style = '-' if conf is None else '--'
                            for j, (c, label) in enumerate(zip(box_corners, ids)):
                                outline = c[[0, 3, 7, 4, 0]]
                                ax.plot(outline[:, 0], outline[:, 1], color=color, linewidth=1.0, linestyle=style)
                                center, front = c.mean(axis=0), c[[4,5,6,7]].mean(axis=0)
                                ax.annotate('', xy=front[:2], xytext=center[:2],
                                    arrowprops=dict(arrowstyle='->', color=color, lw=0.8))
                                text = class_names[int(label)] if 0 <= label < len(class_names) else str(label)
                                if conf is not None:
                                    text += ' %.2f' % conf[j]
                                if col < 2 and j in label_indices:  # 叠加图不加文字，避免遮挡
                                    ax.text(center[0], center[1], text, color=color, fontsize=6,
                                        clip_on=True, bbox=dict(facecolor='white', alpha=0.65, edgecolor='none', pad=0.3))
                                for a, b in edges:
                                    ax3.plot(c[[a,b],0], c[[a,b],1], c[[a,b],2], color=color, linewidth=0.6)
                        ax.set(title=title, xlabel='LiDAR x (m)', ylabel='LiDAR y (m)',
                               xlim=(limits[0], limits[3]), ylim=(limits[1], limits[4]))
                        ax.set_aspect('equal')
                        ax.grid(alpha=0.2)
                        ax.plot(0, 0, 'k+', markersize=10)
                        ax.legend(handles=[Line2D([0], [0], color='limegreen', label='GT', linestyle='-'),
                            Line2D([0], [0], color='orangered', label='Prediction', linestyle='--')],
                            loc='upper right', fontsize=8)
                        ax3.set(xlabel='x (m)', ylabel='y (m)', zlabel='z (m)',
                                xlim=(limits[0], limits[3]), ylim=(limits[1], limits[4]), zlim=(limits[2], limits[5]))
                        ax3.view_init(elev=25, azim=-60)
                        # 保持三轴物理比例，避免将高度夸大。旧版 matplotlib 无此接口时沿用默认显示。
                        if hasattr(ax3, 'set_box_aspect'):
                            ax3.set_box_aspect(np.asarray(limits[3:6]) - np.asarray(limits[:3]))
                    invalid_gt = int((~gt_valid).sum())
                    invalid_pred = int((~pred_valid | ~np.isfinite(scores)).sum())
                    invalid_labels = int(((gt_ids < 0) | (gt_ids >= len(class_names))).sum()
                        + ((pred_ids < 0) | (pred_ids >= len(class_names))).sum())
                    caption = ('Augmented LiDAR frame | sample=%d | GT=%d | decoded=%d | shown=%d | score >= %.2f'
                        % (sample_idx, len(gt_array), len(pred_array), int(keep.sum()), score_thr))
                    caption += chr(10) + ('Invalid geometry/score: GT=%d, prediction=%d | invalid class IDs=%d | arrows=heading'
                        % (invalid_gt, invalid_pred, invalid_labels))
                    fig.suptitle(caption, fontsize=13)
                    fig.tight_layout(rect=(0, 0, 1, 0.94))
                    fig.savefig(str(out_dir / 'gt_pred.png'), dpi=180)
                    # 复用画布输出近距离细节图，只调整视野，不再次推理或解码。
                    for ax in fig.axes:
                        ax.set_xlim(-zoom_radius, zoom_radius)
                        ax.set_ylim(-zoom_radius, zoom_radius)
                        if hasattr(ax, 'set_zlim') and hasattr(ax, 'set_box_aspect'):
                            ax.set_box_aspect((2 * zoom_radius, 2 * zoom_radius, limits[5] - limits[2]))
                    fig.suptitle(caption + chr(10) + 'Zoom: +/- %.1f m around LiDAR origin' % zoom_radius, fontsize=13)
                    fig.savefig(str(out_dir / 'gt_pred_zoom.png'), dpi=180)
                    fig.clear()
                    print('Saved PNG images to:', out_dir)
                    """), dict(globals(), **locals()))
                    '''
                    # (10.29 检测GT迁移) object_targets 已由 DataLoader worker 根据
                    # (10.29 检测GT迁移) 最终增强后的 GT 生成，这里传给 CenterHead.loss。
                    losses = head.loss(
                        gt_bboxes_3d,
                        gt_labels_3d,
                        pred_dict,
                        object_targets=kwargs["object_targets"],
                    ) #* gt_bboxes_3d:(45,9)
                elif head_type == "map":
                    losses = head(x, gt_masks_bev)
                #*---------------------------------------#
                #* 占据预测Head
                elif head_type == "occ": #* "occ"
                    # TODO: [yz] this is so weird!
                    '''

                    '''
                    occ_pred = head(x, lidar_aug_matrix, lidar2ego, kwargs['occ_aug_matrix'])
                    losses = head.loss(occ_pred, kwargs['voxel_semantics'], kwargs['mask_camera'])
                else:
                    raise ValueError(f"unsupported head: {head_type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{head_type}/{name}"] = val * self.loss_scale[head_type]
                    else:
                        outputs[f"stats/{head_type}/{name}"] = val
            return outputs
        else:
            #* =================== 5. 推理分支：逐帧整理各任务结果 ===================
            # 每帧一个字典，检测与占据结果写入同一个outputs[k]。
            outputs = [{} for _ in range(batch_size)]
            for head_type, head in self.heads.items():
                if head_type == "object":
                    pred_dict = head(x, metas)
                    # 检测头输出先解码/后处理为三维框，再把框、分数、类别搬到CPU。
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif head_type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                elif head_type == "occ":
                    # TODO: [yz] this is so weird!
                    occ_pred = head(x, lidar_aug_matrix, lidar2ego, kwargs['occ_aug_matrix']) # x:(1 512 180 180)
                    #* head实际类型由配置决定：原配置为BEVOCCHead2D，高斯配置为GaussianOccHead。
                    # 上一句head(...)会执行相应类的forward，主模型不用为高斯另写分支。
                    # get_occ将[B,X,Y,Z,18]分类输出转换为每个体素的类别，放回CPU。
                    occ_pred = head.get_occ(occ_pred)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "occ_pred": occ_pred[k],  # already in cpu
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {head_type}")
            return outputs
