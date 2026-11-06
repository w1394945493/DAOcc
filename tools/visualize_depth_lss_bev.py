#!/usr/bin/env python3
"""并排展示DepthLSS相机BEV、LiDAR BEV、融合BEV、Occ预测和GT，按场景保存。"""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from visualize_modality_comparison import (
    clear_pretrained, scene_lookup, select_indices, first_tensor,
    load_occ_colors, occupancy_rgb,
)


def parse_args(description=None):
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("config")
    parser.add_argument("--checkpoint",
                        help="可选：与配置匹配的完整模型；不传则随机初始化，不读取预训练权重")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=20,
                        help="默认全验证集均匀20帧；多场景模式下为每场景帧数")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--scene")
    group.add_argument("--scene-index", type=int)
    parser.add_argument("--num-scenes", type=int)
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--list-scenes", action="store_true")
    parser.add_argument("--feature-scale", choices=["independent", "shared"], default="independent")
    parser.add_argument("--camera-mask", action="store_true",
                        help="预测和GT均只投影mask_camera内的占据体素；不影响BEV热力图")
    args = parser.parse_args()
    if args.num_samples < 1 or args.workers < 0:
        parser.error("num-samples必须为正，workers不能为负")
    if args.scene_index is not None and args.scene_index < 0:
        parser.error("scene-index不能为负")
    if args.num_scenes is not None and (args.num_scenes < 1 or args.scene is not None):
        parser.error("num-scenes必须为正，且不能与scene同时使用")
    return args


def feature_heat(feature):
    import torch
    #* 此函数接收已按Occ head流程对齐到ego网格的[B,C,Y,X]，再沿通道求平均。
    # 这是响应强度而非语义类别或占据概率；通道数不同仍可比较空间响应分布。
    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError(f"预期单帧BEV，实际为{tuple(feature.shape)}")
    if not torch.isfinite(feature).all():
        raise ValueError("BEV含NaN/Inf，停止可视化")
    heat = feature[0].float().abs().mean(0).cpu().numpy()
    stats = dict(shape=list(feature.shape), mean=float(feature.mean()),
                 std=float(feature.float().std()), abs_max=float(feature.abs().max()),
                 heat_mean=float(heat.mean()), heat_max=float(heat.max()))
    return heat, stats


def aligned_feature_heat(feature, sampler, data):
    """仅对可视化副本执行Occ几何对齐，不修改送往decoder的真实特征。"""
    #* 与Occ head及visualize_modality_comparison一致：先交换原始张量空间轴，
    # 再利用当前帧的LiDAR增强、lidar2ego和Occ增强矩阵采样到自车网格。
    # 必须先变换完整通道特征，再取绝对值均值；不能先生成热图后随意旋转。
    aligned = sampler(
        feature.detach().float().transpose(-1, -2).contiguous(),
        data["lidar_aug_matrix"].float(), data["lidar2ego"].float(),
        data["occ_aug_matrix"].float(),
    )
    heat, stats = feature_heat(aligned)
    stats["source_shape"] = list(feature.shape)
    return heat, stats


def draw_features(heats, stats, extent, scale, path, pred_rgb, gt_rgb, random_init=False, camera_mask=False, camera_title="DepthLSS camera BEV (ego)"):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    #* 独立百分位色标用于观察结构；共享色标仅表示特征数值幅度，不能视为可信度。
    shared = np.percentile(np.concatenate([h.ravel() for h in heats]), [1, 99])
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for ax, heat, stat, title in zip(axes, heats, stats, [camera_title, "LiDAR BEV (ego)", "Fused BEV (ego, before decoder)"]):
        low, high = shared if scale == "shared" else np.percentile(heat, [1, 99])
        #* 与visualize_modality_comparison保持一致：先按百分位归一化，再截断到[0,1]。
        # 低于第1百分位的响应显示为0，高于第99百分位的响应显示为1；
        # 分母设置下限，避免常量特征图因high==low产生除零。
        normalized = np.clip((heat - low) / max(float(high - low), 1e-12), 0, 1)
        # 保留归一化前的阈值，便于回查原始响应幅度；绘图色标固定为[0,1]。
        stat.update(color_min=float(low), color_max=float(high))
        ax.imshow(normalized, origin="lower", extent=extent, cmap="jet",
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title + (" (random init)" if random_init else ""), fontsize=11, pad=3)
        ax.set_aspect("equal")
        ax.set_axis_off()
    #* Occ沿用模态对比脚本：每个XY柱取最高有效占据体素，按语义类别着色。
    # 五幅图均按ego网格显示，横轴X、纵轴Y，范围[-40,40]米。
    # Occ按类别配色，不对语义颜色做热力图归一化。
    suffix = " (camera mask)" if camera_mask else ""
    for ax, rgb, title in zip(axes[3:], [pred_rgb, gt_rgb], ["Occ prediction (ego)", "Occ GT (ego)"]):
        ax.imshow(rgb, origin="lower", extent=extent, interpolation="nearest")
        ax.set_title(title + suffix + (" (random init)" if random_init and ax is axes[3] else ""),
                     fontsize=11, pad=3)
        ax.set_aspect("equal")
        ax.set_axis_off()
    fig.subplots_adjust(left=.002, right=.998, bottom=.002, top=.95, wspace=.012)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


def main(expected_vtransform="DepthLSSTransformDAOcc", method_name="DepthLSS",
         manifest_name="depth_lss_bev_manifest.json", description=None, frame_callback=None):
    #* 两种方法共用唯一实现，确保采样、FP32推理、ego对齐、归一化与Occ投影一致。
    # 原DAOcc入口只指定视角变换类型和显示名称，不复制或改变可视化流程。
    args = parse_args(description)
    import torch
    import numpy as np
    from mmcv import Config
    from mmcv.parallel import scatter
    from mmcv.runner import load_checkpoint
    from torchpack.utils.config import Config as TorchpackConfig
    from torch.utils.data import Subset
    from tqdm import tqdm
    from mmdet.apis import set_random_seed
    from mmdet3d.utils import recursive_eval
    from mmdet3d.datasets import build_dataset, build_dataloader
    from mmdet3d.models import build_model

    set_random_seed(0, deterministic=True)
    raw = TorchpackConfig()
    raw.load(str(Path(args.config).resolve()), recursive=True)
    cfg = Config(recursive_eval(raw))
    vt = cfg.model.encoders.camera.vtransform
    if vt.type != expected_vtransform:
        raise ValueError(f"{method_name}入口要求{expected_vtransform}配置，实际为{vt.type}")
    if cfg.model.encoders.get("lidar") is None:
        raise ValueError("需要同时包含camera与lidar分支")
    data_cfg = deepcopy(cfg.data.val)
    data_cfg.test_mode = True
    data_cfg.pop("samples_per_gpu", None)
    dataset = build_dataset(data_cfg)
    lookup = scene_lookup(dataset)
    #* 只统计当前验证集包含的场景，不使用nuScenes全量scene.json的索引。
    # 字典保留插入顺序，因此索引由当前验证集data_infos中的首次出现顺序决定。
    scenes = {}
    for info in dataset.data_infos:
        scene_name, scene_token, _ = lookup[info["token"]]
        if scene_token not in scenes:
            scenes[scene_token] = dict(name=scene_name, token=scene_token, frames=0)
        scenes[scene_token]["frames"] += 1
    scene_list = list(scenes.values())
    if args.list_scenes:
        print("验证集场景索引（从0开始） | 场景名 | 验证帧数 | scene token")
        for i, scene in enumerate(scene_list):
            print(f"{i:3d} | {scene['name']} | {scene['frames']} | {scene['token']}")
        return
    if args.scene_index is not None and args.num_scenes is None:
        if args.scene_index >= len(scene_list):
            raise ValueError(f"scene-index={args.scene_index}越界；验证集共{len(scene_list)}个场景")
        selected_scene = scene_list[args.scene_index]
        args.scene = selected_scene["token"]
        print(f"选择验证集场景[{args.scene_index}] {selected_scene['name']}，"
              f"共{selected_scene['frames']}帧", flush=True)
    candidates = [i for i, info in enumerate(dataset.data_infos)
                  if args.scene is None or args.scene in lookup[info["token"]][:2]]
    if not candidates:
        raise ValueError(f"没有找到所选场景的验证帧：{args.scene}")
    indices = candidates if args.all_frames else [candidates[i] for i in
               select_indices(len(candidates), args.num_samples)]
    if args.num_scenes is not None:
        #* 多场景模式：先选连续M个场景，再分别在各场景内均匀抽N帧。
        # 不能把M个场景混在一起抽N帧，否则无法保证每个场景都有相同样本数。
        start = args.scene_index if args.scene_index is not None else 0
        stop = start + args.num_scenes
        if stop > len(scene_list):
            raise ValueError(f"请求场景索引[{start},{stop})，但验证集只有{len(scene_list)}个场景")
        indices = []
        for scene_index in range(start, stop):
            scene = scene_list[scene_index]
            frame_indices = [i for i, info in enumerate(dataset.data_infos)
                             if lookup[info["token"]][1] == scene["token"]]
            # 按场景内原始关键帧序号排序，确保均匀采样沿时间顺序进行。
            frame_indices.sort(key=lambda i: lookup[dataset.data_infos[i]["token"]][2])
            chosen = frame_indices if args.all_frames else [frame_indices[j] for j in
                      select_indices(len(frame_indices), args.num_samples)]
            indices.extend(chosen)
            print(f"场景[{scene_index}] {scene['name']}：{len(frame_indices)}帧中选择{len(chosen)}帧",
                  flush=True)
    torch.cuda.set_device(args.gpu)
    loader = build_dataloader(Subset(dataset, indices), samples_per_gpu=1,
                             workers_per_gpu=args.workers, dist=False, shuffle=False)

    #* 严格加载与配置匹配的完整模型，禁止缺失/不匹配的层悄悄使用随机权重。
    model_cfg = deepcopy(cfg.model)
    clear_pretrained(model_cfg)
    model = build_model(model_cfg)
    #* 不指定checkpoint时显式初始化，并已清除配置内的预训练路径，避免隐式加载。
    # 随机特征仅用于检查流程、形状及可视化，不代表训练后的语义或融合效果。
    if args.checkpoint:
        load_checkpoint(model, str(Path(args.checkpoint).expanduser()), map_location="cpu", strict=True)
    else:
        model.init_weights()
        print("未指定checkpoint：使用随机初始化，图中将标注random init。", flush=True)
    model.float().eval()
    for module in model.modules():
        if hasattr(module, "fp16_enabled"):
            module.fp16_enabled = False
    if model.fuser is None:
        raise ValueError("需要配置fuser才能生成融合BEV")
    if "occ" not in model.heads:
        raise ValueError("需要occ head才能生成占据预测")
    #* 复用实际Occ head的几何采样器，但移除可学习通道映射和额外上采样。
    # 三组特征通道数不同；这里只统一空间网格，不改变特征的语义内容或模型权重。
    transform = cfg.model.heads.occ.coordinate_transform
    if (transform.type != "CrossCoordinateSample" or
            list(transform.point_num) != [200, 200, 1] or
            list(transform.point_range) != [-40., 40., -40., 40., 0., 0.]):
        raise ValueError("仅支持Occ3D自车系±40m、200×200的坐标变换")
    sampler = deepcopy(model.heads["occ"].coordinate_transform)
    if sampler.point_type != "ego":
        raise ValueError("Occ几何采样器必须输出ego坐标系")
    sampler.transfer_conv, sampler.extra_up = None, False
    sampler.float().eval().cuda(args.gpu)
    colors = load_occ_colors(ROOT / "tools/vis_occ3d.py")
    model.cuda(args.gpu)
    extent = [-40, 40, -40, 40]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(method=method_name, vtransform=vt.type, config=str(Path(args.config).resolve()),
                  checkpoint=str(Path(args.checkpoint).expanduser().resolve()) if args.checkpoint else None,
                  initialization="checkpoint" if args.checkpoint else "random",
                  precision="fp32", selected_indices=indices, feature_scale=args.feature_scale,
                  coordinates="ego; horizontal X, vertical Y; origin lower",
                  extent=extent, reduction="mean(abs(channels)) after Occ-head geometric resampling",
                  feature_stage="camera/lidar branch outputs and fuser output; before decoder",
                  camera_mask=args.camera_mask, occ_extent=[-40, 40, -40, 40],
                  occ_coordinates="ego; horizontal X, vertical Y; origin lower",
                  occ_projection="highest valid occupied voxel per XY column", frames=[])
    with torch.no_grad():
        for index, batch in zip(indices, tqdm(loader, total=len(indices), desc=f"{method_name}分支对照")):
            data = scatter(batch, [args.gpu])[0]
            #* 一次正式推理同时生成Occ预测和三组BEV热力图，不重复执行encoder。
            # 正式forward在调用fuser之前已将分支列表恢复为model.encoders的顺序。
            # Hook立即压缩为CPU二维热图，避免为可视化长时间保留完整GPU特征。
            captured = {}
            def capture_fusion(module, inputs, output):
                branches = inputs[0]
                if len(branches) != len(model.encoders):
                    raise ValueError("融合输入数量与encoder数量不一致")
                for name, feature in zip(model.encoders, branches):
                    captured[name] = aligned_feature_heat(feature, sampler, data)
                captured["fusion"] = aligned_feature_heat(output, sampler, data)
            handle = model.fuser.register_forward_hook(capture_fusion)
            try:
                outputs = model(**data)
            finally:
                handle.remove()
            camera_heat, camera_stats = captured["camera"]
            lidar_heat, lidar_stats = captured["lidar"]
            fused_heat, fused_stats = captured["fusion"]
            if camera_heat.shape != lidar_heat.shape:
                raise ValueError("两分支BEV空间尺寸不一致")

            #* GT和预测同帧匹配，保持Occ3D的[X,Y,Z]=[200,200,16]排列。
            sem = first_tensor(data["voxel_semantics"])
            if sem.ndim == 4:
                sem = sem[0]
            sem = sem.detach().cpu().numpy()
            prediction = outputs[0]["occ_pred"]
            if torch.is_tensor(prediction):
                prediction = prediction.detach().cpu().numpy()
            prediction = np.asarray(prediction)
            if sem.shape != (200, 200, 16) or prediction.shape != sem.shape:
                raise ValueError(f"Occ3D形状异常：GT={sem.shape}, prediction={prediction.shape}")
            # 掩码同时应用于预测和GT；空体素、忽略体素及掩码外区域保持白色。
            mask = None
            if args.camera_mask:
                mask = first_tensor(data["mask_camera"])
                if mask.ndim == 4:
                    mask = mask[0]
                mask = mask.detach().cpu().numpy()
                if mask.shape != sem.shape:
                    raise ValueError("mask_camera与GT形状不一致")
            pred_rgb = occupancy_rgb(prediction, colors, mask)
            gt_rgb = occupancy_rgb(sem, colors, mask)
            del outputs, captured
            token = dataset.data_infos[index]["token"]
            scene, scene_token, number = lookup[token]
            path = args.output_dir / scene / f"frame_{number:03d}_{token}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            stats = [camera_stats, lidar_stats, fused_stats]
            if frame_callback is None:
                draw_features([camera_heat, lidar_heat, fused_heat], stats, extent, args.feature_scale, path,
                              pred_rgb, gt_rgb, random_init=args.checkpoint is None,
                              camera_mask=args.camera_mask, camera_title=f"{method_name} camera BEV (ego)")
            else:
                #* 统一对照入口可接收同一批数组，直接绘制2×5图，无需重复推理。
                path = Path(frame_callback(
                    frame=dict(index=index, token=token, scene=scene, scene_token=scene_token,
                               scene_frame_number=number),
                    heats=[camera_heat, lidar_heat, fused_heat], stats=stats, extent=extent,
                    pred_rgb=pred_rgb, gt_rgb=gt_rgb, random_init=args.checkpoint is None,
                ))
            report["frames"].append(dict(index=index, token=token, scene=scene,
                scene_token=scene_token, scene_frame_number=number, path=str(path.resolve()),
                camera=stats[0], lidar=stats[1], fusion=stats[2], occ_shape=list(sem.shape)))
            (args.output_dir / manifest_name).write_text(
                json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            del data, batch
    print(f"已保存{len(report['frames'])}帧：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
