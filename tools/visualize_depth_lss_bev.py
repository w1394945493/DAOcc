#!/usr/bin/env python3
"""并排展示DepthLSS相机BEV、LiDAR BEV和fuser融合BEV，按场景保存。"""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from visualize_modality_comparison import clear_pretrained, scene_lookup, select_indices


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--checkpoint",
                        help="可选：训练后的DepthLSS完整模型；不传则随机初始化，不读取预训练权重")
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
    #* 融合前分支均使用[B,C,Y,X]；通道绝对值取平均得到[Y,X]，不再转置。
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


def draw_features(heats, stats, extent, scale, path, random_init=False):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    #* 独立百分位色标用于观察结构；共享色标仅表示特征数值幅度，不能视为可信度。
    shared = np.percentile(np.concatenate([h.ravel() for h in heats]), [1, 99])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, heat, stat, title in zip(axes, heats, stats, ["DepthLSS camera BEV", "LiDAR BEV", "Fused BEV (fuser output)"]):
        low, high = shared if scale == "shared" else np.percentile(heat, [1, 99])
        stat.update(color_min=float(low), color_max=float(high))
        ax.imshow(heat, origin="lower", extent=extent, cmap="jet",
                  vmin=float(low), vmax=max(float(high), float(low)+1e-12), interpolation="nearest")
        ax.set_title(title + (" (random init)" if random_init else ""), fontsize=11, pad=3)
        ax.set_aspect("equal")
        ax.set_axis_off()
    fig.subplots_adjust(left=.002, right=.998, bottom=.002, top=.95, wspace=.012)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


def main():
    args = parse_args()
    import torch
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
    if vt.type != "DepthLSSTransformDAOcc":
        raise ValueError("请使用DepthLSSTransformDAOcc配置，确保相机与LiDAR的BEV轴一致")
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

    #* 严格加载指定模型，不允许旧BEVTransformV2权重缺失depthnet时继续随机推理。
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
    model.cuda(args.gpu)
    extent = [vt.xbound[0], vt.xbound[1], vt.ybound[0], vt.ybound[1]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(config=str(Path(args.config).resolve()),
                  checkpoint=str(Path(args.checkpoint).expanduser().resolve()) if args.checkpoint else None,
                  initialization="checkpoint" if args.checkpoint else "random",
                  precision="fp32", selected_indices=indices, feature_scale=args.feature_scale,
                  coordinates="LiDAR coordinates; horizontal X, vertical Y; origin lower",
                  extent=extent, reduction="mean(abs(channels)); no ego resampling",
                  feature_stage="camera/lidar branch outputs and fuser output; before decoder", frames=[])
    with torch.no_grad():
        for index, batch in zip(indices, tqdm(loader, total=len(indices), desc="DepthLSS分支对照")):
            data = scatter(batch, [args.gpu])[0]
            #* 直接调用正式模型的分支提取及融合方法，不执行decoder或Occ预测。
            # 验证pipeline保留真实图像增强矩阵和聚合点云，与模型推理输入一致。
            camera = model.extract_camera_features(
                data["img"], data["points"], data["camera2ego"], data["lidar2ego"],
                data["lidar2camera"], data["lidar2image"], data["camera_intrinsics"],
                data["camera2lidar"], data["img_aug_matrix"], data["lidar_aug_matrix"], data["metas"])
            camera_heat, camera_stats = feature_heat(camera)
            lidar = model.extract_lidar_features(data["points"])
            lidar_heat, lidar_stats = feature_heat(lidar)
            if camera_heat.shape != lidar_heat.shape:
                raise ValueError("两分支BEV空间尺寸不一致")
            #* 按正式BEVFusion的encoder顺序融合，避免把LiDAR和相机通道顺序颠倒。
            # 顺序以model.encoders实际构造结果为准，不根据YAML文本顺序推测。
            branch_features = {"camera": camera, "lidar": lidar}
            fused = model.fuser([branch_features[name] for name in model.encoders])
            fused_heat, fused_stats = feature_heat(fused)
            del camera, lidar, fused, branch_features
            token = dataset.data_infos[index]["token"]
            scene, scene_token, number = lookup[token]
            path = args.output_dir / scene / f"frame_{number:03d}_{token}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            stats = [camera_stats, lidar_stats, fused_stats]
            draw_features([camera_heat, lidar_heat, fused_heat], stats, extent, args.feature_scale, path,
                          random_init=args.checkpoint is None)
            report["frames"].append(dict(index=index, token=token, scene=scene,
                scene_token=scene_token, scene_frame_number=number, path=str(path.resolve()),
                camera=stats[0], lidar=stats[1], fusion=stats[2]))
            (args.output_dir / "depth_lss_bev_manifest.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            del data, batch
    print(f"已保存{len(report['frames'])}帧：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
