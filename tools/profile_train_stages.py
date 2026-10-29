#!/usr/bin/env python
"""短迭代分析 DAOcc 训练各阶段耗时。

该脚本只使用一张 GPU，不注册正式 Runner 的 EMA、验证、保存和日志 Hook，目标是
定位核心训练流水线的瓶颈：等待 DataLoader、网络前向、loss、反向及优化器更新。
前向/loss/反向的 GPU 时间使用 CUDA Event；直接用 time.time() 会因 CUDA 异步执行
得到错误结果。为读取每轮结果，脚本会在轮末同步 GPU，因此只应用于测速，不用于
正式训练吞吐对比。

示例（物理 4 号 GPU 映射为进程内 0 号 GPU）：
    CUDA_VISIBLE_DEVICES=4 python tools/profile_train_stages.py \
        configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_inin.yaml \
        --gpu 0 --batch-size 4 --workers 4 \
        --warmup-iters 10 --profile-iters 50 \
        --output /tmp/daocc_train_timing.json
"""

import argparse
import json
import math
import random
import statistics
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import build_optimizer, load_checkpoint, wrap_fp16_model
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torchpack.utils.config import Config as TorchpackConfig

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import convert_sync_batchnorm, recursive_eval


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="训练 YAML 配置")
    parser.add_argument("--gpu", type=int, default=0, help="当前进程可见的 GPU 编号")
    parser.add_argument("--batch-size", type=int, help="覆盖每卡 batch size")
    parser.add_argument("--workers", type=int, help="覆盖每卡 DataLoader worker 数")
    parser.add_argument("--warmup-iters", type=int, default=10, help="不计入统计的预热轮数")
    parser.add_argument("--profile-iters", type=int, default=50, help="正式统计轮数")
    parser.add_argument("--checkpoint", help="可选模型 checkpoint；不恢复优化器状态")
    parser.add_argument("--fp32", action="store_true", help="关闭配置中的 FP16 混合精度")
    parser.add_argument("--output", help="可选 JSON 结果路径")
    parser.add_argument("--seed", type=int, help="覆盖配置中的随机种子")
    args, config_overrides = parser.parse_known_args()

    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size 必须为正整数")
    if args.workers is not None and args.workers < 0:
        parser.error("--workers 必须为非负整数")
    if args.warmup_iters < 0 or args.profile_iters <= 0:
        parser.error("--warmup-iters 必须非负，--profile-iters 必须为正整数")
    return args, config_overrides


def load_config(path, overrides):
    # 使用局部 TorchpackConfig，避免依赖 tools/train.py 的全局 configs 对象。
    raw = TorchpackConfig()
    raw.load(path, recursive=True)
    raw.update(overrides)
    return Config(recursive_eval(raw), filename=path)


def make_scaler(fp16_cfg):
    """按照 MMCV Fp16OptimizerHook 的规则创建 GradScaler。"""
    loss_scale = fp16_cfg.get("loss_scale", 512.0)
    if loss_scale == "dynamic":
        return GradScaler()
    if isinstance(loss_scale, Mapping):
        return GradScaler(**dict(loss_scale))
    if isinstance(loss_scale, (float, int)):
        return GradScaler(init_scale=float(loss_scale))
    raise TypeError("fp16.loss_scale 必须是数值、'dynamic' 或字典")


class CudaRegionRecorder:
    """包装模型子阶段，收集当前 iteration 的 CUDA Event。"""

    def __init__(self):
        self.events = []

    @staticmethod
    def _event():
        return torch.cuda.Event(enable_timing=True)

    def reset(self):
        self.events = []

    def _wrap(self, owner, attribute, stage_name):
        original = getattr(owner, attribute)

        def timed(*args, _original=original, _stage=stage_name, **kwargs):
            start, end = self._event(), self._event()
            start.record()
            result = _original(*args, **kwargs)
            end.record()
            self.events.append((_stage, start, end))
            return result

        setattr(owner, attribute, timed)

    def install(self, module):
        # 相机分支：图像 backbone -> FPN neck -> 视角变换到 BEV。
        camera = module.encoders["camera"]
        self._wrap(camera["backbone"], "forward", "forward.camera_backbone")
        self._wrap(camera["neck"], "forward", "forward.camera_neck")
        self._wrap(camera["vtransform"], "forward", "forward.camera_vtransform")

        # LiDAR 分支：体素化（含整理 voxel 输出）-> 稀疏骨干 -> 可选 BEV 模块。
        # camera-only 配置不构建 lidar encoder，因此相应指标保持为 0。
        if "lidar" in module.encoders:
            lidar = module.encoders["lidar"]
            self._wrap(module, "voxelize", "forward.lidar_voxelize")
            self._wrap(lidar["backbone"], "forward", "forward.lidar_backbone")
            if "bev_backbone" in lidar:
                self._wrap(lidar["bev_backbone"], "forward", "forward.lidar_bev_backbone")
            if "bev_neck" in lidar:
                self._wrap(lidar["bev_neck"], "forward", "forward.lidar_bev_neck")

        if module.fuser is not None:
            self._wrap(module.fuser, "forward", "forward.fuser")
        if module.decoder is not None:
            self._wrap(module.decoder["backbone"], "forward", "forward.decoder_backbone")
            self._wrap(module.decoder["neck"], "forward", "forward.decoder_neck")

        # forward_single 中各 head 的预测和 loss 交叉执行，分别包装才能独立统计。
        for head_name, head in module.heads.items():
            self._wrap(head, "forward", "forward.%s_head" % head_name)
            self._wrap(head, "loss", "loss.%s" % head_name)

        self._wrap(module, "_parse_losses", "loss.parse")

    def elapsed_by_stage(self):
        elapsed = {}
        for name, start, end in self.events:
            elapsed[name] = elapsed.get(name, 0.0) + start.elapsed_time(end)
        return elapsed


def percentile(values, percent):
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(rows, batch_size, precision):
    metric_names = [
        "data_wait_ms",
        "network_forward_ms",
        "camera_backbone_ms",
        "camera_neck_ms",
        "camera_vtransform_ms",
        "lidar_voxelize_ms",
        "lidar_backbone_ms",
        "lidar_bev_backbone_ms",
        "lidar_bev_neck_ms",
        "fuser_ms",
        "decoder_backbone_ms",
        "decoder_neck_ms",
        "object_head_forward_ms",
        "occ_head_forward_ms",
        "forward_other_ms",
        "head_loss_ms",
        "object_loss_ms",
        "occ_loss_ms",
        "loss_other_ms",
        "loss_parse_ms",
        "loss_total_ms",
        "forward_loss_total_ms",
        "backward_optimizer_ms",
        "core_cuda_ms",
        "iteration_wall_ms",
        "peak_memory_gib",
    ]
    metrics = {}
    for name in metric_names:
        values = [row[name] for row in rows]
        metrics[name] = {
            "mean": statistics.fmean(values),
            "p50": percentile(values, 50),
            "p90": percentile(values, 90),
            "min": min(values),
            "max": max(values),
        }

    mean_wall_seconds = metrics["iteration_wall_ms"]["mean"] / 1000.0
    return {
        "precision": precision,
        "profile_iters": len(rows),
        "batch_size_per_gpu": batch_size,
        "mean_samples_per_second": batch_size / mean_wall_seconds,
        "metrics": metrics,
        "iterations": rows,
    }


def print_summary(summary):
    labels = [
        ("data_wait_ms", "等待 DataLoader"),
        ("network_forward_ms", "网络前向（含数据搬运）"),
        ("camera_backbone_ms", "  相机 backbone"),
        ("camera_neck_ms", "  相机 neck"),
        ("camera_vtransform_ms", "  相机视角变换"),
        ("lidar_voxelize_ms", "  LiDAR 体素化"),
        ("lidar_backbone_ms", "  LiDAR backbone"),
        ("lidar_bev_backbone_ms", "  LiDAR BEV backbone"),
        ("lidar_bev_neck_ms", "  LiDAR BEV neck"),
        ("fuser_ms", "  BEV 融合"),
        ("decoder_backbone_ms", "  Decoder backbone"),
        ("decoder_neck_ms", "  Decoder neck"),
        ("object_head_forward_ms", "  Object head 预测"),
        ("occ_head_forward_ms", "  Occupancy head 预测"),
        ("forward_other_ms", "  前向其他/数据搬运"),
        ("head_loss_ms", "各预测头 loss"),
        ("object_loss_ms", "  Object loss"),
        ("occ_loss_ms", "  Occupancy loss"),
        ("loss_other_ms", "  其他 head loss"),
        ("loss_parse_ms", "loss 汇总/日志值"),
        ("loss_total_ms", "loss 合计"),
        ("forward_loss_total_ms", "前向 + loss 总计"),
        ("backward_optimizer_ms", "反向 + 裁剪 + optimizer"),
        ("core_cuda_ms", "核心 CUDA 总计"),
        ("iteration_wall_ms", "iteration 墙钟总计"),
        ("peak_memory_gib", "峰值显存 GiB"),
    ]
    print("\nDAOcc 单卡训练分阶段测速")
    print("precision=%s, batch=%d, measured_iters=%d" % (
        summary["precision"], summary["batch_size_per_gpu"], summary["profile_iters"]))
    print("%-30s %12s %12s %12s" % ("阶段", "平均", "P50", "P90"))
    for key, label in labels:
        item = summary["metrics"][key]
        unit = "GiB" if key == "peak_memory_gib" else "ms"
        print("%-30s %10.3f %-3s %10.3f %10.3f" % (
            label, item["mean"], unit, item["p50"], item["p90"]))
    print("平均吞吐：%.3f samples/s（单卡，包含 DataLoader 等待和轮末同步）" %
          summary["mean_samples_per_second"])
    print("说明：forward 是用完整 forward/loss CUDA 时间减去 loss CUDA 时间得到；")
    print("      iteration 墙钟时间还包含 Python、梯度清零及必要的 CUDA 同步开销。")


def main():
    args, overrides = parse_args()
    cfg = load_config(args.config, overrides)

    if not torch.cuda.is_available():
        raise RuntimeError("该测速脚本需要 CUDA GPU")
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", False))

    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    batch_size = args.batch_size or int(cfg.data.samples_per_gpu)
    workers = args.workers if args.workers is not None else int(cfg.data.workers_per_gpu)

    print("构建训练数据集……")
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=batch_size,
        workers_per_gpu=workers,
        num_gpus=1,
        dist=False,
        shuffle=True,
        seed=seed,
    )

    print("构建模型和优化器……")
    model = build_model(cfg.model)
    model.init_weights()
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location="cpu")
    if cfg.get("sync_bn", None):
        sync_cfg = cfg.sync_bn if isinstance(cfg.sync_bn, dict) else {"exclude": []}
        model = convert_sync_batchnorm(model, exclude=sync_cfg.get("exclude", []))

    recorder = CudaRegionRecorder()
    recorder.install(model)

    model = MMDataParallel(model.cuda(), device_ids=[args.gpu])
    optimizer = build_optimizer(model, cfg.optimizer)

    fp16_cfg = None if args.fp32 else cfg.get("fp16", None)
    scaler = None
    if fp16_cfg is not None:
        # 与正式 Fp16OptimizerHook.before_run() 一致：转换模型并启用 auto_fp16。
        wrap_fp16_model(model)
        scaler = make_scaler(fp16_cfg)
        precision = "FP16/FP32 mixed"
    else:
        precision = "FP32"

    grad_clip = cfg.get("optimizer_config", {}).get("grad_clip", None)
    model.train()
    iterator = iter(data_loader)
    total_iters = args.warmup_iters + args.profile_iters
    rows = []

    print("开始测速：warmup=%d，profile=%d，batch=%d，workers=%d，precision=%s" %
          (args.warmup_iters, args.profile_iters, batch_size, workers, precision))

    for iteration in range(total_iters):
        iteration_wall_start = time.perf_counter()
        data_start = time.perf_counter()
        try:
            data_batch = next(iterator)
        except StopIteration:
            iterator = iter(data_loader)
            data_batch = next(iterator)
        data_wait_ms = (time.perf_counter() - data_start) * 1000.0

        recorder.reset()
        torch.cuda.reset_peak_memory_stats(args.gpu)
        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        backward_start = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)

        forward_start.record()
        outputs = model.train_step(data_batch, optimizer)
        forward_end.record()

        backward_start.record()
        model.zero_grad()
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(outputs["loss"]).backward()
            scaler.unscale_(optimizer)
            if grad_clip is not None:
                clip_grad_norm_(model.parameters(), **dict(grad_clip))
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs["loss"].backward()
            if grad_clip is not None:
                clip_grad_norm_(model.parameters(), **dict(grad_clip))
            optimizer.step()
        backward_end.record()

        # Event 的 elapsed_time 只在事件执行完成后有效。同步会干扰流水并降低吞吐，
        # 所以本脚本只用于短时测速，不可将这里的结果当作无探针的极限吞吐。
        torch.cuda.synchronize(args.gpu)
        iteration_wall_ms = (time.perf_counter() - iteration_wall_start) * 1000.0

        forward_loss_total_ms = forward_start.elapsed_time(forward_end)
        stage_ms = recorder.elapsed_by_stage()
        object_loss_ms = stage_ms.get("loss.object", 0.0)
        occ_loss_ms = stage_ms.get("loss.occ", 0.0)
        all_head_loss_ms = sum(
            value for name, value in stage_ms.items()
            if name.startswith("loss.") and name != "loss.parse"
        )
        loss_other_ms = max(0.0, all_head_loss_ms - object_loss_ms - occ_loss_ms)
        head_loss_ms = all_head_loss_ms
        loss_parse_ms = stage_ms.get("loss.parse", 0.0)
        loss_total_ms = head_loss_ms + loss_parse_ms
        network_forward_ms = max(0.0, forward_loss_total_ms - loss_total_ms)

        forward_values = {
            "camera_backbone_ms": stage_ms.get("forward.camera_backbone", 0.0),
            "camera_neck_ms": stage_ms.get("forward.camera_neck", 0.0),
            "camera_vtransform_ms": stage_ms.get("forward.camera_vtransform", 0.0),
            "lidar_voxelize_ms": stage_ms.get("forward.lidar_voxelize", 0.0),
            "lidar_backbone_ms": stage_ms.get("forward.lidar_backbone", 0.0),
            "lidar_bev_backbone_ms": stage_ms.get("forward.lidar_bev_backbone", 0.0),
            "lidar_bev_neck_ms": stage_ms.get("forward.lidar_bev_neck", 0.0),
            "fuser_ms": stage_ms.get("forward.fuser", 0.0),
            "decoder_backbone_ms": stage_ms.get("forward.decoder_backbone", 0.0),
            "decoder_neck_ms": stage_ms.get("forward.decoder_neck", 0.0),
            "object_head_forward_ms": stage_ms.get("forward.object_head", 0.0),
            "occ_head_forward_ms": stage_ms.get("forward.occ_head", 0.0),
        }
        forward_other_ms = max(0.0, network_forward_ms - sum(forward_values.values()))
        backward_optimizer_ms = backward_start.elapsed_time(backward_end)
        peak_memory_gib = torch.cuda.max_memory_allocated(args.gpu) / (1024.0 ** 3)

        if iteration >= args.warmup_iters:
            row = {
                "iteration": iteration - args.warmup_iters + 1,
                "data_wait_ms": data_wait_ms,
                "network_forward_ms": network_forward_ms,
                **forward_values,
                "forward_other_ms": forward_other_ms,
                "head_loss_ms": head_loss_ms,
                "object_loss_ms": object_loss_ms,
                "occ_loss_ms": occ_loss_ms,
                "loss_other_ms": loss_other_ms,
                "loss_parse_ms": loss_parse_ms,
                "loss_total_ms": loss_total_ms,
                "forward_loss_total_ms": forward_loss_total_ms,
                "backward_optimizer_ms": backward_optimizer_ms,
                "core_cuda_ms": forward_loss_total_ms + backward_optimizer_ms,
                "iteration_wall_ms": iteration_wall_ms,
                "peak_memory_gib": peak_memory_gib,
            }
            rows.append(row)

        is_warmup = iteration < args.warmup_iters
        phase = "warmup" if is_warmup else "profile"
        phase_index = iteration + 1 if is_warmup else iteration - args.warmup_iters + 1
        phase_total = args.warmup_iters if is_warmup else args.profile_iters
        print("[%s %d/%d] data %.1f ms | forward+loss %.1f ms | "
              "backward+step %.1f ms | wall %.1f ms" % (
                  phase, phase_index, phase_total,
                  data_wait_ms, forward_loss_total_ms,
                  backward_optimizer_ms, iteration_wall_ms))

    summary = summarize(rows, batch_size, precision)
    print_summary(summary)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
        print("JSON 结果已保存：%s" % output_path)


if __name__ == "__main__":
    main()
