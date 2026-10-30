#!/usr/bin/env python3
"""单卡短训练，定位 DAOcc 混合精度训练中的 NaN/Inf 来源。

默认加载用户指定的完整训练配置，复现正式训练使用的
``wrap_fp16_model + GradScaler``，但不启动 MMCV Runner、EMA、验证或保存 Hook。

示例：
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python tools/debug_fp16_nan.py \
      configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_inin.yaml \
      --gpu 0 --batch-size 1 --workers 2 --max-iters 100 \
      --output-dir /tmp/daocc_fp16_nan
"""

import argparse
import json
import math
import random
import traceback
from collections.abc import Mapping
from pathlib import Path
from types import MethodType

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
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--checkpoint", help="可选 checkpoint；不恢复优化器状态")
    parser.add_argument("--seed", type=int, help="覆盖配置随机种子")
    parser.add_argument("--initial-scale", type=float, help="覆盖初始 loss scale")
    parser.add_argument(
        "--center-head-fp32",
        action="store_true",
        help="仅在本诊断脚本中关闭 object CenterHead 前向的 autocast，并强制使用 FP32",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-batch", action="store_true", help="异常时保存实际 batch")
    parser.add_argument(
        "--detect-anomaly",
        action="store_true",
        help="启用 autograd anomaly；很慢，但可能给出产生 NaN 梯度的算子栈",
    )
    parser.add_argument(
        "--stop-on-gradient-overflow",
        action="store_true",
        help="第一次梯度 Inf/NaN 后停止；默认让 GradScaler 降 scale 后继续观察",
    )
    args, overrides = parser.parse_known_args()
    if args.batch_size <= 0 or args.workers < 0 or args.max_iters <= 0:
        parser.error("batch-size/max-iters 必须为正数，workers 不能为负数")
    return args, overrides


def load_config(path, overrides):
    raw = TorchpackConfig()
    raw.load(path, recursive=True)
    raw.update(overrides)
    return Config(recursive_eval(raw), filename=path)


def make_scaler(fp16_cfg, initial_scale=None):
    loss_scale = fp16_cfg.get("loss_scale", 512.0)
    if loss_scale == "dynamic":
        kwargs = {}
    elif isinstance(loss_scale, Mapping):
        kwargs = dict(loss_scale)
    elif isinstance(loss_scale, (float, int)):
        kwargs = {"init_scale": float(loss_scale)}
    else:
        raise TypeError("fp16.loss_scale 必须是数值、'dynamic' 或字典")
    if initial_scale is not None:
        kwargs["init_scale"] = initial_scale
    return GradScaler(**kwargs)


def cast_floating_to_fp32(value):
    """递归将 CenterHead 输入中的浮点 Tensor 转为 FP32。"""
    if torch.is_tensor(value):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, Mapping):
        return type(value)((key, cast_floating_to_fp32(item)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(cast_floating_to_fp32(item) for item in value)
    if isinstance(value, list):
        return [cast_floating_to_fp32(item) for item in value]
    return value


def force_object_center_head_fp32(model):
    """只在当前进程临时将 object CenterHead 前向切换到 FP32。

    CenterHead.loss 已带 ``force_fp32(preds_dicts)``；这里进一步让 shared_conv
    和六个 task head 的预测卷积也退出 autocast。该包装不写回模型源码。
    """
    if "object" not in model.heads:
        raise KeyError("模型没有名为 object 的检测 head")
    head = model.heads["object"]
    original_forward = head.forward

    def fp32_forward(_self, feats, metas):
        with torch.cuda.amp.autocast(enabled=False):
            return original_forward(cast_floating_to_fp32(feats), metas)

    head.forward = MethodType(fp32_forward, head)
    return head


def iter_tensors(value, prefix="value", seen=None):
    """递归遍历普通容器、DataContainer 和稀疏张量中的 Tensor。"""
    if seen is None:
        seen = set()
    obj_id = id(value)
    if obj_id in seen:
        return
    seen.add(obj_id)

    if torch.is_tensor(value):
        yield prefix, value
        return
    features = getattr(value, "features", None)
    if torch.is_tensor(features):
        yield prefix + ".features", features
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from iter_tensors(item, f"{prefix}.{key}", seen)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from iter_tensors(item, f"{prefix}[{index}]", seen)
        return
    # MMCV DataContainer：其 data 可能是 Tensor，也可能是按 GPU 分组的 list。
    if value.__class__.__name__ == "DataContainer":
        yield from iter_tensors(value.data, prefix + ".data", seen)


def tensor_issue(name, tensor):
    if not (tensor.is_floating_point() or tensor.is_complex()):
        return None
    finite = torch.isfinite(tensor)
    if bool(finite.all()):
        return None
    total = tensor.numel()
    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())
    finite_values = tensor.detach()[finite]
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": total,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "finite_min": float(finite_values.min().item()) if finite_values.numel() else None,
        "finite_max": float(finite_values.max().item()) if finite_values.numel() else None,
    }


def find_issues(value, prefix="value", limit=20):
    issues = []
    for name, tensor in iter_tensors(value, prefix):
        issue = tensor_issue(name, tensor)
        if issue is not None:
            issues.append(issue)
            if len(issues) >= limit:
                break
    return issues


class FirstNonFiniteModule:
    """记录当前 iteration 第一个输出 NaN/Inf 的叶子模块。"""

    def __init__(self):
        self.first = None
        self.handles = []

    def reset(self):
        self.first = None

    def install(self, model):
        for name, module in model.named_modules():
            if not name or any(module.children()):
                continue

            def hook(_module, _inputs, output, module_name=name):
                if self.first is not None:
                    return
                issues = find_issues(output, "output", limit=1)
                if issues:
                    self.first = {"module": module_name, "issue": issues[0]}

            self.handles.append(module.register_forward_hook(hook))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def gradient_issues(model, limit=20):
    issues = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        issue = tensor_issue(name + ".grad", parameter.grad)
        if issue:
            issues.append(issue)
            if len(issues) >= limit:
                break
    return issues


def parameter_issues(model, limit=20):
    issues = []
    for name, parameter in model.named_parameters():
        issue = tensor_issue(name, parameter.data)
        if issue:
            issues.append(issue)
            if len(issues) >= limit:
                break
    return issues


def json_value(value):
    if torch.is_tensor(value):
        return float(value.detach().float().mean().item())
    if isinstance(value, (float, int)):
        return float(value)
    return str(value)


def save_failure(args, iteration, report, batch=None):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"iteration_{iteration:05d}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if args.save_batch and batch is not None:
        torch.save(batch, args.output_dir / f"iteration_{iteration:05d}_batch.pth")
    print(f"诊断已保存：{path}")


def main():
    args, overrides = parse_args()
    cfg = load_config(args.config, overrides)
    if cfg.get("fp16", None) is None:
        raise RuntimeError("当前配置 fp16=None；本脚本用于复现混合精度 NaN")
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU")

    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", False))
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print("构建训练数据集……")
    dataset = build_dataset(cfg.data.train)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=args.workers,
        num_gpus=1,
        dist=False,
        shuffle=True,
        seed=seed,
    )

    print("构建模型、优化器和 GradScaler……")
    model = build_model(cfg.model)
    model.init_weights()
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, map_location="cpu")
    if cfg.get("sync_bn", None):
        sync_cfg = cfg.sync_bn if isinstance(cfg.sync_bn, dict) else {"exclude": []}
        model = convert_sync_batchnorm(model, exclude=sync_cfg.get("exclude", []))

    optimizer = build_optimizer(model, cfg.optimizer)
    wrap_fp16_model(model)
    if args.center_head_fp32:
        head = force_object_center_head_fp32(model)
        print(
            "仅本脚本启用 CenterHead FP32：%s（shared_conv 和所有 task head）"
            % head.__class__.__name__
        )
    model = MMDataParallel(model.cuda(), device_ids=[args.gpu])
    scaler = make_scaler(cfg.fp16, args.initial_scale)
    grad_clip = cfg.get("optimizer_config", {}).get("grad_clip", None)
    module_probe = FirstNonFiniteModule()
    module_probe.install(model.module)
    model.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_info = {
        "config": str(Path(args.config).resolve()),
        "seed": seed,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "max_iters": args.max_iters,
        "initial_scale": float(scaler.get_scale()),
        "center_head_fp32": args.center_head_fp32,
        "detect_anomaly": args.detect_anomaly,
    }
    (args.output_dir / "run_info.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2)
    )
    print("初始 loss scale：%.1f" % scaler.get_scale())

    iterator = iter(loader)
    overflow_count = 0
    try:
        for iteration in range(1, args.max_iters + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)

            report = {"iteration": iteration, "scale_before": float(scaler.get_scale())}
            batch_issues = find_issues(batch, "batch")
            if batch_issues:
                report.update(kind="nonfinite_batch", issues=batch_issues)
                save_failure(args, iteration, report, batch)
                print("输入 batch 已含 NaN/Inf，停止。")
                break

            module_probe.reset()
            optimizer.zero_grad()
            try:
                outputs = model.train_step(batch, optimizer)
            except Exception as error:
                report.update(
                    kind="forward_exception",
                    error=repr(error),
                    traceback=traceback.format_exc(),
                    first_nonfinite_module=module_probe.first,
                )
                save_failure(args, iteration, report, batch)
                raise

            loss = outputs["loss"]
            report["loss"] = float(loss.detach().float().item())
            report["log_vars"] = {
                key: json_value(value) for key, value in outputs.get("log_vars", {}).items()
            }
            loss_issues = find_issues(loss, "loss")
            log_issues = [
                {"name": key, "value": json_value(value)}
                for key, value in outputs.get("log_vars", {}).items()
                if isinstance(value, (float, int)) and not math.isfinite(float(value))
            ]
            if loss_issues or log_issues:
                report.update(
                    kind="nonfinite_forward_loss",
                    loss_issues=loss_issues,
                    log_issues=log_issues,
                    first_nonfinite_module=module_probe.first,
                )
                save_failure(args, iteration, report, batch)
                print("前向或 loss 已产生 NaN/Inf，未执行反向，停止。")
                break

            try:
                context = torch.autograd.detect_anomaly(check_nan=True) if args.detect_anomaly else nullcontext()
                with context:
                    scaler.scale(loss).backward()
            except Exception as error:
                report.update(
                    kind="backward_exception",
                    error=repr(error),
                    traceback=traceback.format_exc(),
                    first_nonfinite_module=module_probe.first,
                )
                save_failure(args, iteration, report, batch)
                raise

            scaled_grad_issues = gradient_issues(model.module)
            scaler.unscale_(optimizer)
            unscaled_grad_issues = gradient_issues(model.module)
            if grad_clip is not None:
                grad_norm = clip_grad_norm_(model.parameters(), **dict(grad_clip))
                report["grad_norm"] = float(grad_norm.detach().float().item())

            scaler.step(optimizer)
            scaler.update()
            report["scale_after"] = float(scaler.get_scale())
            scale_reduced = report["scale_after"] < report["scale_before"]
            parameter_bad = parameter_issues(model.module)

            if scaled_grad_issues or unscaled_grad_issues or scale_reduced or parameter_bad:
                overflow_count += 1
                report.update(
                    kind="gradient_overflow" if not parameter_bad else "nonfinite_parameter",
                    scaled_gradient_issues=scaled_grad_issues,
                    unscaled_gradient_issues=unscaled_grad_issues,
                    parameter_issues=parameter_bad,
                    first_nonfinite_module=module_probe.first,
                )
                save_failure(args, iteration, report, batch)
                print(
                    "[%d/%d] loss=%.6g | scale %.1f -> %.1f | 梯度异常=%s | 参数异常=%s"
                    % (
                        iteration,
                        args.max_iters,
                        report["loss"],
                        report["scale_before"],
                        report["scale_after"],
                        bool(scaled_grad_issues or unscaled_grad_issues),
                        bool(parameter_bad),
                    )
                )
                if parameter_bad or args.stop_on_gradient_overflow:
                    break
            else:
                print(
                    "[%d/%d] loss=%.6g | scale=%.1f | gradients/parameters finite"
                    % (iteration, args.max_iters, report["loss"], report["scale_after"])
                )
        else:
            print("完成 %d iterations，未发现前向 NaN/Inf。" % args.max_iters)
    finally:
        module_probe.close()

    print("检测到的梯度溢出次数：%d" % overflow_count)
    print("诊断目录：%s" % args.output_dir.resolve())


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc_info):
        return False


if __name__ == "__main__":
    main()
