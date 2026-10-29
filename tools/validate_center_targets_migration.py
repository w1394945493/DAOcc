#!/usr/bin/env python
"""验证 CenterHead 检测 GT 从 loss 迁移到数据 pipeline 后是否保持一致。"""

import argparse
import copy
import random

import numpy as np
import torch
from mmcv import Config
from torchpack.utils.config import Config as TorchpackConfig

from mmdet3d.datasets import build_dataset
from mmdet3d.models.builder import build_head
from mmdet3d.utils import recursive_eval


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="启用了 GenerateCenterHeadTargets 的 YAML 配置")
    parser.add_argument("--num-samples", type=int, default=20, help="验证的真实样本数")
    parser.add_argument("--batch-size", type=int, default=2, help="模拟 collate 后的 batch 大小")
    parser.add_argument("--seed", type=int, default=0, help="抽样及模拟预测的随机种子")
    args, overrides = parser.parse_known_args()
    if args.num_samples <= 0:
        parser.error("--num-samples 必须为正整数")
    if args.batch_size <= 0:
        parser.error("--batch-size 必须为正整数")
    return args, overrides


def load_config(path, overrides):
    # (10.29 检测GT迁移) 按 tools/train.py 相同方式递归加载配置，确保 pipeline
    # (10.29 检测GT迁移) 中引用的 model.heads.object 参数已完成插值解析。
    raw = TorchpackConfig()
    raw.load(path, recursive=True)
    raw.update(overrides)
    return Config(recursive_eval(raw), filename=path)


def unwrap_sample(sample):
    # (10.29 检测GT迁移) dataset[i] 已经过 DefaultFormatBundle3D；框和标签
    # (10.29 检测GT迁移) 位于 DataContainer.data，object_targets 保持嵌套张量字典。
    return (
        sample["gt_bboxes_3d"].data,
        sample["gt_labels_3d"].data,
        sample["object_targets"],
    )


def stack_migrated_targets(targets):
    # (10.29 检测GT迁移) 模拟 MMCV collate：同一个 task 的单样本张量沿第 0 维
    # (10.29 检测GT迁移) 堆叠，从而检查真实训练时使用的 [B, ...] 结构。
    return {
        key: [
            torch.stack([sample[key][task_id] for sample in targets], dim=0)
            for task_id in range(len(targets[0][key]))
        ]
        for key in ("heatmaps", "anno_boxes", "inds", "masks")
    }


def old_targets_to_dict(old_targets):
    return dict(
        zip(("heatmaps", "anno_boxes", "inds", "masks"), old_targets)
    )


def compare_targets(old_targets, migrated_targets, batch_number):
    # (10.29 检测GT迁移) 使用 torch.equal 做严格逐元素比较，不使用容差；这些目标
    # (10.29 检测GT迁移) 理应由相同 GT 产生，任何坐标、类别或展平顺序差异都应报错。
    for key in ("heatmaps", "anno_boxes", "inds", "masks"):
        old_tasks = old_targets[key]
        new_tasks = migrated_targets[key]
        if len(old_tasks) != len(new_tasks):
            raise AssertionError(
                f"batch {batch_number}: {key} task 数不同："
                f"旧={len(old_tasks)}，新={len(new_tasks)}"
            )
        for task_id, (old_tensor, new_tensor) in enumerate(zip(old_tasks, new_tasks)):
            if old_tensor.shape != new_tensor.shape:
                raise AssertionError(
                    f"batch {batch_number} task {task_id} {key} 形状不同："
                    f"旧={tuple(old_tensor.shape)}，新={tuple(new_tensor.shape)}"
                )
            if not torch.equal(old_tensor, new_tensor):
                difference = (old_tensor.float() - new_tensor.float()).abs()
                raise AssertionError(
                    f"batch {batch_number} task {task_id} {key} 数值不同："
                    f"不同元素数={(difference != 0).sum().item()}，"
                    f"最大绝对误差={difference.max().item()}"
                )


def make_predictions(targets, generator):
    # (10.29 检测GT迁移) 构造与 CenterHead 六个 task 输出同形状的模拟预测，
    # (10.29 检测GT迁移) 用同一随机种子分别计算迁移前后 loss 和预测梯度。
    predictions = []
    for heatmap in targets["heatmaps"]:
        batch_size, num_classes, height, width = heatmap.shape
        task_prediction = {
            "heatmap": torch.randn(
                batch_size, num_classes, height, width, generator=generator
            ),
            "reg": torch.randn(batch_size, 2, height, width, generator=generator),
            "height": torch.randn(batch_size, 1, height, width, generator=generator),
            "dim": torch.randn(batch_size, 3, height, width, generator=generator),
            "rot": torch.randn(batch_size, 2, height, width, generator=generator),
            "vel": torch.randn(batch_size, 2, height, width, generator=generator),
        }
        predictions.append([task_prediction])
    return predictions


def clone_predictions(predictions):
    cloned = copy.deepcopy(predictions)
    retained = []
    for task in cloned:
        task_retained = {}
        for name, tensor in task[0].items():
            # (10.29 检测GT迁移) 真实预测是卷积产生的非叶子张量，CenterHead 会对
            # (10.29 检测GT迁移) heatmap 执行 sigmoid_()；加 0 构造同类非叶子张量，
            # (10.29 检测GT迁移) retain_grad() 用于稍后直接核对每个预测分支的梯度。
            tensor.requires_grad_(True)
            tensor = tensor + 0
            tensor.retain_grad()
            task[0][name] = tensor
            task_retained[name] = tensor
        retained.append(task_retained)
    return cloned, retained


def compare_losses_and_gradients(head, boxes, labels, old_targets, new_targets, generator):
    base_predictions = make_predictions(old_targets, generator)
    old_predictions, old_retained = clone_predictions(base_predictions)
    new_predictions, new_retained = clone_predictions(base_predictions)

    old_losses = head.loss(
        boxes, labels, old_predictions, object_targets=old_targets
    )
    new_losses = head.loss(
        boxes, labels, new_predictions, object_targets=new_targets
    )

    # (10.29 检测GT迁移) loss 必须逐项严格一致；随后对总 loss 反向传播，确认迁移
    # (10.29 检测GT迁移) 没有改变 heatmap 与五个框分支收到的训练梯度。
    if old_losses.keys() != new_losses.keys():
        raise AssertionError("迁移前后 loss 字段不同")
    for name in old_losses:
        if not torch.equal(old_losses[name], new_losses[name]):
            raise AssertionError(
                f"{name} 不一致：旧={old_losses[name].item()}，"
                f"新={new_losses[name].item()}"
            )

    sum(old_losses.values()).backward()
    sum(new_losses.values()).backward()
    for task_id, (old_task, new_task) in enumerate(zip(old_retained, new_retained)):
        for name in old_task:
            old_grad = old_task[name].grad
            new_grad = new_task[name].grad
            if old_grad is None or new_grad is None:
                raise AssertionError(f"task {task_id} {name} 没有梯度")
            if not torch.equal(old_grad, new_grad):
                difference = (old_grad - new_grad).abs()
                raise AssertionError(
                    f"task {task_id} {name} 梯度不同，"
                    f"最大绝对误差={difference.max().item()}"
                )
    return old_losses


def main():
    args, overrides = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_config(args.config, overrides)
    dataset = build_dataset(cfg.data.train)
    head = build_head(cfg.model.heads.object)
    generator = torch.Generator().manual_seed(args.seed)

    # (10.29 检测GT迁移) 均匀抽取数据集不同位置，避免只验证连续、相似场景；
    # (10.29 检测GT迁移) 每次 dataset[index] 仍会执行配置中的真实随机数据增强。
    indices = np.linspace(
        0, len(dataset) - 1, num=min(args.num_samples, len(dataset)), dtype=np.int64
    ).tolist()
    verified_samples = 0
    verified_batches = 0
    last_losses = None

    for start in range(0, len(indices), args.batch_size):
        batch_indices = indices[start : start + args.batch_size]
        samples = [unwrap_sample(dataset[index]) for index in batch_indices]
        boxes = [sample[0] for sample in samples]
        labels = [sample[1] for sample in samples]
        migrated = stack_migrated_targets([sample[2] for sample in samples])

        # (10.29 检测GT迁移) 这里直接调用源代码中保留的旧 get_targets()，它是
        # (10.29 检测GT迁移) 独立基准，不依赖新增 pipeline 的 object_targets。
        old = old_targets_to_dict(head.get_targets(boxes, labels))
        compare_targets(old, migrated, verified_batches + 1)
        last_losses = compare_losses_and_gradients(
            head, boxes, labels, old, migrated, generator
        )

        verified_samples += len(samples)
        verified_batches += 1
        print(
            f"[通过 {verified_batches}] 样本索引={batch_indices}："
            "targets、loss、预测梯度全部一致"
        )

    print("\nCenterHead 检测 GT 迁移验证通过")
    print(f"真实样本数：{verified_samples}")
    print(f"模拟 batch 数：{verified_batches}")
    print("比较内容：heatmaps、anno_boxes、inds、masks")
    print(f"loss 项数：{len(last_losses) if last_losses is not None else 0}")
    print("结论：迁移未改变 CenterHead 的监督目标、loss 数值和预测梯度。")


if __name__ == "__main__":
    main()
