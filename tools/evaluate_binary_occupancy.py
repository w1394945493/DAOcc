#!/usr/bin/env python3
"""Evaluate binary occupied/free geometry metrics for DAOcc.

Semantic labels 0--16 are merged into ``occupied``; label 17 is ``free``;
label 255 is ignored.  Metrics are accumulated globally over voxels rather
than averaged per frame:

    IoU       = TP / (TP + FP + FN)
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model
from torchpack.utils.config import configs
from tqdm import tqdm

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


DEFAULT_CONFIG = (
    "/home/wangyushen/forks/CUDA-DAOcc/DAOcc/configs/nuscenes/occ3d/"
    "daocc_occ3d_nus_w_mask_inin_spconv2_voxel_iter.yaml"
)
DEFAULT_CHECKPOINT = (
    "/home/wangyushen/Data/Outputs/daocc/train/daocc/w_mask/"
    "train_spconv2_voxel_iter_fp16_htc/latest.pth"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate binary occupancy IoU, precision and recall"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--mask", choices=("camera", "lidar", "none"), default="camera",
        help="evaluation visibility mask (default: camera)",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="maximum selected frames; 0 evaluates all frames",
    )
    parser.add_argument("--out", default="binary_occupancy_metrics.json")
    args, cfg_options = parser.parse_known_args()
    if args.gpu < 0 or args.workers < 0:
        parser.error("--gpu and --workers must be >= 0")
    if args.interval < 1 or args.max_samples < 0:
        parser.error("--interval must be >= 1 and --max-samples must be >= 0")
    return args, cfg_options


def prepare_dataset_cfg(dataset_cfg):
    """Convert sequence/wrapper options to deterministic frame inference."""
    if isinstance(dataset_cfg, list):
        for child in dataset_cfg:
            prepare_dataset_cfg(child)
        return
    dataset_cfg.pop("samples_per_gpu", None)
    if dataset_cfg.get("type") == "NuScenesSequenceDataset":
        dataset_cfg["type"] = "NuScenesDatasetOccupancy"
    for key in ("sequences_split_num", "keep_consistent_seq_aug"):
        dataset_cfg.pop(key, None)
    if "dataset" in dataset_cfg:
        prepare_dataset_cfg(dataset_cfg["dataset"])
    if "datasets" in dataset_cfg:
        prepare_dataset_cfg(dataset_cfg["datasets"])


def as_numpy(value):
    """Unwrap DataContainer/list/tensor values into a NumPy array."""
    while hasattr(value, "data") and not torch.is_tensor(value):
        value = value.data
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")


def main():
    args, cfg_options = parse_args()
    for path in (args.config, args.checkpoint):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    configs.load(args.config, recursive=True)
    configs.update(cfg_options)
    cfg = Config(recursive_eval(configs), filename=args.config)

    dataset_cfg = copy.deepcopy(cfg.data[args.split])
    prepare_dataset_cfg(dataset_cfg)
    dataset = build_dataset(dataset_cfg)
    indices = list(range(0, len(dataset), args.interval))
    if args.max_samples:
        indices = indices[:args.max_samples]
    if not indices:
        raise RuntimeError("No evaluation frames selected")
    loader = build_dataloader(
        torch.utils.data.Subset(dataset, indices),
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False,
    )

    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = cfg.get("cudnn_benchmark", False)
    model = build_model(cfg.model)
    if cfg.get("fp16") is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model.cuda(args.gpu), device_ids=[args.gpu])
    model.eval()

    tp = fp = fn = tn = ignored = 0
    mask_key = {"camera": "mask_camera", "lidar": "mask_lidar"}.get(args.mask)
    for data in tqdm(loader, total=len(indices), desc="Binary occupancy evaluation"):
        with torch.inference_mode():
            output = model(**data)[0]
        if "occ_pred" not in output:
            raise KeyError("Model output does not contain occ_pred")
        prediction = as_numpy(output["occ_pred"]).squeeze()
        ground_truth = as_numpy(data["voxel_semantics"]).squeeze()
        if prediction.shape != ground_truth.shape:
            raise ValueError(
                f"Prediction shape {prediction.shape} != GT shape {ground_truth.shape}"
            )

        valid = ground_truth != 255
        if mask_key is not None:
            visibility = as_numpy(data[mask_key]).squeeze().astype(bool)
            if visibility.shape != ground_truth.shape:
                raise ValueError(
                    f"{mask_key} shape {visibility.shape} != GT shape {ground_truth.shape}"
                )
            valid &= visibility
        ignored += int(valid.size - np.count_nonzero(valid))

        gt_occupied = (ground_truth >= 0) & (ground_truth < 17)
        pred_occupied = (prediction >= 0) & (prediction < 17)
        tp += int(np.count_nonzero(valid & gt_occupied & pred_occupied))
        fp += int(np.count_nonzero(valid & ~gt_occupied & pred_occupied))
        fn += int(np.count_nonzero(valid & gt_occupied & ~pred_occupied))
        tn += int(np.count_nonzero(valid & ~gt_occupied & ~pred_occupied))

    iou = safe_ratio(tp, tp + fp + fn)
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    accuracy = safe_ratio(tp + tn, tp + fp + fn + tn)
    result = {
        "definition": {
            "occupied_labels": "0-16",
            "free_label": 17,
            "ignore_label": 255,
            "mask": args.mask,
            "aggregation": "global voxel counts",
        },
        "dataset": {
            "split": args.split,
            "frames": len(indices),
            "interval": args.interval,
        },
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "ignored": ignored},
        "metrics": {
            "binary_occupancy_iou": iou,
            "occupancy_precision": precision,
            "occupancy_recall": recall,
            "binary_voxel_accuracy": accuracy,
        },
    }

    print("\nBinary occupancy geometry metrics")
    print(f"  Evaluated frames : {len(indices)}")
    print(f"  Visibility mask  : {args.mask}")
    print(f"  Occupancy IoU    : {iou * 100:.2f}%")
    print(f"  Precision        : {precision * 100:.2f}%")
    print(f"  Recall           : {recall * 100:.2f}%")
    print(f"  Voxel accuracy   : {accuracy * 100:.2f}% (reported for reference)")
    print(f"  TP/FP/FN/TN      : {tp}/{fp}/{fn}/{tn}")

    output_path = Path(args.out).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"  JSON             : {output_path}")


if __name__ == "__main__":
    main()
