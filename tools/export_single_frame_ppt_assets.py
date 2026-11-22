#!/usr/bin/env python3
"""Export presentation-ready assets for one DAOcc validation frame.

The script runs DAOcc once and writes four groups of independent images:

* the six original, unaugmented nuScenes camera images;
* a bird's-eye-view image of the LiDAR input used by the model;
* one oblique occupancy image with high-contrast 3D detection boxes;
* six occupancy renderings viewed from the calibrated camera poses.

Mayavi/VTK in the ``daocc-cu121`` environment needs an X server.  Run this
script through the same ``xvfb-run`` command used by
``visualize_occ_detection_video.py``.
"""

import argparse
import copy
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model
from torchpack.utils.config import configs

# Importing this module first also applies the tested Qt/xcb runtime repair.
from visualize_occ_detection_video import (
    BOX_EDGES,
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    OCC_COLORS,
    VOXEL_SIZE,
    configure_qt_for_mayavi,
    draw_line_set,
    occupied_voxels,
    prepare_dataset_cfg,
    set_camera,
    to_numpy,
    transform_box_corners,
)

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one DAOcc frame and export independent PPT assets"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--index", type=int, default=0,
        help="zero-based frame index in the selected split (default: 0)",
    )
    selection.add_argument(
        "--token", default=None, help="exact nuScenes sample token",
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--out-dir", default="viz/single_frame_ppt_assets")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bbox-score", type=float, default=0.30)
    parser.add_argument(
        "--bbox-line-width", type=float, default=2.0,
        help="foreground width of detection box lines in screen pixels",
    )
    parser.add_argument("--overview-width", type=int, default=1400)
    parser.add_argument("--overview-height", type=int, default=900)
    parser.add_argument("--elevation", type=float, default=55.0)
    parser.add_argument("--camera-distance", type=float, default=82.0)
    parser.add_argument("--focal-x", type=float, default=8.0)
    parser.add_argument(
        "--no-camera-mask", action="store_true",
        help="render occupied voxels outside mask_camera too",
    )
    parser.add_argument(
        "--render-interior", action="store_true",
        help="render enclosed same-class voxels as well (slower)",
    )
    args, cfg_options = parser.parse_known_args()
    if args.index is not None and args.index < 0:
        parser.error("--index must be >= 0")
    if args.workers < 0:
        parser.error("--workers must be >= 0")
    if not 0 <= args.bbox_score <= 1:
        parser.error("--bbox-score must be in [0, 1]")
    if args.bbox_line_width <= 0:
        parser.error("--bbox-line-width must be positive")
    for value in (
        args.overview_width, args.overview_height,
    ):
        if value <= 0:
            parser.error("output image dimensions must be positive")
    return args, cfg_options


def choose_index(dataset, index, token):
    if token is None:
        if index >= len(dataset):
            raise IndexError(f"index {index} is outside dataset length {len(dataset)}")
        return index
    matches = [i for i, info in enumerate(dataset.data_infos)
               if str(info.get("token")) == token]
    if not matches:
        raise ValueError(f"sample token not found in {len(dataset)} {dataset.__class__.__name__} frames: {token}")
    return matches[0]


def resolve_data_path(path, dataset):
    path = Path(path).expanduser()
    if path.is_file():
        return path.resolve()
    roots = [
        getattr(dataset, "dataset_root", None),
        getattr(dataset, "data_root", None),
    ]
    for root in roots:
        if root:
            candidate = Path(root).expanduser() / path
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(path)


def export_original_cameras(info, dataset, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    cameras = info.get("cams", {})
    if len(cameras) != 6:
        raise ValueError(f"expected six cameras, found {list(cameras)}")
    names = [name for name in CAMERA_ORDER if name in cameras]
    names.extend(name for name in cameras if name not in names)
    outputs = []
    for number, name in enumerate(names, 1):
        source = resolve_data_path(cameras[name]["data_path"], dataset)
        suffix = source.suffix.lower() if source.suffix else ".jpg"
        target = output_dir / f"{number:02d}_{name}{suffix}"
        shutil.copy2(str(source), str(target))
        outputs.append(target)
    return outputs


def original_camera_sizes(info, dataset):
    """Return source (width, height) in the same order as model cameras."""
    sizes = []
    for name, camera in info["cams"].items():
        source = resolve_data_path(camera["data_path"], dataset)
        image = mmcv.imread(str(source), flag="unchanged")
        if image is None or image.ndim < 2:
            raise RuntimeError(f"failed to read camera image dimensions: {source}")
        height, width = image.shape[:2]
        sizes.append((int(width), int(height)))
    return sizes


def unwrap(value):
    while hasattr(value, "data") and not torch.is_tensor(value):
        value = value.data
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def extract_points(data):
    points = unwrap(data["points"])
    if hasattr(points, "tensor"):
        points = points.tensor
    points = to_numpy(points).squeeze()
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"expected point array (N, >=3), got {points.shape}")
    return points


def export_lidar_bev(points, output_path, point_range):
    """Draw the actual multi-sweep LiDAR tensor passed to the network."""
    x, y = points[:, 0], points[:, 1]
    valid = np.isfinite(x) & np.isfinite(y)
    valid &= (x >= point_range[0]) & (x <= point_range[3])
    valid &= (y >= point_range[1]) & (y <= point_range[4])
    x, y = x[valid], y[valid]
    if points.shape[1] > 3:
        intensity = points[valid, 3].astype(np.float32)
        low, high = np.percentile(intensity, [2, 98]) if len(intensity) else (0, 1)
        intensity = np.clip((intensity - low) / max(high - low, 1e-6), 0, 1)
        colors = plt.get_cmap("viridis")(0.12 + 0.78 * intensity)
    else:
        colors = "#16324f"

    figure, axis = plt.subplots(figsize=(8.0, 8.0), dpi=220)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    # Ego-forward (+x) points upward, matching the occupancy overview.
    axis.scatter(-y, x, s=0.20, c=colors, alpha=0.82, linewidths=0,
                 rasterized=True)
    axis.set_xlim(-point_range[4], -point_range[1])
    axis.set_ylim(point_range[0], point_range[3])
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")
    figure.savefig(str(output_path), bbox_inches="tight", pad_inches=0.03,
                   facecolor="white")
    plt.close(figure)


def make_occupancy_glyph(mlab, figure, occupancy, camera_mask, surface_only):
    xyz, labels = occupied_voxels(
        occupancy, camera_mask, surface_only=surface_only
    )
    if not len(xyz):
        raise RuntimeError("no occupied voxels remain after visibility filtering")
    glyph = mlab.points3d(
        xyz[:, 0], xyz[:, 1], xyz[:, 2], labels,
        mode="cube", scale_factor=VOXEL_SIZE * 0.94, opacity=1.0,
        vmin=0, vmax=17, figure=figure,
    )
    glyph.glyph.scale_mode = "data_scaling_off"
    glyph.module_manager.scalar_lut_manager.lut.table = OCC_COLORS
    return glyph


def filtered_box_corners(output, data, threshold):
    boxes = output.get("boxes_3d")
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 8, 3), dtype=np.float32)
    scores = to_numpy(output.get("scores_3d", np.empty(0))).reshape(-1)
    if len(scores) != len(boxes):
        raise ValueError(f"box/score count mismatch: {len(boxes)} vs {len(scores)}")
    keep = torch.from_numpy(scores >= threshold)
    return transform_box_corners(
        boxes[keep], data["lidar_aug_matrix"], data["lidar2ego"],
        data["occ_aug_matrix"],
    )


def add_detection_boxes(mlab, figure, corners, line_width):
    if not len(corners):
        return []
    points = corners.reshape(-1, 3)
    lines = np.asarray([
        (box_id * 8 + start, box_id * 8 + end)
        for box_id in range(len(corners))
        for start, end in BOX_EDGES
    ], dtype=np.int64)
    # A subtle black underlay separates boxes from similarly coloured voxels
    # without making their screen-space footprint visually heavy.
    underlay = draw_line_set(
        mlab, figure, points, lines, color=(0.02, 0.02, 0.02),
        line_width=line_width + 1.2,
    )
    foreground = draw_line_set(
        mlab, figure, points, lines, color=(1.0, 0.78, 0.0),
        line_width=line_width,
    )
    return [underlay, foreground]


def render_detection_overview(mlab, output_path, output, data, camera_mask,
                              args):
    figure = mlab.figure(
        size=(args.overview_width, args.overview_height),
        bgcolor=(1.0, 1.0, 1.0), fgcolor=(0.1, 0.1, 0.1),
    )
    try:
        figure.scene.disable_render = True
        make_occupancy_glyph(
            mlab, figure, output["occ_pred"], camera_mask,
            surface_only=not args.render_interior,
        )
        corners = filtered_box_corners(output, data, args.bbox_score)
        add_detection_boxes(mlab, figure, corners, args.bbox_line_width)
        figure.scene.disable_render = False
        set_camera(
            figure, args.elevation, args.camera_distance, args.focal_x
        )
        mlab.savefig(str(output_path), figure=figure,
                     size=(args.overview_width, args.overview_height))
        return len(corners)
    finally:
        mlab.close(figure)


def calibrated_camera_pose(camera2ego, occ_aug_matrix):
    transform = np.asarray(camera2ego, dtype=np.float64)
    occ_aug = np.asarray(occ_aug_matrix, dtype=np.float64)
    rotation = occ_aug[:3, :3] @ transform[:3, :3]
    position = occ_aug[:3, :3] @ transform[:3, 3]
    # nuScenes camera coordinates: +z optical axis, +y points down.
    focal = position + rotation @ np.array([0.0, 0.0, 20.0])
    view_up = rotation @ np.array([0.0, -1.0, 0.0])
    return position, focal, view_up


def set_calibrated_camera(figure, camera2ego, intrinsic, occ_aug_matrix,
                          image_width, image_height):
    position, focal, view_up = calibrated_camera_pose(
        camera2ego, occ_aug_matrix
    )
    k = np.asarray(intrinsic, dtype=np.float64)[:3, :3]
    # Render at the original image resolution and derive the field of view
    # from that exact sensor size rather than assuming the principal point is
    # precisely at the image centre.
    source_width = float(image_width)
    source_height = float(image_height)
    vertical_fov = math.degrees(2.0 * math.atan(source_height / (2.0 * k[1, 1])))
    camera = figure.scene.camera
    camera.position = tuple(position)
    camera.focal_point = tuple(focal)
    camera.view_up = tuple(view_up)
    camera.parallel_projection = False
    camera.view_angle = float(vertical_fov)
    camera.clipping_range = (0.08, 160.0)
    # Keep the calibrated principal point when it is slightly off-centre.
    camera.window_center = (
        float(2.0 * (k[0, 2] / source_width - 0.5)),
        float(2.0 * (k[1, 2] / source_height - 0.5)),
    )
    figure.scene.render()


def render_camera_occupancies(mlab, output_dir, output, data, camera_mask,
                              args, camera_names, camera_sizes):
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(camera_sizes) != 6:
        raise ValueError(f"expected six camera image sizes, got {camera_sizes}")
    figure = mlab.figure(
        size=camera_sizes[0],
        bgcolor=(1.0, 1.0, 1.0), fgcolor=(0.1, 0.1, 0.1),
    )
    outputs = []
    try:
        figure.scene.disable_render = True
        make_occupancy_glyph(
            mlab, figure, output["occ_pred"], camera_mask,
            surface_only=not args.render_interior,
        )
        figure.scene.disable_render = False
        transforms = to_numpy(data["camera2ego"]).squeeze()
        intrinsics = to_numpy(data["camera_intrinsics"]).squeeze()
        occ_aug = to_numpy(data["occ_aug_matrix"]).squeeze()
        if transforms.shape != (6, 4, 4) or intrinsics.shape != (6, 4, 4):
            raise ValueError(
                f"expected six calibrated cameras, got {transforms.shape} and {intrinsics.shape}"
            )
        for number, (name, transform, intrinsic, image_size) in enumerate(
                zip(camera_names, transforms, intrinsics, camera_sizes), 1):
            image_width, image_height = image_size
            figure.scene.set_size((image_width, image_height))
            set_calibrated_camera(
                figure, transform, intrinsic, occ_aug,
                image_width, image_height,
            )
            target = output_dir / f"{number:02d}_{name}_occupancy.png"
            mlab.savefig(str(target), figure=figure,
                         size=(image_width, image_height))
            outputs.append(target)
    finally:
        mlab.close(figure)
    return outputs


def write_manifest(output_dir, sample_index, token, camera_files, lidar_file,
                   detection_file, perspective_files, box_count, args):
    lines = [
        f"dataset_index: {sample_index}",
        f"sample_token: {token}",
        f"visibility_mask: {'none' if args.no_camera_mask else 'mask_camera'}",
        f"detection_score_threshold: {args.bbox_score}",
        f"displayed_detection_boxes: {box_count}",
        "",
        "original_camera_images:",
    ]
    lines.extend(f"  - {path.relative_to(output_dir)}" for path in camera_files)
    lines.extend([
        f"lidar_bev: {lidar_file.relative_to(output_dir)}",
        f"occupancy_with_3d_detection: {detection_file.relative_to(output_dir)}",
        "camera_view_occupancy:",
    ])
    lines.extend(f"  - {path.relative_to(output_dir)}" for path in perspective_files)
    (output_dir / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    sample_index = choose_index(dataset, args.index, args.token)
    info = dataset.data_infos[sample_index]
    token = str(info["token"])

    output_dir = Path(args.out_dir).expanduser().resolve() / f"{sample_index:04d}_{token}"
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_files = export_original_cameras(
        info, dataset, output_dir / "01_camera_inputs"
    )

    subset = torch.utils.data.Subset(dataset, [sample_index])
    loader = build_dataloader(
        subset, samples_per_gpu=1, workers_per_gpu=args.workers,
        dist=False, shuffle=False,
    )
    data = next(iter(loader))

    lidar_file = output_dir / "02_lidar_input_bev.png"
    point_range = cfg.get("point_cloud_range", [-54, -54, -5, 54, 54, 3])
    export_lidar_bev(extract_points(data), lidar_file, point_range)

    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = cfg.get("cudnn_benchmark", False)
    model = build_model(cfg.model)
    if cfg.get("fp16") is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model.cuda(args.gpu), device_ids=[args.gpu])
    model.eval()
    print(f"Running inference for index={sample_index}, token={token} ...", flush=True)
    with torch.inference_mode():
        output = model(**data)[0]
    if "occ_pred" not in output:
        raise KeyError("model output does not contain occ_pred")

    configure_qt_for_mayavi()
    from mayavi import mlab

    mlab.options.offscreen = True
    camera_mask = None if args.no_camera_mask else data.get("mask_camera")
    detection_file = output_dir / "03_occupancy_with_3d_detection.png"
    box_count = render_detection_overview(
        mlab, detection_file, output, data, camera_mask, args
    )

    # Dataset insertion order is also the order used by camera2ego and the
    # image tensor.  Keep the same names instead of silently reordering poses.
    camera_names = list(info["cams"].keys())
    camera_sizes = original_camera_sizes(info, dataset)
    perspective_files = render_camera_occupancies(
        mlab, output_dir / "04_camera_view_occupancy", output, data,
        camera_mask, args, camera_names, camera_sizes,
    )
    write_manifest(
        output_dir, sample_index, token, camera_files, lidar_file,
        detection_file, perspective_files, box_count, args,
    )
    print(f"Saved PPT assets: {output_dir}")
    print(f"  six original cameras : {output_dir / '01_camera_inputs'}")
    print(f"  LiDAR input          : {lidar_file}")
    print(f"  occupancy + boxes    : {detection_file} ({box_count} boxes)")
    print(f"  six camera views     : {output_dir / '04_camera_view_occupancy'}")


if __name__ == "__main__":
    main()
