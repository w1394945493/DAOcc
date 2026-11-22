#!/usr/bin/env python3
"""Render DAOcc occupancy and 3D detections as scene videos.

The camera is fixed behind and above the ego vehicle.  It looks down by about
55 degrees, with the ego forward (+X) direction pointing towards the top of
the image.  Occupancy is drawn in the Occ3D semantic palette and detections
are overlaid as yellow wireframes.
"""

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def ensure_xcb_runtime():
    """Restart once with Conda's xcb runtime visible to PyQt's xcb plugin."""
    if not os.environ.get("DISPLAY") or os.environ.get("DAOCC_XCB_RUNTIME_READY") == "1":
        return
    # The PyQt5 wheel in daocc-cu121 needs libxcb-xinerama.so.0.  It is
    # available in the local Conda package cache but is not linked into this
    # environment.  LD_LIBRARY_PATH must be present when the process starts,
    # hence the one-time exec before importing any third-party extension.
    conda_root = Path(sys.prefix).resolve().parents[1]
    candidates = sorted(
        (conda_root / "pkgs").glob("libxcb-*/lib/libxcb-xinerama.so.0"),
        reverse=True,
    )
    if not candidates:
        return
    library_dir = str(candidates[0].parent)
    paths = [item for item in os.environ.get("LD_LIBRARY_PATH", "").split(":") if item]
    if library_dir not in paths:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = ":".join([library_dir] + paths)
        env["DAOCC_XCB_RUNTIME_READY"] = "1"
        os.execvpe(sys.executable, [sys.executable] + sys.argv, env)


ensure_xcb_runtime()


def configure_qt_for_mayavi():
    """Restore PyQt's plugins after OpenCV changes Qt environment variables."""
    try:
        import PyQt5
    except ImportError:
        return
    plugins = Path(PyQt5.__file__).resolve().parent / "Qt5" / "plugins"
    platforms = plugins / "platforms"
    if not platforms.is_dir():
        return
    # Importing cv2 (indirectly through mmdet3d) rewrites these variables to
    # cv2/qt/plugins.  Mayavi then finds OpenCV's incomplete xcb installation.
    # Assign rather than setdefault, and call this again immediately before
    # importing Mayavi so the later cv2 import cannot win.
    os.environ["QT_PLUGIN_PATH"] = str(plugins)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms)
    # daocc-cu121's VTK is an X11 build.  When xvfb-run supplies DISPLAY,
    # Qt and VTK must both use X11/xcb.  Qt's offscreen plugin combined with
    # vtkXOpenGLRenderWindow can hang or abort during the first render.
    os.environ["QT_QPA_PLATFORM"] = "xcb" if os.environ.get("DISPLAY") else "offscreen"
    os.environ["ETS_TOOLKIT"] = "qt"


configure_qt_for_mayavi()

import mmcv
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

# Same semantic palette as tools/vis_occ3d.py.  IDs 0 and 17 are hidden.
OCC_COLORS = np.array(
    [
        [0, 0, 0, 255], [255, 120, 50, 255], [255, 192, 203, 255],
        [255, 255, 0, 255], [0, 150, 245, 255], [0, 255, 255, 255],
        [200, 180, 0, 255], [255, 0, 0, 255], [255, 240, 150, 255],
        [135, 60, 0, 255], [160, 32, 240, 255], [255, 0, 255, 255],
        [139, 137, 137, 255], [75, 0, 75, 255], [150, 240, 80, 255],
        [230, 230, 250, 255], [0, 175, 0, 255], [0, 0, 255, 255],
    ],
    dtype=np.uint8,
)
VOXEL_SIZE = 0.4
OCC_ORIGIN = np.array([-40.0, -40.0, -1.0], dtype=np.float32)
BOX_EDGES = (
    # LiDARInstance3DBoxes.corners order:
    # (x0y0z0, x0y0z1, x0y1z1, x0y1z0,
    #  x1y0z0, x1y0z1, x1y1z1, x1y1z0).
    (0, 1), (0, 3), (0, 4),
    (1, 2), (1, 5),
    (3, 2), (3, 7),
    (4, 5), (4, 7),
    (2, 6), (5, 6), (6, 7),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render fixed oblique-view occupancy + 3D detection videos"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", default="viz/occ_detection_video")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--scenes", nargs="+", default=None,
        help="scene names/tokens to render, e.g. scene-0103; default: evenly select scenes",
    )
    parser.add_argument(
        "--scene-index", type=int, default=None,
        help="zero-based index in the sorted validation scene list",
    )
    parser.add_argument("--num-scenes", type=int, default=3)
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="maximum frames per scene; 0 keeps every frame",
    )
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--bbox-score", type=float, default=0.35)
    parser.add_argument("--bbox-radius", type=float, default=0.075)
    parser.add_argument(
        "--frustum-depth", type=float, default=1.0,
        help="display depth of calibrated camera frustums in metres (default: 1.0)",
    )
    parser.add_argument(
        "--frustum-line-width", type=float, default=1.2,
        help="camera frustum line width in screen pixels (default: 1.2)",
    )
    parser.add_argument(
        "--no-camera-mask", action="store_true",
        help="render occupied voxels outside mask_camera as well",
    )
    parser.add_argument(
        "--render-interior", action="store_true",
        help="also render fully enclosed same-class voxels (slower and normally invisible)",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--elevation", type=float, default=55.0)
    parser.add_argument("--camera-distance", type=float, default=82.0)
    parser.add_argument("--focal-x", type=float, default=8.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--workers", type=int, default=0,
        help="DataLoader workers; keep 0 to avoid forking after CUDA initialization",
    )
    parser.add_argument(
        "--no-video", action="store_true", help="only save PNG frames"
    )
    args, cfg_options = parser.parse_known_args()
    if args.num_scenes < 1 or args.max_frames < 0 or args.workers < 0:
        parser.error(
            "--num-scenes must be positive; --max-frames and --workers must be >= 0"
        )
    if args.scene_index is not None and args.scene_index < 0:
        parser.error("--scene-index must be >= 0")
    if args.scenes and args.scene_index is not None:
        parser.error("--scenes and --scene-index cannot be used together")
    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("fps and image dimensions must be positive")
    if not 0 < args.elevation < 90:
        parser.error("--elevation must be between 0 and 90 degrees")
    if args.bbox_radius <= 0:
        parser.error("--bbox-radius must be positive")
    if args.frustum_depth <= 0 or args.frustum_line_width <= 0:
        parser.error("--frustum-depth and --frustum-line-width must be positive")
    return args, cfg_options


def prepare_dataset_cfg(dataset_cfg):
    """Remove training/wrapper-only options for deterministic BS=1 inference."""
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


def load_scene_names(dataset):
    """Return scene-token -> readable-name using nuScenes tables when present."""
    version = getattr(dataset, "version", "")
    table = Path(dataset.dataset_root) / version / "scene.json"
    if not table.is_file():
        return {}
    with table.open("r", encoding="utf-8") as handle:
        return {row["token"]: row["name"] for row in json.load(handle)}


def scene_key(info):
    value = info.get("scene_token")
    if value:
        return str(value)
    occ = info.get("occ3d", info.get("occ_gt_path", {}))
    path = occ.get("occ_path", "") if isinstance(occ, dict) else str(occ)
    for part in Path(path).parts:
        if part.startswith("scene-"):
            return part
    raise ValueError(f"Cannot determine scene for sample {info.get('token')}")


def select_scene_indices(dataset, requested, scene_index, num_scenes, max_frames):
    names = load_scene_names(dataset)
    grouped = {}
    for index, info in enumerate(dataset.data_infos):
        grouped.setdefault(scene_key(info), []).append(index)
    for indices in grouped.values():
        indices.sort(key=lambda i: dataset.data_infos[i]["timestamp"])

    keys = sorted(grouped, key=lambda k: names.get(k, k))
    aliases = {key: key for key in keys}
    aliases.update({names[key]: key for key in keys if key in names})
    if requested:
        unknown = [item for item in requested if item not in aliases]
        if unknown:
            choices = ", ".join(names.get(k, k) for k in keys[:12])
            raise ValueError(f"Unknown scenes {unknown}; available examples: {choices}")
        selected = list(dict.fromkeys(aliases[item] for item in requested))
    elif scene_index is not None:
        if scene_index >= len(keys):
            raise ValueError(
                f"scene-index={scene_index} is out of range; dataset has {len(keys)} scenes"
            )
        selected = keys[scene_index:scene_index + num_scenes]
    else:
        count = min(num_scenes, len(keys))
        positions = np.linspace(0, len(keys) - 1, count, dtype=int)
        selected = [keys[position] for position in positions]

    index_to_scene = {}
    ordered_indices = []
    scene_labels = {}
    for key in selected:
        label = names.get(key, key)
        scene_labels[key] = label
        indices = grouped[key]
        if max_frames and len(indices) > max_frames:
            positions = np.linspace(0, len(indices) - 1, max_frames, dtype=int)
            indices = [indices[position] for position in positions]
        for index in indices:
            ordered_indices.append(index)
            index_to_scene[index] = key
    return ordered_indices, index_to_scene, scene_labels


def to_numpy(value):
    while hasattr(value, "data") and not torch.is_tensor(value):
        value = value.data
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def occupied_voxels(occupancy, camera_mask=None, surface_only=True):
    occupancy = to_numpy(occupancy).squeeze()
    if occupancy.shape != (200, 200, 16):
        raise ValueError(f"Expected occupancy (200, 200, 16), got {occupancy.shape}")
    visible = (occupancy > 0) & (occupancy < 17)
    if camera_mask is not None:
        camera_mask = to_numpy(camera_mask).squeeze()
        if camera_mask.shape != occupancy.shape:
            raise ValueError(
                f"Expected mask_camera {occupancy.shape}, got {camera_mask.shape}"
            )
        visible &= camera_mask.astype(bool)
    if surface_only:
        # A voxel completely surrounded by the same semantic class cannot be
        # seen from outside.  Removing these interior cubes preserves the
        # rendered surface and semantic boundaries while reducing glyph work.
        padded = np.pad(occupancy, 1, mode="constant", constant_values=255)
        padded_visible = np.pad(visible, 1, mode="constant", constant_values=False)
        same_neighbors = (
            (padded[:-2, 1:-1, 1:-1] == occupancy)
            & (padded[2:, 1:-1, 1:-1] == occupancy)
            & (padded[1:-1, :-2, 1:-1] == occupancy)
            & (padded[1:-1, 2:, 1:-1] == occupancy)
            & (padded[1:-1, 1:-1, :-2] == occupancy)
            & (padded[1:-1, 1:-1, 2:] == occupancy)
        )
        visible_neighbors = (
            padded_visible[:-2, 1:-1, 1:-1]
            & padded_visible[2:, 1:-1, 1:-1]
            & padded_visible[1:-1, :-2, 1:-1]
            & padded_visible[1:-1, 2:, 1:-1]
            & padded_visible[1:-1, 1:-1, :-2]
            & padded_visible[1:-1, 1:-1, 2:]
        )
        visible &= ~(same_neighbors & visible_neighbors)
    indices = np.argwhere(visible)
    xyz = (indices.astype(np.float32) + 0.5) * VOXEL_SIZE + OCC_ORIGIN
    return xyz, occupancy[visible].astype(np.float32)


def transform_box_corners(boxes, lidar_aug_matrix, lidar2ego, occ_aug_matrix):
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 8, 3), dtype=np.float32)
    corners = boxes.corners.detach().cpu().numpy().astype(np.float64)
    lidar_aug = to_numpy(lidar_aug_matrix).squeeze().astype(np.float64)
    lidar_to_ego = to_numpy(lidar2ego).squeeze().astype(np.float64)
    occ_aug = to_numpy(occ_aug_matrix).squeeze().astype(np.float64)
    for name, matrix in (
        ("lidar_aug_matrix", lidar_aug), ("lidar2ego", lidar_to_ego),
        ("occ_aug_matrix", occ_aug),
    ):
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected {name} (4, 4), got {matrix.shape}")
    # Model boxes live in the augmented LiDAR frame.  Occ3D voxels live in
    # the augmented ego frame used by CrossCoordinateSample.
    corners = (corners - lidar_aug[:3, 3]) @ np.linalg.inv(lidar_aug[:3, :3]).T
    corners = corners @ lidar_to_ego[:3, :3].T + lidar_to_ego[:3, 3]
    corners = corners @ occ_aug[:3, :3].T
    return corners.astype(np.float32)


def set_camera(figure, elevation, distance, focal_x):
    """Set a rear/upper camera and project ego +X vertically upwards on screen."""
    angle = math.radians(elevation)
    focal = np.array([focal_x, 0.0, 0.5], dtype=np.float64)
    look = np.array([math.cos(angle), 0.0, -math.sin(angle)], dtype=np.float64)
    position = focal - distance * look
    forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    view_up = forward - np.dot(forward, look) * look
    view_up /= np.linalg.norm(view_up)

    camera = figure.scene.camera
    camera.position = tuple(position)
    camera.focal_point = tuple(focal)
    camera.view_up = tuple(view_up)
    camera.parallel_projection = False
    camera.view_angle = 42.0
    camera.clipping_range = (0.1, 300.0)
    figure.scene.render()


def draw_line_set(mlab, figure, points, lines, color, line_width):
    """Draw disconnected 3D line segments with one lightweight VTK pipeline."""
    if not len(points) or not len(lines):
        return None
    points = np.asarray(points, dtype=np.float32)
    lines = np.asarray(lines, dtype=np.int64)
    source = mlab.pipeline.scalar_scatter(
        points[:, 0], points[:, 1], points[:, 2], figure=figure
    )
    source.mlab_source.dataset.lines = lines
    source.update()
    surface = mlab.pipeline.surface(source, color=color, figure=figure)
    surface.actor.property.line_width = float(line_width)
    return source


def draw_six_camera_frustums(mlab, figure, occ_aug_matrix, camera2ego,
                             camera_intrinsics, display_depth, line_width):
    """Draw six calibrated camera frustums in the occupancy ego frame."""
    camera2ego = to_numpy(camera2ego).squeeze().astype(np.float64)
    intrinsics = to_numpy(camera_intrinsics).squeeze().astype(np.float64)
    if camera2ego.shape != (6, 4, 4):
        raise ValueError(f"Expected camera2ego (6, 4, 4), got {camera2ego.shape}")
    if intrinsics.shape != (6, 4, 4):
        raise ValueError(
            f"Expected camera_intrinsics (6, 4, 4), got {intrinsics.shape}"
        )

    # Each frustum consists of one optical center, four far-plane corners,
    # four center-to-corner rays and the four edges of the far rectangle.
    # Pose and field of view are calibrated; depth is only a display scale.
    points = []
    lines = []
    for transform, intrinsic in zip(camera2ego, intrinsics):
        k = intrinsic[:3, :3]
        # nuScenes principal points are near the image center; deriving width
        # and height from (cx, cy) avoids depending on augmented image shapes.
        width, height = 2.0 * k[0, 2], 2.0 * k[1, 2]
        pixels = np.array(
            [[0.0, 0.0, 1.0], [width, 0.0, 1.0],
             [width, height, 1.0], [0.0, height, 1.0]],
            dtype=np.float64,
        )
        # Camera coordinates use z as the optical axis.  K^-1 maps image
        # corners to rays whose z component is one.
        far_camera = (np.linalg.inv(k) @ pixels.T).T * display_depth
        origin_ego = transform[:3, 3]
        far_ego = far_camera @ transform[:3, :3].T + transform[:3, 3]
        offset = len(points)
        points.extend((origin_ego,) + tuple(far_ego))
        lines.extend(
            # Four viewing rays.
            (offset, offset + corner) for corner in range(1, 5)
        )
        lines.extend((
            # Far image-plane rectangle.
            (offset + 1, offset + 2), (offset + 2, offset + 3),
            (offset + 3, offset + 4), (offset + 4, offset + 1),
        ))

    points = np.asarray(points, dtype=np.float64)
    occ_aug = to_numpy(occ_aug_matrix).squeeze().astype(np.float64)
    points = points @ occ_aug[:3, :3].T
    return draw_line_set(
        mlab, figure, points, lines, color=(0.02, 0.02, 0.02),
        line_width=line_width,
    )


def render_frame(mlab, figure, output_path, occupancy, camera_mask, boxes, scores,
                 lidar_aug_matrix, lidar2ego, occ_aug_matrix, score_threshold,
                 bbox_radius, elevation, distance, focal_x, camera2ego,
                 camera_intrinsics, frustum_depth, frustum_line_width,
                 render_state, render_interior=False, verbose=False):
    # Updating a reused mlab_source with a different point count changes x/y/z
    # and scalars in separate VTK arrays.  Prevent an intermediate render from
    # indexing the new shorter array with the previous frame's point count.
    figure.scene.disable_render = True
    xyz, labels = occupied_voxels(
        occupancy, camera_mask, surface_only=not render_interior
    )
    mask_status = "mask_camera enabled" if camera_mask is not None else "camera mask disabled"
    if verbose:
        print(f"Rendering {len(xyz):,} occupied voxels ({mask_status})...", flush=True)
    glyph = render_state.get("occupancy_glyph")
    if glyph is None and len(xyz):
        glyph = mlab.points3d(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], labels,
            mode="cube", scale_factor=VOXEL_SIZE * 0.94, opacity=1.0,
            vmin=0, vmax=17, figure=figure,
        )
        # Every occupancy cube has the same physical size; fixed scaling also
        # avoids keeping a per-point scale/vector array across frame sizes.
        glyph.glyph.scale_mode = "data_scaling_off"
        glyph.module_manager.scalar_lut_manager.lut.table = OCC_COLORS
        render_state["occupancy_glyph"] = glyph
    elif glyph is not None:
        # Reuse the VTK glyph pipeline; rebuilding it with mlab.clf() for
        # every frame is substantially more expensive than updating arrays.
        glyph.mlab_source.reset(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], scalars=labels
        )
    if verbose:
        print("Occupancy glyph created; drawing detection boxes...", flush=True)

    previous = render_state.pop("box_source", None)
    if previous is not None:
        previous.remove()
    if boxes is not None:
        score_array = to_numpy(scores).reshape(-1)
        keep = score_array >= score_threshold
        boxes = boxes[torch.from_numpy(keep)]
        corners = transform_box_corners(
            boxes, lidar_aug_matrix, lidar2ego, occ_aug_matrix
        )
        if verbose:
            print(f"Drawing {len(corners)} detection boxes in one VTK pipeline...", flush=True)
        if len(corners):
            # Creating one mlab.plot3d pipeline per edge is extremely slow.
            # Store all corners in one PolyData and connect them as lines.
            points = corners.reshape(-1, 3)
            lines = np.asarray(
                [
                    (box_index * 8 + start, box_index * 8 + end)
                    for box_index in range(len(corners))
                    for start, end in BOX_EDGES
                ],
                dtype=np.int64,
            )
            # Render PolyData lines directly instead of generating tube mesh.
            render_state["box_source"] = draw_line_set(
                mlab, figure, points, lines, color=(1.0, 0.78, 0.0),
                line_width=max(1.0, bbox_radius * 24.0),
            )

    previous = render_state.pop("camera_source", None)
    if previous is not None:
        previous.remove()
    render_state["camera_source"] = draw_six_camera_frustums(
        mlab, figure, occ_aug_matrix, camera2ego, camera_intrinsics,
        frustum_depth, frustum_line_width,
    )
    figure.scene.disable_render = False
    set_camera(figure, elevation, distance, focal_x)
    if verbose:
        print(f"Scene ready; saving {output_path}...", flush=True)
    mlab.savefig(str(output_path), figure=figure, size=figure.scene.get_size())
    if verbose:
        print(f"Saved {output_path}", flush=True)


def write_video(frame_paths, video_path, fps):
    if not frame_paths:
        return
    # An explicit concat list avoids accidentally including stale PNGs from a
    # previous run that used a different --max-frames value.
    duration = 1.0 / fps
    with tempfile.NamedTemporaryFile("w", suffix=".ffconcat", encoding="utf-8") as manifest:
        manifest.write("ffconcat version 1.0\n")
        for path in frame_paths:
            escaped = str(path.resolve()).replace("'", "'\\''")
            manifest.write(f"file '{escaped}'\nduration {duration:.9f}\n")
        # concat uses the following file to apply the final duration.
        escaped = str(frame_paths[-1].resolve()).replace("'", "'\\''")
        manifest.write(f"file '{escaped}'\n")
        manifest.flush()
        command = [
            "ffmpeg", "-y", "-loglevel", "warning", "-f", "concat", "-safe", "0",
            "-i", manifest.name, "-r", str(fps), "-fps_mode", "cfr",
            "-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_path),
        ]
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as error:
            raise RuntimeError("ffmpeg is required for MP4 output; use --no-video") from error
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"ffmpeg failed for {video_path}") from error


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
    indices, index_to_scene, scene_labels = select_scene_indices(
        dataset, args.scenes, args.scene_index, args.num_scenes, args.max_frames
    )
    if not indices:
        raise RuntimeError("No frames selected")

    subset = torch.utils.data.Subset(dataset, indices)
    loader = build_dataloader(
        subset, samples_per_gpu=1, workers_per_gpu=args.workers,
        dist=False, shuffle=False,
    )

    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = cfg.get("cudnn_benchmark", False)
    model = build_model(cfg.model)
    if cfg.get("fp16") is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model.cuda(args.gpu), device_ids=[args.gpu])
    model.eval()

    # mmdet3d imports cv2 while constructing the model.  cv2 points Qt back
    # to its bundled plugins, so repair the environment only after all model
    # imports/building have completed and immediately before Mayavi starts.
    configure_qt_for_mayavi()
    from mayavi import mlab

    mlab.options.offscreen = True
    figure = mlab.figure(
        size=(args.width, args.height), bgcolor=(1.0, 1.0, 1.0),
        fgcolor=(0.1, 0.1, 0.1),
    )
    output_root = Path(args.out_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    scene_frames = {key: [] for key in scene_labels}
    render_state = {}

    try:
        progress = tqdm(loader, desc="Inference and rendering")
        for position, data in enumerate(progress):
            dataset_index = indices[position]
            scene = index_to_scene[dataset_index]
            label = scene_labels[scene]
            info = dataset.data_infos[dataset_index]
            token = str(info["token"])
            frames_dir = output_root / label / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            frame_path = frames_dir / f"{len(scene_frames[scene]):04d}_{token}.png"

            if position == 0:
                tqdm.write("First frame loaded; starting CUDA inference...")
            with torch.inference_mode():
                output = model(**data)[0]
            if "occ_pred" not in output:
                raise KeyError("Model output does not contain occ_pred")
            if position == 0:
                tqdm.write("First inference complete; starting Mayavi rendering...")
            render_frame(
                mlab, figure, frame_path, output["occ_pred"],
                None if args.no_camera_mask else data.get("mask_camera"),
                output.get("boxes_3d"), output.get("scores_3d", np.empty(0)),
                data["lidar_aug_matrix"], data["lidar2ego"], data["occ_aug_matrix"],
                args.bbox_score, args.bbox_radius,
                args.elevation, args.camera_distance, args.focal_x,
                data["camera2ego"], data["camera_intrinsics"],
                args.frustum_depth, args.frustum_line_width,
                render_state, render_interior=args.render_interior,
                verbose=(position == 0),
            )
            scene_frames[scene].append(frame_path)
            if position == 0:
                tqdm.write(f"First frame saved: {frame_path}")
    finally:
        mlab.close(figure)

    if not args.no_video:
        for scene, frame_paths in scene_frames.items():
            label = scene_labels[scene]
            video_path = output_root / f"{label}.mp4"
            write_video(frame_paths, video_path, args.fps)
            print(f"Video: {video_path} ({len(frame_paths)} frames)")
    for scene, frame_paths in scene_frames.items():
        print(f"Frames: {output_root / scene_labels[scene] / 'frames'} ({len(frame_paths)})")


if __name__ == "__main__":
    main()
