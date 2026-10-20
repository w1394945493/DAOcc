import os
import argparse
# 此脚本不使用 OpenCV；导入 cv2 会将 Qt 插件路径指向其自带目录，干扰 Mayavi。
# import cv2, imageio
#* Mayavi 使用 PyQt5 创建窗口；显式指定当前环境的 Qt 插件，避免误用 OpenCV 自带的不完整插件。
try:
    import PyQt5

    pyqt_plugin_root = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
    pyqt_platform_root = os.path.join(pyqt_plugin_root, "platforms")
    if os.path.isdir(pyqt_platform_root):
        os.environ["QT_PLUGIN_PATH"] = pyqt_plugin_root
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = pyqt_platform_root
except ImportError:
    #* 未安装 PyQt5 时保留系统 Qt 配置，让 Mayavi 输出明确的依赖错误。
    pass

#* WSLg 默认通过 XWayland 提供 DISPLAY；允许命令行环境变量覆盖此默认值。
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("ETS_TOOLKIT", "qt")
# Mayavi 延迟到绘图时导入，参数帮助和数组校验不需要启动图形界面。
import numpy as np
# torch 在本脚本中未使用，无需加载。
# import torch


colors = np.array(
    [
        [0, 0, 0, 255],
        [255, 120, 50, 255],  # barrier              orangey
        [255, 192, 203, 255],  # bicycle              pink
        [255, 255, 0, 255],  # bus                  yellow
        [0, 150, 245, 255],  # car                  blue
        [0, 255, 255, 255],  # construction_vehicle cyan
        [200, 180, 0, 255],  # motorcycle           dark orange
        [255, 0, 0, 255],  # pedestrian           red
        [255, 240, 150, 255],  # traffic_cone         light yellow
        [135, 60, 0, 255],  # trailer              brown
        [160, 32, 240, 255],  # truck                purple
        [255, 0, 255, 255],  # driveable_surface    dark pink
        # [175,   0,  75, 255],       # other_flat           dark red
        [139, 137, 137, 255],
        [75, 0, 75, 255],  # sidewalk             dard purple
        [150, 240, 80, 255],  # terrain              light green
        [230, 230, 250, 255],  # manmade              white
        [0, 175, 0, 255],  # vegetation           green
        [0, 0, 255, 255],  # ego car
        [255, 99, 71, 255],
        [0, 191, 255, 255]
    ]
).astype(np.uint8)

voxel_size = 0.4
pc_range = [-40, -40, -1, 40, 40, 5.4]

def load_visible_voxels(visual_path, camera_mask_path=None):
    #* 兼容 CUDA-DAOcc 导出的逗号分隔 occ_pred.txt，按 TensorRT 输出顺序恢复体素数组。
    if str(visual_path).lower().endswith(".txt"):
        semantics = np.fromfile(visual_path, dtype=np.int32, sep=",")
        expected_size = 200 * 200 * 16
        if semantics.size != expected_size:
            raise ValueError(
                f"Expected {expected_size} occupancy values in {visual_path}, got {semantics.size}"
            )
        semantics = semantics.reshape(200, 200, 16)
    else:
        # 兼容预测 .npy 和原始真值 labels.npz，使用相同的 Occ3D 体素坐标。
        loaded = np.load(visual_path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            with loaded:
                semantics = loaded["semantics"]
        else:
            semantics = loaded
    if semantics.shape != (200, 200, 16):
        raise ValueError(f"Expected occupancy shape (200, 200, 16), got {semantics.shape}")

    # 保留原脚本隐藏类别 0 和 free=17 的行为，同时排除 ignore=255。
    visible = (semantics > 0) & (semantics < 17)
    if camera_mask_path is not None:
        # 掩码必须对应同一帧；npz 读取 mask_camera，npy 则直接读取掩码数组。
        loaded_mask = np.load(camera_mask_path, allow_pickle=False)
        if isinstance(loaded_mask, np.lib.npyio.NpzFile):
            with loaded_mask:
                if "mask_camera" not in loaded_mask:
                    raise ValueError(f"mask_camera not found in {camera_mask_path}")
                camera_mask = loaded_mask["mask_camera"]
        else:
            camera_mask = loaded_mask
        if camera_mask.shape != semantics.shape:
            raise ValueError(f"Mask shape {camera_mask.shape} != occupancy shape {semantics.shape}")
        if not np.all((camera_mask == 0) | (camera_mask == 1)):
            raise ValueError("Camera mask must contain only 0 and 1")
        visible &= camera_mask.astype(bool)

    # 向量化提取可见占用体素，避免三层 Python 循环；统一浮点坐标以保留体素中心。
    indices = np.argwhere(visible)
    voxels = np.empty((len(indices), 4), dtype=np.float64)
    voxels[:, :3] = (indices + 0.5) * voxel_size + np.asarray(pc_range[:3])
    voxels[:, 3] = semantics[visible]
    return voxels


def load_lidar2ego(transform_path):
    #* 支持直接保存的 4x4 .npy/.txt，以及 CUDA-DAOcc 示例数据中的 example-data.pth。
    suffix = os.path.splitext(transform_path)[1].lower()
    if suffix == ".npy":
        matrix = np.load(transform_path, allow_pickle=False)
    elif suffix in (".txt", ".csv"):
        matrix = np.loadtxt(transform_path)
    elif suffix in (".pth", ".pt"):
        import torch

        saved = torch.load(transform_path, map_location="cpu")
        if not isinstance(saved, dict) or "lidar2ego" not in saved:
            raise ValueError(f"lidar2ego not found in {transform_path}")
        matrix = saved["lidar2ego"]
        #* mmcv DataContainer 和批次列表会额外包裹矩阵，逐层取出当前样例的第一项。
        if not torch.is_tensor(matrix) and hasattr(matrix, "data"):
            matrix = matrix.data
        while isinstance(matrix, (list, tuple)):
            matrix = matrix[0]
        if torch.is_tensor(matrix):
            matrix = matrix.detach().cpu().numpy()
    else:
        raise ValueError("lidar2ego must be a .pth, .pt, .npy, .txt or .csv file")
    matrix = np.asarray(matrix, dtype=np.float64).squeeze()
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected lidar2ego shape (4, 4), got {matrix.shape}")
    return matrix


def draw_prediction_boxes(mlab, box_path, lidar2ego=None, box_radius=0.10):
    #* CUDA-DAOcc 每行格式为 x y z w l h vx vy yaw class_id score，坐标位于 LiDAR 系。
    boxes = np.loadtxt(box_path, dtype=np.float64, ndmin=2)
    if boxes.size == 0:
        return 0
    if boxes.shape[1] != 11:
        raise ValueError(f"Expected 11 columns in {box_path}, got {boxes.shape[1]}")

    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    #* 检测类别顺序与 occupancy 类别顺序不同，将检测 id 映射到对应的 occupancy 颜色。
    object_to_occ_color = (4, 10, 5, 3, 9, 1, 6, 2, 7, 8)
    for x, y, z, width, length, height, _, _, yaw, class_id, _ in boxes:
        local_xy = np.array([
            [width / 2, length / 2],
            [-width / 2, length / 2],
            [-width / 2, -length / 2],
            [width / 2, -length / 2],
        ])
        rotation = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ])
        xy = local_xy @ rotation.T + np.array([x, y])
        lower = np.column_stack((xy, np.full(4, z - height / 2)))
        upper = np.column_stack((xy, np.full(4, z + height / 2)))
        corners = np.vstack((lower, upper))
        if lidar2ego is not None:
            #* Occ3D occupancy 位于 ego 系；对框的全部角点应用 LiDAR 到 ego 的完整旋转和平移。
            corners_h = np.column_stack((corners, np.ones(len(corners))))
            corners = (lidar2ego @ corners_h.T).T[:, :3]
        object_id = int(class_id)
        color_index = object_to_occ_color[object_id] if 0 <= object_id < len(object_to_occ_color) else 19
        color = tuple((colors[color_index, :3] / 255.0).tolist())
        for start, end in edges:
            #* 使用加粗的类别颜色线绘制框，保留类别信息且不添加黑色外轮廓。
            mlab.plot3d(
                corners[[start, end], 0],
                corners[[start, end], 1],
                corners[[start, end], 2],
                color=color,
                tube_radius=box_radius,
            )
    return len(boxes)


def main():
    parser = argparse.ArgumentParser(description="Visualize Occ3D occupancy with optional boxes and camera mask")
    parser.add_argument("visual_path", help="occupancy .txt, .npy or labels.npz")
    #* 可叠加 CUDA-DAOcc 保存的 object_pred.txt 三维检测框。
    parser.add_argument("--boxes", metavar="PATH", help="CUDA-DAOcc object_pred.txt")
    #* Occ3D occupancy 使用 ego 坐标；传入 example-data.pth 后将 LiDAR 检测框转换到同一坐标系。
    parser.add_argument(
        "--lidar2ego", metavar="PATH",
        help="example-data.pth or a 4x4 .npy/.txt LiDAR-to-ego matrix")
    #* 检测框保留对应类别颜色，可通过半径参数调整粗细。
    parser.add_argument("--bbox-radius", type=float, default=0.10, help="box tube radius (default: 0.10)")
    # 指定此参数才按相机掩码过滤，不指定则维持全范围占用可视化。
    parser.add_argument("--camera-mask", metavar="PATH", help="same-frame labels.npz or binary mask .npy")
    args = parser.parse_args()
    fov_voxels = load_visible_voxels(args.visual_path, args.camera_mask)
    print(f"Visible occupied voxels: {len(fov_voxels)}; camera mask: {args.camera_mask or 'disabled'}")
    if len(fov_voxels) == 0:
        print("No occupied voxels to display after filtering.")
        return
    import mayavi.mlab as mlab

    # 默认窗口缩小为 1280×720，便于在桌面查看和操作。
    figure = mlab.figure(size=(1280, 720), bgcolor=(1, 1, 1))
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05*voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=19,
    )


    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

    box_count = 0
    if args.boxes is not None:
        lidar2ego = load_lidar2ego(args.lidar2ego) if args.lidar2ego else None
        if lidar2ego is None:
            print("Warning: boxes remain in LiDAR coordinates; use --lidar2ego for Occ3D alignment.")
        if args.bbox_radius <= 0:
            parser.error("--bbox-radius must be > 0")
        box_count = draw_prediction_boxes(mlab, args.boxes, lidar2ego, args.bbox_radius)
        coordinate_name = "ego" if lidar2ego is not None else "LiDAR"
        print(f"Prediction boxes: {box_count}; box coordinates: {coordinate_name}")

    mlab.view(azimuth=45, elevation=45, distance=180, focalpoint=[0, 0, -10])
    mlab.show()


if __name__ == "__main__":
    main()
