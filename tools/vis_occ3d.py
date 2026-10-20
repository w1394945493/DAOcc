import os
import argparse
# 此脚本不使用 OpenCV；导入 cv2 会将 Qt 插件路径指向其自带目录，干扰 Mayavi。
# import cv2, imageio
# 仅清理可能从外部继承的 OpenCV Qt 路径，保留用户的其他 Qt 配置。
for qt_variable in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
    if "/cv2/qt/" in os.environ.get(qt_variable, "").replace("\\", "/"):
        os.environ.pop(qt_variable, None)
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


def main():
    parser = argparse.ArgumentParser(description="Visualize Occ3D occupancy with an optional camera mask")
    parser.add_argument("visual_path", help="occupancy .npy or labels.npz")
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

    mlab.view(azimuth=45, elevation=45, distance=180, focalpoint=[0, 0, -10])
    mlab.show()


if __name__ == "__main__":
    main()
