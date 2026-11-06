#!/usr/bin/env python3
"""原DAOcc（BEVTransformV2）相机BEV、LiDAR BEV、融合BEV、Occ预测与GT对照。

与visualize_depth_lss_bev.py完全共用采样、推理、坐标对齐、归一化和绘图流程。
两次运行使用相同验证集、场景/帧选项和camera-mask选项，即可比较同帧结果。
"""
from visualize_depth_lss_bev import main as visualize_bev


def main():
    #* 仅切换允许的视角变换、图标题和清单文件名；不修改模型或特征内容。
    # 仍保留五列，包括融合BEV，便于与DepthLSS输出逐列对照。
    visualize_bev(
        expected_vtransform="BEVTransformV2",
        method_name="DAOcc",
        manifest_name="daocc_bev_manifest.json",
        description=__doc__,
    )


if __name__ == "__main__":
    main()
