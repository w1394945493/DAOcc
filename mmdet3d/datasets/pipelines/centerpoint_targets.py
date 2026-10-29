import torch

from mmdet3d.core.utils import draw_heatmap_gaussian, gaussian_radius
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class GenerateCenterHeadTargets:
    # (10.29 检测GT迁移) 在 DataLoader worker 中，根据完成三维增强和范围过滤后的
    # (10.29 检测GT迁移) GT 框生成 CenterHead 需要的全部监督目标。这样训练主进程
    # (10.29 检测GT迁移) 不再于每次 head.loss() 中串行执行 get_targets()。
    def __init__(self, tasks, train_cfg, norm_bbox=True):
        self.tasks = tasks
        self.train_cfg = train_cfg
        self.norm_bbox = norm_bbox

    def __call__(self, results):
        gt_bboxes_3d = results["gt_bboxes_3d"]
        gt_labels_3d = torch.as_tensor(results["gt_labels_3d"], dtype=torch.long)

        # (10.29 检测GT迁移) gravity_center 将框中心统一为 CenterHead 使用的重心坐标；
        # (10.29 检测GT迁移) 后面的尺寸、朝向和速度仍沿用原框张量中的字段顺序。
        gt_bboxes = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]), dim=1
        ).cpu()
        gt_labels_3d = gt_labels_3d.cpu()

        max_objs = self.train_cfg["max_objs"] * self.train_cfg["dense_reg"]
        grid_size = torch.tensor(self.train_cfg["grid_size"])
        pc_range = torch.tensor(self.train_cfg["point_cloud_range"])
        voxel_size = torch.tensor(self.train_cfg["voxel_size"])
        feature_map_size = torch.div(
            grid_size[:2],
            self.train_cfg["out_size_factor"],
            rounding_mode="trunc",
        )

        # (10.29 检测GT迁移) 按 CenterHead 的 6 个 task 重新组织 GT。
        # (10.29 检测GT迁移) 每个 task 内的类别编号从 1 开始，0 留作背景。
        task_boxes = []
        task_classes = []
        class_offset = 0
        for task in self.tasks:
            boxes_in_task = []
            classes_in_task = []
            for class_id in range(class_offset, class_offset + len(task)):
                class_mask = gt_labels_3d == class_id
                boxes_in_task.append(gt_bboxes[class_mask])
                classes_in_task.append(gt_labels_3d[class_mask] + 1 - class_offset)
            task_boxes.append(torch.cat(boxes_in_task, dim=0))
            task_classes.append(torch.cat(classes_in_task, dim=0).long())
            class_offset += len(task)

        heatmaps, anno_boxes, inds, masks = [], [], [], []
        for task_id, task in enumerate(self.tasks):
            # (10.29 检测GT迁移) heatmap 监督类别中心；其空间大小由
            # (10.29 检测GT迁移) grid_size / out_size_factor 得到，当前为 180×180。
            heatmap = gt_bboxes.new_zeros(
                (len(task), feature_map_size[1], feature_map_size[0])
            )
            anno_box = gt_bboxes.new_zeros((max_objs, 10), dtype=torch.float32)
            ind = gt_labels_3d.new_zeros((max_objs,), dtype=torch.int64)
            mask = gt_bboxes.new_zeros((max_objs,), dtype=torch.uint8)

            num_objs = min(task_boxes[task_id].shape[0], max_objs)
            for obj_id in range(num_objs):
                cls_id = task_classes[task_id][obj_id] - 1
                width = task_boxes[task_id][obj_id][3]
                length = task_boxes[task_id][obj_id][4]
                width = width / voxel_size[0] / self.train_cfg["out_size_factor"]
                length = length / voxel_size[1] / self.train_cfg["out_size_factor"]

                if width <= 0 or length <= 0:
                    continue

                radius = gaussian_radius(
                    (length, width), min_overlap=self.train_cfg["gaussian_overlap"]
                )
                radius = max(self.train_cfg["min_radius"], int(radius))

                x, y, z = task_boxes[task_id][obj_id][0:3]
                coor_x = (
                    (x - pc_range[0])
                    / voxel_size[0]
                    / self.train_cfg["out_size_factor"]
                )
                coor_y = (
                    (y - pc_range[1])
                    / voxel_size[1]
                    / self.train_cfg["out_size_factor"]
                )
                center = torch.tensor([coor_x, coor_y], dtype=torch.float32)
                center_int = center.to(torch.int32)

                # (10.29 检测GT迁移) 过滤增强后落到检测特征图之外的框，避免产生
                # (10.29 检测GT迁移) 越界索引；逻辑与原 get_targets_single() 一致。
                if not (
                    0 <= center_int[0] < feature_map_size[0]
                    and 0 <= center_int[1] < feature_map_size[1]
                ):
                    continue

                draw_heatmap_gaussian(
                    heatmap[cls_id], center_int[[1, 0]], radius
                )
                # (10.29 检测GT迁移) 沿用原 CenterHead 的展平顺序 x * H + y；
                # (10.29 检测GT迁移) 不能改成常见的 y * W + x，否则 gather 位置会变化。
                new_idx = center_int[0] * feature_map_size[1] + center_int[1]
                ind[obj_id] = new_idx
                mask[obj_id] = 1

                rot = task_boxes[task_id][obj_id][6]
                box_dim = task_boxes[task_id][obj_id][3:6]
                if self.norm_bbox:
                    box_dim = box_dim.log()
                vx, vy = task_boxes[task_id][obj_id][7:]

                # (10.29 检测GT迁移) 框回归目标依次为中心小数偏移、z、log 尺寸、
                # (10.29 检测GT迁移) sin/cos 朝向及 vx/vy；形状固定为 [max_objs, 10]。
                anno_box[obj_id] = torch.cat(
                    [
                        center - torch.tensor([center_int[0], center_int[1]]),
                        z.unsqueeze(0),
                        box_dim,
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0),
                        vx.unsqueeze(0),
                        vy.unsqueeze(0),
                    ]
                )

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            inds.append(ind)
            masks.append(mask)

        # (10.29 检测GT迁移) MMCV collate 会递归处理该字典，把每个 task 的
        # (10.29 检测GT迁移) 单样本张量堆叠为 [B, ...]，随后递归搬到训练 GPU。
        results["object_targets"] = {
            "heatmaps": heatmaps,
            "anno_boxes": anno_boxes,
            "inds": inds,
            "masks": masks,
        }
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(tasks={self.tasks}, "
            f"train_cfg={self.train_cfg}, norm_bbox={self.norm_bbox})"
        )
