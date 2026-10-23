import argparse
#* 读取 nuScenes 的场景表和样本表，用于定位每个样本的 Occ3D 标注。
import json

#* 读取和写回 info 的 pkl 文件，保留已有标注和元数据。
import mmcv
from os import path as osp

#* 转换器改为在生成基础 info 时导入；仅关联已有 pkl 不加载转换器。
# from data_converter import nuscenes_converter as nuscenes_converter
#* 当前 DAOcc 配置不使用目标库，保留导入供后续恢复。
# from data_converter.create_gt_database import create_groundtruth_database
# from data_converter import kitti_converter as kitti



#* 按 nuScenes 的 sample token 和 scene name 关联真实 Occ3D 标注，不依赖文件遍历顺序。
def attach_occ3d_annotations(root_path, version, out_dir, info_prefix, occ3d_root):
    """Attach Occ3D directories to train/val infos after validating all paths."""
    #* 展开用户目录并转为绝对路径，先检查真值根目录是否存在。
    occ3d_root = osp.abspath(osp.expanduser(occ3d_root))
    if not osp.isdir(occ3d_root):
        raise FileNotFoundError(f"Occ3D root directory does not exist: {occ3d_root}")
    #* scene.json 提供场景 token 到场景名的映射，场景名对应 gts 下的目录。
    with open(osp.join(root_path, version, "scene.json")) as f:
        scenes = {scene["token"]: scene["name"] for scene in json.load(f)}
    #* sample.json 将样本 token 关联到所属场景，避免将不同场景的标注配错。
    with open(osp.join(root_path, version, "sample.json")) as f:
        #* 同时保留原始场景 token，关联 Occ3D 时也可补齐旧版 info 的场景字段。
        samples = json.load(f)
        sample_scene_tokens = {sample["token"]: sample["scene_token"] for sample in samples}
        # sample_scenes = {
        #     sample["token"]: scenes[sample["scene_token"]] for sample in json.load(f)
        # }
        sample_scenes = {
            sample["token"]: scenes[sample["scene_token"]] for sample in samples
        }

    #* 暂存两个划分的修改；路径检查完成前不写回文件。
    pending = []
    missing = []
    for split in ("train", "val"):
        info_path = osp.join(out_dir, f"{info_prefix}_infos_{split}.pkl")
        data = mmcv.load(info_path)
        for info in data["infos"]:
            token = info["token"]
            #* info 中的样本必须属于当前 nuScenes 版本，否则立即提示数据不匹配。
            if token not in sample_scenes:
                raise ValueError(f"Unknown sample token in {info_path}: {token}")
            #* Ray-IoU 需要顶层 scene_token；不能用 occ_path 或样本 token 代替。
            info["scene_token"] = sample_scene_tokens[token]
            #* 标注布局为 gts/场景名/样本 token/labels.npz；这里只验证文件存在性。
            occ_path = osp.join(occ3d_root, sample_scenes[token], token)
            label_path = osp.join(occ_path, "labels.npz")
            if not osp.isfile(label_path):
                missing.append(label_path)
            #* 加载器会追加 labels.npz；保存绝对目录，避免依赖训练启动目录。
            info.setdefault("occ3d", {})["occ_path"] = occ_path
        pending.append((info_path, data))

    #* 全部 train/val 样本检查通过才写回，缺标注时不静默丢弃样本。
    if missing:
        examples = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"Missing Occ3D labels for {len(missing)} samples. "
            f"First {min(10, len(missing))} paths:\n{examples}"
        )
    #* 写回原 train/val info，保留原文件名和其他字段，并打印成功关联数量。
    for info_path, data in pending:
        mmcv.dump(data, info_path)
        print(f"Linked Occ3D labels for {len(data['infos'])} samples: {info_path}")


#* =================== 关联 SurroundOcc 标注 ===================
def attach_surroundocc_annotations(info_dir, out_dir, info_prefix, surroundocc_root,
                                   output_prefix=None):
    """保留原 info 内容和顺序，检查两个划分的标注后再保存。"""
    info_dir = osp.abspath(osp.expanduser(info_dir))
    out_dir = osp.abspath(osp.expanduser(out_dir))
    surroundocc_root = osp.abspath(osp.expanduser(surroundocc_root))
    if not osp.isdir(surroundocc_root):
        raise FileNotFoundError(f"SurroundOcc samples directory does not exist: {surroundocc_root}")
    pending, missing = [], []
    detected_version = None
    for split in ("train", "val"):
        filename = f"{info_prefix}_infos_{split}.pkl"
        data = mmcv.load(osp.join(info_dir, filename))
        #* 仅关联模式根据 pkl 元数据判断版本，不依赖命令行默认的 --version。
        version = data.get("metadata", {}).get("version")
        if not version:
            raise ValueError(f"Missing metadata.version in {filename}")
        if detected_version is not None and version != detected_version:
            raise ValueError("Train/val info versions do not match")
        detected_version = version
        #* 输入前缀不变；默认输出标明 surroundocc，mini 标记统一放在其后。
        # 显式 --output-prefix 仍允许自定义；避免已有 surroundocc/mini 后缀重复。
        target_prefix = output_prefix or info_prefix
        if output_prefix is None:
            if target_prefix.endswith("_mini"):
                target_prefix = target_prefix[:-5]
            if not target_prefix.endswith("_surroundocc"):
                target_prefix += "_surroundocc"
        if version == "v1.0-mini" and not target_prefix.endswith("_mini"):
            target_prefix += "_mini"
        output_filename = f"{target_prefix}_infos_{split}.pkl"
        for info in data["infos"]:
            #* 用当前关键帧的完整点云文件名匹配，不用 sweep、遍历序号或近似时间戳。
            # 示例：xxx.pcd.bin 对应 xxx.pcd.bin.npy；保留 .pcd.bin 后缀。
            label_path = osp.join(surroundocc_root, osp.basename(info["lidar_path"]) + ".npy")
            if not osp.isfile(label_path):
                missing.append(label_path)
            #* 加载器需要完整 npy 路径；原 occ3d、scene_token、sweeps 等字段全部保留。
            info.setdefault("surround_occ", {})["occ_path"] = label_path
        pending.append((osp.join(out_dir, output_filename), data))
    #* 不静默丢帧；解压未完成时明确报错，且两个输出 pkl 均不写入。
    if missing:
        raise FileNotFoundError(
            f"Missing SurroundOcc labels for {len(missing)} samples. "
            "Finish extracting the annotations before retrying. First 10 paths:\n"
            + "\n".join(missing[:10])
        )
    mmcv.mkdir_or_exist(out_dir)
    for path, data in pending:
        mmcv.dump(data, path)
        print(f"Linked SurroundOcc labels for {len(data['infos'])} samples: {path}")


def nuscenes_data_prep(
    root_path,
    info_prefix,
    version,
    dataset_name,
    out_dir,
    max_sweeps=10,
    load_augmented=None,
    #* 不传此参数时只生成普通 info；传入时补充 occ3d.occ_path。
    occ3d_root=None,
    #* 可选的 SurroundOcc samples 目录，可与 Occ3D 同时关联。
    surroundocc_root=None,
):
    """Prepare data related to nuScenes dataset.

    Generate info pickle files and optionally link Occ3D annotations.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the info pickle files.
        occ3d_root (str, optional): Root containing scene/sample/labels.npz.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    #* 从头生成 mini info 时也显式标记版本，后续标注关联沿用同一前缀。
    if version == "v1.0-mini" and not info_prefix.endswith("_mini"):
        info_prefix += "_mini"
    if load_augmented is None:
        # otherwise, infos must have been created, we just skip.
        from data_converter import nuscenes_converter
        nuscenes_converter.create_nuscenes_infos(
            #* 将输出目录传给转换器，确保 info 的生成与读取位置一致。
            root_path, info_prefix, version=version, max_sweeps=max_sweeps,
            out_dir=out_dir,
        )

        # if version == "v1.0-test":
        #     info_test_path = osp.join(root_path, f"{info_prefix}_infos_test.pkl")
        #     nuscenes_converter.export_2d_annotation(root_path, info_test_path, version=version)
        #     return

        # info_train_path = osp.join(root_path, f"{info_prefix}_infos_train.pkl")
        # info_val_path = osp.join(root_path, f"{info_prefix}_infos_val.pkl")
        # nuscenes_converter.export_2d_annotation(root_path, info_train_path, version=version)
        # nuscenes_converter.export_2d_annotation(root_path, info_val_path, version=version)

    #* test 集不关联 Occ3D 真值；生成基础 info 后直接返回。
    if version == "v1.0-test":
        return

    #* 仅关联训练/验证集的 occupancy 真值，保留普通 info 文件名。
    if occ3d_root is not None:
        attach_occ3d_annotations(root_path, version, out_dir, info_prefix, occ3d_root)

    #* 基础 info 生成后，关联 SurroundOcc；不修改类别或标注数组。
    if surroundocc_root is not None:
        attach_surroundocc_annotations(out_dir, out_dir, info_prefix, surroundocc_root)

    #* 当前训练 pipeline 未启用 ObjectPaste，无需生成目标 .bin 和 dbinfos；保留原调用。
    # create_groundtruth_database(
    #     dataset_name,
    #     root_path,
    #     info_prefix,
    #* 以下保留的输出参数用于日后恢复目标库生成，当前不会执行。
    #     osp.join(out_dir, f"{info_prefix}_infos_train.pkl"),
    #     database_save_path=osp.join(out_dir, f"{info_prefix}_gt_database"),
    #     db_info_save_path=osp.join(out_dir, f"{info_prefix}_dbinfos_train.pkl"),
    #     load_augmented=load_augmented,
    # )


def waymo_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    workers,
                    max_sweeps=5):
    """Prepare the info file for waymo dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
        max_sweeps (int): Number of input consecutive frames. Default: 5 \
            Here we store pose information of these frames for later use.
    """
    from data_converter import waymo_converter as waymo
    from data_converter import kitti_converter as kitti

    splits = ['training', 'validation', 'testing']
    for i, split in enumerate(splits):
        load_dir = osp.join(root_path, 'waymo_format', split)
        if split == 'validation':
            save_dir = osp.join(out_dir, 'kitti_format', 'training')
        else:
            save_dir = osp.join(out_dir, 'kitti_format', split)
        converter = waymo.Waymo2KITTI(
            load_dir,
            save_dir,
            prefix=str(i),
            workers=workers,
            test_mode=(split == 'test'))
        converter.convert()
    # Generate waymo infos
    out_dir = osp.join(out_dir, 'kitti_format')
    kitti.create_waymo_info_file(out_dir, info_prefix, max_sweeps=max_sweeps)
    # create_groundtruth_database(
    #     'WaymoDataset',
    #     out_dir,
    #     info_prefix,
    #     f'{out_dir}/{info_prefix}_infos_train.pkl',
    #     relative_path=False,
    #     with_mask=False)


parser = argparse.ArgumentParser(description="Data converter arg parser")
parser.add_argument("dataset", metavar="kitti", help="name of the dataset")
parser.add_argument(
    "--root-path",
    type=str,
    default="./data/kitti",
    help="specify the root path of dataset",
)
parser.add_argument(
    "--version",
    type=str,
    default="v1.0",
    required=False,
    help="specify the dataset version, no need for kitti",
)
parser.add_argument(
    "--max-sweeps",
    type=int,
    default=10,
    required=False,
    help="specify sweeps of lidar per example",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default="./data/kitti",
    required=False,
    help="name of info pkl",
)
#* 可选的 Occ3D 真值根目录，布局为 <root>/<scene_name>/<sample_token>/labels.npz。
parser.add_argument(
    "--occ3d-root",
    type=str,
    default=None,
    help="Occ3D ground-truth root containing scene/sample/labels.npz",
)
#* 指向直接存放 *.pcd.bin.npy 的目录，不是其上级目录。
parser.add_argument("--surroundocc-root", type=str, default=None,
                    help="SurroundOcc samples directory containing *.pcd.bin.npy")
#* 传入已有 info 目录即启用仅关联模式：跳过基础数据生成，另存到 --out-dir。
# 输入使用 --extra-tag；输出可另设 --output-prefix，mini 自动添加 _mini。
parser.add_argument("--info-dir", type=str, default=None,
                    help="Reuse existing train/val infos and only attach SurroundOcc labels")
#* 仅关联模式可单独指定输出前缀；不影响输入文件的查找。
parser.add_argument("--output-prefix", type=str, default=None,
                    help="Output prefix for --info-dir mode (mini suffix is added automatically)")
parser.add_argument("--extra-tag", type=str, default="kitti")
parser.add_argument("--painted", default=False, action="store_true")
parser.add_argument("--virtual", default=False, action="store_true")
parser.add_argument(
    "--workers", type=int, default=4, help="number of threads to be used"
)
args = parser.parse_args()

if __name__ == "__main__":
    if args.output_prefix and not args.info_dir:
        parser.error("--output-prefix requires --info-dir; otherwise use --extra-tag")
    load_augmented = None
    if args.virtual:
        if args.painted:
            load_augmented = "mvp"
        else:
            load_augmented = "pointpainting"

    #* =================== 复用已有 pkl 的快捷入口 ===================
    if args.info_dir is not None:
        if args.dataset != "nuscenes" or not args.surroundocc_root:
            parser.error("--info-dir requires nuscenes and --surroundocc-root")
        if args.occ3d_root or args.virtual or args.painted:
            parser.error("--info-dir only attaches SurroundOcc; existing Occ3D fields are preserved")
        attach_surroundocc_annotations(
            args.info_dir, args.out_dir, args.extra_tag, args.surroundocc_root,
            output_prefix=args.output_prefix)
    elif args.dataset == "nuscenes" and args.version != "v1.0-mini":
        train_version = f"{args.version}-trainval"
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            load_augmented=load_augmented,
            #* 透传命令行真值根目录；准备函数仅对 train/val 执行关联。
            occ3d_root=args.occ3d_root,
            surroundocc_root=args.surroundocc_root,
        )
        test_version = f"{args.version}-test"
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            load_augmented=load_augmented,
            #* 透传命令行真值根目录；准备函数仅对 train/val 执行关联。
            occ3d_root=args.occ3d_root,
            surroundocc_root=args.surroundocc_root,
        )
    elif args.dataset == "nuscenes" and args.version == "v1.0-mini":
        train_version = f"{args.version}"
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name="NuScenesDataset",
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            load_augmented=load_augmented,
            #* 透传命令行真值根目录；准备函数仅对 train/val 执行关联。
            occ3d_root=args.occ3d_root,
            surroundocc_root=args.surroundocc_root,
        )
    elif args.dataset == 'waymo':
        waymo_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            workers=args.workers,
            max_sweeps=args.max_sweeps)
