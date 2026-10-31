import argparse
import copy
import json
import os
import warnings

import mmcv
import torch
from torchpack.utils.config import configs
# from torchpack import distributed as dist # (10.31) 避免通过TorchPack的dist.init()初始化MPI
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.utils import recursive_eval

# for Ray-IoU metric
# RuntimeError: Too many open files. Communication with the workers is no longer possible. Please increase the limit using ulimit -n in the shell or change the sharing strategy by calling torch.multiprocessing.se t_sharing_strategy('file_system') at the beginning of your code
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--dist", action='store_true', help="distributed or not")
    # 原参数用于把全部预测写入一个文件，注释保留；现在使用 --out-dir 指定目录。
    # parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument("--out-dir", help="output root for scene/sample_token.pkl prediction files")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        default=['bbox'],
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC'
        '[CUSTOM] no practical effect in this project, just not None',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show-dir", help="directory where results will be saved")
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both specified, "
            "--options is deprecated in favor of --eval-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options
    return args



def prediction_paths(dataset, out_dir):
    # 推理前一次性解析所有帧的场景路径，避免每帧重复读取 nuScenes 数据表。
    infos = getattr(dataset, "data_infos", None)
    # nuScenes 的 sample.json 记录每帧所属场景，scene.json 提供可读场景名。
    # 直接读取数据表，不重复初始化 NuScenes，也不依赖 occupancy 标注目录的命名。
    sample_scenes = None
    version = getattr(dataset, "version", "")
    if version in ("v1.0-mini", "v1.0-trainval", "v1.0-test"):
        table_dir = os.path.join(dataset.dataset_root, version)
        with open(os.path.join(table_dir, "scene.json")) as f:
            scene_names = {scene["token"]: scene["name"] for scene in json.load(f)}
        with open(os.path.join(table_dir, "sample.json")) as f:
            sample_scenes = {
                sample["token"]: scene_names[sample["scene_token"]]
                for sample in json.load(f)
            }
        if infos is None or len(infos) != len(dataset):
            raise ValueError("nuScenes data_infos must match prediction count")

    filenames = []
    for index in range(len(dataset)):
        # nuScenes 使用 sample token 命名；其他数据集没有 token 时使用样本序号。
        token = infos[index].get("token") if infos is not None else None
        name = str(token) if token is not None else f"sample_{index:06d}"
        # 文件名必须是单个路径组件，避免意外写到指定目录之外。
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            raise ValueError(f"Invalid sample filename: {name!r}")
        # 原来所有帧平铺在输出根目录，注释保留。
        # filenames.append(name + ".pkl")
        if sample_scenes is not None:
            # 不允许未知帧静默落入错误场景；全部路径验证通过后才开始写文件。
            if token not in sample_scenes:
                raise ValueError(f"Sample token missing from nuScenes tables: {token!r}")
            scene_name = sample_scenes[token]
            if (not scene_name or scene_name in (".", "..")
                    or "/" in scene_name or "\\" in scene_name):
                raise ValueError(f"Invalid scene directory name: {scene_name!r}")
            filenames.append(os.path.join(scene_name, name + ".pkl"))
        else:
            # 其他数据集没有 nuScenes 场景表，维持原来的逐帧平铺保存方式。
            filenames.append(name + ".pkl")
    # 写入前检查重名，避免同一次评估中不同样本相互覆盖。
    if len(set(filenames)) != len(filenames):
        raise ValueError("Duplicate sample tokens: prediction filenames would collide")
    return [os.path.join(out_dir, filename) for filename in filenames]


def install_prediction_writer(model, data_loader, out_dir):
    # 测试加载器 shuffle=False，标准顺序/分布式采样器可重复遍历且顺序一致。
    # 按本进程 batch_sampler 的实际索引匹配结果，兼容单卡、多卡和 batch size > 1。
    paths = prediction_paths(data_loader.dataset, out_dir)
    batches = iter(data_loader.batch_sampler)
    rank, _ = get_dist_info()

    def save_after_forward(module, inputs, outputs):
        indices = next(batches)
        if len(indices) != len(outputs):
            raise ValueError("Prediction count does not match current batch indices")
        for index, prediction in zip(indices, outputs):
            output_path = paths[index]
            mmcv.mkdir_or_exist(os.path.dirname(output_path))
            # 先写本进程临时文件再原子替换，避免中断留下半个 pkl。
            # 分布式采样补齐可能重复最后几帧；不同进程临时文件互不冲突。
            temporary_path = f"{output_path}.rank{rank}.pid{os.getpid()}.tmp"
            try:
                mmcv.dump(prediction, temporary_path, file_format="pkl")
                os.replace(temporary_path, output_path)
            finally:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)

    # forward hook 在当前 batch 返回后、下一 batch 推理前立即保存每帧预测。
    # 原始 outputs 不作修改，推理函数继续收集它们用于完整数据集评估。
    return model.register_forward_hook(save_after_forward)



def main():
    args = parse_args()
    # if args.dist:
    #     dist.init()
    # 改为使用 MMCV 已经导入的 init_dist：
    if args.dist:
        init_dist(args.launcher, backend="nccl")
    
    torch.backends.cudnn.benchmark = True
    if args.dist:
        # torch.cuda.set_device(dist.local_rank())
        # 设置当前 GPU 的部分也改成 MMCV/PyTorch 的 local rank：
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        torch.cuda.set_device(0)

    # 保存操作改为检查输出目录，其余评估和可视化选项保持原逻辑。
    # assert args.out or args.eval or args.format_only or args.show or args.show_dir, (
    assert args.out_dir or args.eval or args.format_only or args.show or args.show_dir, (
        "Please specify at least one operation (save/eval/format/show the "
        'results / save the results) with the argument "--out-dir", "--eval"'
        ', "--format-only", "--show" or "--show-dir"'
    )

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    # 原检查针对单个 pkl 文件；新参数为目录，不再要求 .pkl 后缀。
    # if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
    #     raise ValueError("The output file must be a pkl file.")
    if args.out_dir:
        args.out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
        mmcv.mkdir_or_exist(args.out_dir)

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    print(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    # distributed = True
    if args.dist:
        distributed = True
        print('-' * 66)
        print('[CUSTOM] distributed mode is on!')
        print('-' * 66)
    else:
        distributed = False
        print('-' * 66)
        print('[CUSTOM] distributed mode is off!')
        print('-' * 66)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=None if distributed else 1,
        dist=distributed,
        shuffle=False,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        # 原调用保留在下方统一执行，先完成模型包装再安装逐帧保存钩子。
        # outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        # outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
    # 只有指定输出目录才安装保存钩子；没有指定时保持原推理行为。
    writer = install_prediction_writer(model, data_loader, args.out_dir) if args.out_dir else None
    try:
        if not distributed:
            outputs = single_gpu_test(model, data_loader)
        else:
            outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
    finally:
        # 推理结束或异常退出均移除钩子，已完成写入的预测文件保留。
        if writer is not None:
            writer.remove()
    if distributed:
        rank, _ = get_dist_info()
    else:
        rank = 0
    if rank == 0:
        # 原整批保存代码注释保留，改为每个样本保存一个 pkl。
        # if args.out:
        #     print(f"\nwriting results to {args.out}")
        #     mmcv.dump(outputs, args.out)
        # 原推理结束后统一保存的代码注释保留，避免再次写入全部结果。
        # if args.out_dir:
        #     save_sample_predictions(outputs, dataset, args.out_dir)
        if args.out_dir:
            print(f"Per-frame predictions saved during inference to {args.out_dir}")
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get("evaluation", {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                "interval",
                "tmpdir",
                "start",
                "gpu_collect",
                "save_best",
                "rule",
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == "__main__":
    main()
