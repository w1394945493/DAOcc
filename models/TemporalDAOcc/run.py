"""注册时序模块、显式合成配置，并复用原 DAOcc 训练/测试入口。"""
import argparse
import importlib.util
from pathlib import Path
import runpy
import sys
import tempfile


def plain(value):
    # Torchpack 的嵌套 Config 需递归转成普通字典才能安全保存 YAML。
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'test', 'stream'])
    parser.add_argument('config')
    parser.add_argument('--daocc-root', help='可选择已编译 CUDA 扩展的 DAOcc 源码目录')
    parser.add_argument('--checkpoint', help='stream 模式权重；test 权重仍作为位置参数')
    parser.add_argument('--out', help='stream 输出全部预测的 pkl 文件')
    args, remaining = parser.parse_known_args()
    local_root = Path(__file__).resolve().parents[2]
    root = Path(args.daocc_root).resolve() if args.daocc_root else local_root
    sys.path.insert(0, str(root))
    # * 模型代码始终来自本目录，底层 mmdet3d 可使用已编译的另一份源码。
    spec = importlib.util.spec_from_file_location(
        'temporal_daocc_plugin', Path(__file__).with_name('__init__.py'),
        submodule_search_locations=[str(Path(__file__).parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    import yaml
    from torchpack.utils.config import Config
    path = Path(args.config).resolve()
    overlay = yaml.safe_load(path.read_text())
    base = (path.parent / overlay.pop('base_config')).resolve()
    config = Config()
    config.load(str(base), recursive=True)
    config.update(overlay)
    if args.action == 'stream':
        if not args.checkpoint or not args.out:
            parser.error('stream 模式必须指定 --checkpoint 和 --out')
        config.update(remaining)
        stream(config, args)
        return
    # 完整配置放到临时目录，原入口不会再次继承 nuscenes 的隐式默认配置。
    with tempfile.TemporaryDirectory(prefix='temporal-daocc-') as directory:
        effective = Path(directory) / 'effective.yaml'
        effective.write_text(yaml.safe_dump(plain(config), sort_keys=False))
        extras = remaining
        if args.out:
            extras += ['--out', args.out]
        if args.checkpoint:
            parser.error('train 用 --load_from；test 使用位置参数指定权重')
        sys.argv = [str(root / 'tools' / (args.action+'.py')), str(effective)] + extras
        runpy.run_path(sys.argv[0], run_name='__main__')


def stream(config, args):
    import mmcv
    import torch
    from tqdm import tqdm
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet3d.models import build_model
    from mmdet3d.datasets import build_dataset
    from mmdet3d.utils import recursive_eval
    cfg = mmcv.Config(recursive_eval(config))
    data_cfg = cfg.data.val.copy()
    data_cfg.update(temporal_mode='stream', test_mode=True)
    dataset = build_dataset(data_cfg)
    model = build_model(cfg.model)
    if cfg.get('fp16') is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cuda().eval()
    results, state = [None]*len(dataset), None
    for index in tqdm(dataset.stream_indices, desc='按场景顺序推理'):
        # * 数据按场景、时间排序；输出放回原索引，便于与原评估代码对齐。
        data = scatter(collate([dataset[index]], samples_per_gpu=1), [0])[0]
        predictions, state = model.forward_stream(data, state)
        results[index] = predictions[0]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    mmcv.dump(results, args.out)
    print('已保存预测：', args.out)


if __name__ == '__main__':
    main()
