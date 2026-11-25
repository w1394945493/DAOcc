"""独立的流式训练 NaN 记录脚本；正常 dist_train.py 保持不变。"""

#* (11.25) 在原流式训练配置、序列采样和缓存流程上启用持久化异常记录。
import argparse
import os
import random
import time

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import init_dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis.train_stream import prepare_streaming_config
from mmdet3d.apis.train_stream_nan_debug import train_stream_model_nan_debug
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none', help='job launcher')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)
    cfg = Config(recursive_eval(configs), filename=args.config)
    if not cfg.get('streaming', {}).get('enabled', False):
        raise ValueError('dist_train_stream_nan_debug.py requires streaming.enabled=True')
    cfg = prepare_streaming_config(cfg)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    if args.launcher == 'none':
        distributed = False
        torch.cuda.set_device(0)
    else:
        distributed = True
        init_dist(args.launcher, backend='nccl')

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir
    os.makedirs(cfg.run_dir, exist_ok=True)
    cfg.dump(os.path.join(cfg.run_dir, 'configs.yaml'))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logger = get_root_logger(
        log_file=os.path.join(cfg.run_dir, f'{timestamp}.log'))
    logger.info('Config:\n%s', cfg.pretty_text)
    logger.info('Persistent stream NaN debug entry; distributed=%s', distributed)

    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    datasets = [build_dataset(cfg.data.train)]
    model = build_model(cfg.model)
    model.init_weights()
    if cfg.get('sync_bn', None):
        if not isinstance(cfg.sync_bn, dict):
            cfg.sync_bn = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg.sync_bn['exclude'])

    logger.info('Model:\n%s', model)
    train_stream_model_nan_debug(
        model, datasets, cfg, distributed=distributed,
        validate=True, timestamp=timestamp)


if __name__ == '__main__':
    main()
