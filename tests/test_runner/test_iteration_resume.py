"""CPU回归：真实CBGS/GroupSampler、断点恢复、EMA、按步评估及结束步保存。"""
import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import Hook
from mmdet.datasets import build_dataloader
from mmdet3d.datasets.dataset_wrappers import CBGSDataset
from mmdet3d.runner.resumable_epoch_based_runner import (
    ResumableEpochBasedRunner, prepare_resumable_datasets, resumable_loader,
)
from mmdet3d.core.hook.iteration_training import IterationEvalHook, IterationDistEvalHook


class ToyDataset(Dataset):
    CLASSES = ('a', 'b')
    resample = True

    def __init__(self):
        self.flag = np.zeros(12, dtype=np.uint8)
        self.data_infos = [dict(token=str(i)) for i in range(12)]

    def __len__(self):
        return 12

    def get_cat_ids(self, i):
        return [i % 2]

    def __getitem__(self, i):
        return dict(x=torch.tensor([i / 12.]), ids=i)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def evaluate(self, results, **kwargs):
        return dict(score=1.)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1, 1)
        self.trace = []

    def train_step(self, data, optimizer):
        assert self.training, '评估结束后未恢复训练模式'
        self.trace.extend(data['ids'].tolist())
        loss = (self.fc(data['x']) - 1).square().mean()
        return dict(loss=loss, log_vars={'loss': loss.item()}, num_samples=len(data['ids']))


class Interrupt(Hook):
    def __init__(self, step):
        self.step = step

    def after_train_iter(self, runner):
        if runner.iter + 1 == self.step:
            raise RuntimeError('simulate interruption')


def build_run(directory, resume=None, stop=None, workers=0, max_iters=11, keep=3, distributed=False, cbgs=True):
    np.random.seed((999 if resume else 23) + (torch.distributed.get_rank() if distributed else 0))
    torch.manual_seed(23)
    ds = CBGSDataset(ToyDataset()) if cbgs else ToyDataset()
    prepare_resumable_datasets([ds], resume)
    loader = resumable_loader(build_dataloader(
        ds, 2, workers, num_gpus=1, dist=distributed, seed=17), 17)
    model = (MMDistributedDataParallel(ToyModel(), device_ids=None) if distributed
             else MMDataParallel(ToyModel(), device_ids=[]))
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    runner = ResumableEpochBasedRunner(model=model, optimizer=optimizer,
        work_dir=str(directory), logger=logging.getLogger('iteration-test'),
        meta={"fp16": {"loss_scaler": {"scale": 128.}}}, max_epochs=4, max_iters=max_iters)
    runner.set_dataset([ds])
    runner.register_training_hooks(
        dict(policy='CosineAnnealing', by_epoch=False, min_lr_ratio=.1),
        dict(type='OptimizerHook'),
        dict(type='ResumableCheckpointHook', interval=3, max_keep_ckpts=keep),
        None, custom_hooks_config=dict(type='ResumableEMAHook', priority='ABOVE_NORMAL'))
    evaluated = []
    def fake_test(model, loader, **kwargs):
        evaluated.append(runner.iter + 1)
        model.eval()
        return []
    eval_class = IterationDistEvalHook if distributed else IterationEvalHook
    runner.register_hook(eval_class(DataLoader(ToyDataset()),
        interval=3, by_epoch=False), priority='LOW')
    if stop:
        runner.register_hook(Interrupt(stop), priority='VERY_LOW')
    if resume:
        runner.resume(str(resume))
    target = 'mmdet.apis.multi_gpu_test' if distributed else 'mmdet.apis.single_gpu_test'
    with patch(target, side_effect=fake_test):
        try:
            runner.run([loader], [('train', 1)])
        except RuntimeError as error:
            if str(error) != 'simulate interruption':
                raise
    return runner, evaluated


def test_resume(tmp_path):
    for workers in (0, 2):
        root = tmp_path / str(workers)
        full, evaluations = build_run(root / 'full', workers=workers)
        assert evaluations == [3, 6, 9, 11]
        assert full.iter == 11 and full.epoch == 1
        for stop in (3, 6):  # 轮中和轮末两种断点
            partial, _ = build_run(root / str(stop), stop=stop, workers=workers)
            checkpoint = root / str(stop) / f'iter_{stop}.pth'
            resumed, _ = build_run(root / str(stop), resume=checkpoint, workers=workers)
            assert partial.model.module.trace + resumed.model.module.trace == full.model.module.trace
            for key, value in full.model.module.state_dict().items():
                torch.testing.assert_close(value, resumed.model.module.state_dict()[key], rtol=0, atol=0)
                torch.testing.assert_close(full.ema_model.ema.state_dict()[key],
                                           resumed.ema_model.ema.state_dict()[key], rtol=0, atol=0)
            assert resumed.ema_model.updates == full.ema_model.updates == 11
            assert resumed.current_lr() == full.current_lr()
        short, evaluations = build_run(root / 'short', max_iters=2, keep=1)
        assert short.iter == 2 and short.epoch == 0 and evaluations == [2]
        checkpoint = torch.load(root / 'short' / 'latest.pth', map_location='cpu')
        assert checkpoint['meta']['resume_state']['inner_iter_completed'] == 2
        assert checkpoint['meta']['iter'] == 2
        assert checkpoint['meta']['fp16']['loss_scaler']['scale'] == 128.
        assert len(list((root / 'short').glob('iter_*.pth'))) == 1


def test_plain_and_guard(tmp_path):
    full, _ = build_run(tmp_path / 'plain_full', cbgs=False)
    partial, _ = build_run(tmp_path / 'plain_resume', stop=3, cbgs=False)
    path = tmp_path / 'plain_resume' / 'iter_3.pth'
    state = torch.load(path, map_location='cpu')
    resumed, _ = build_run(tmp_path / 'plain_resume', resume=path, cbgs=False)
    assert partial.model.module.trace + resumed.model.module.trace == full.model.module.trace
    state['meta']['resume_state']['loader']['batch_size'] = 99
    invalid = tmp_path / 'changed_batch.pth'
    torch.save(state, invalid)
    try:
        build_run(tmp_path / 'invalid', resume=invalid, cbgs=False)
    except ValueError as error:
        assert '恢复条件改变' in str(error)
    else:
        raise AssertionError('batch改变时应拒绝轮内恢复')


def distributed_check(directory):
    # torchrun启动两个CPU rank，验证DDP分片和rank0写入/所有rank恢复。
    torch.distributed.init_process_group('gloo')
    root = Path(directory)
    full, _ = build_run(root / 'full', distributed=True)
    partial, _ = build_run(root / 'resume', stop=3, distributed=True)
    torch.distributed.barrier()
    resumed, _ = build_run(root / 'resume', resume=root / 'resume' / 'iter_3.pth', distributed=True)
    assert partial.model.module.trace + resumed.model.module.trace == full.model.module.trace
    for key, value in full.model.module.state_dict().items():
        torch.testing.assert_close(value, resumed.model.module.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(full.ema_model.ema.state_dict()[key],
                                   resumed.ema_model.ema.state_dict()[key], rtol=0, atol=0)
    print(f'PASS: rank {torch.distributed.get_rank()} DDP/CBGS/EMA恢复')
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    import sys
    import tempfile
    if len(sys.argv) == 2:
        distributed_check(sys.argv[1])
    else:
        with tempfile.TemporaryDirectory() as directory:
            test_resume(Path(directory))
            test_plain_and_guard(Path(directory))
        print('PASS: CBGS恢复、worker=0/2、轮中/轮末续训、EMA、按步评估和不足一轮预算')
