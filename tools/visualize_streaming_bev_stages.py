#!/usr/bin/env python3
"""加载完整流式权重，保存真实推理中六阶段 BEV 对照图（单卡，验证 BS=1）。

复用正式验证数据流；不重算 warp/refine/decoder，不更新参数或 BN。
用 --cfg-options 指定训练时的结构，例如 streaming.fuse_before_decoder=True。
输出为新建 run_* 目录，不覆盖既有结果；默认仅保存六宫格 PNG 图片；添加 --save-video 时按场景合成为 MP4 视频。
"""
import argparse
import math
import cv2
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import MethodType

# 复用现有脚本的配置解析、模型构建和正式流式验证入口。
from visualize_streaming_bev import (
    Config, DictAction, TorchpackConfig, recursive_eval, prepare_streaming_config,
    build_stream_val_dataloader, build_model, convert_sync_batchnorm,
    load_checkpoint, wrap_fp16_model, set_random_seed, scatter, safe_name,
    np, torch, plt,
)


class StageCapture:
    """仅旁路读取本次真实 forward 的中间结果，不改变其计算路径。"""
    def __init__(self, model):
        self.model = model
        self.original = model.process_history
        self.features = {}
        self.info = {}
        self.handles = []

    def save(self, name, tensor):
        if tensor is None:
            self.features[name] = None
            return
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            raise ValueError(f'{name}: expected tensor [B,C,X,Y]')
        # CPU 独立快照，防止后续原地操作、缓存覆盖改变可视化内容。
        self.features[name] = tensor.detach().float().cpu().clone()
        if not torch.isfinite(self.features[name]).all():
            raise RuntimeError(f'{name} contains NaN/Inf')
        self.info[name] = dict(shape=list(tensor.shape), dtype=str(tensor.dtype))

    def __enter__(self):
        observer = self
        def capture(this, current, history, valid):
            #* 1. process_history 的 history 已经过 warp；原始历史从真实缓存读取。
            observer.metas = deepcopy(this._stream_metas)
            observer.old_metas = deepcopy(this.bev_cache.metas)
            observer.valid = ([False] * current.shape[0] if valid is None
                              else valid.detach().cpu().tolist())
            observer.save('raw_history', this.bev_cache.bev)
            observer.save('warped_history', history)
            observer.save('current', current)
            alignment = this._history_alignment
            observer.mask = (None if alignment is None else
                             alignment['valid_mask'].detach().cpu().numpy().copy())
            #* 2. 保留模型原有首帧回退和 RefineNet 逻辑，hook 捕获实际细化输出。
            output = observer.original(current, history, valid)
            observer.save('fused', output)
            return output
        self.model.process_history = MethodType(capture, self.model)
        #* (11.17) 第三格捕获当前配置实际执行的历史更新模块。
        fusion = self.model.history_fusion
        updater = fusion.state_update if fusion.use_controlled_update else fusion.refine_net
        self.handles.append(updater.register_forward_hook(
            lambda module, inputs, output: self.save('refined_history', output)))
        self.handles.append(self.model.decoder['neck'].register_forward_hook(
            lambda module, inputs, output: self.save('decoded', output)))
        return self

    def clear(self):
        self.features.clear()
        self.info.clear()
        self.mask = None

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        del self.model.process_history


def save_frame(root, capture, step, bounds, shared_scale):
    before = capture.model.fuse_before_decoder
    keys = ['raw_history', 'warped_history', 'refined_history', 'current', 'fused', 'decoded']
    if set(keys) != set(capture.features):
        raise RuntimeError('未捕获全部六个阶段，请检查模型是否实际执行融合及 decoder')
    for slot, meta in enumerate(capture.metas):
        valid = capture.valid[slot]
        old = capture.old_metas[slot] if valid else None
        maps, arrays = {}, {}
        for key in keys:
            tensor = capture.features[key]
            #* 3. 首帧前两格没有历史；第三格仍展示实际 RefineNet(current) 回退结果。
            if tensor is None or (key in keys[:2] and not valid):
                maps[key] = None
            else:
                arrays[key] = tensor[slot].numpy()
                maps[key] = np.abs(arrays[key]).mean(axis=0).T
        folder = root / f'slot_{slot:02d}' / safe_name(meta['scene_token'])
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"step_{step:05d}_{safe_name(meta['token'])}"
        titles = ['1. Cached history (previous coordinates)',
                  '2. Ego-motion + augmentation warp',
                  '3. Observation-controlled history update' if capture.model.history_fusion.use_controlled_update else
                  '3. Refined history' if not isinstance(capture.model.history_fusion.refine_net, torch.nn.Identity)
                  else '3. Aligned history (RefineNet disabled)',
                  '4. Current BEV (before temporal fusion)',
                  '5. Fused BEV (next history cache)',
                  '6. Decoder output (final shared BEV)' if before else
                  '6. Decoder output (BEFORE temporal fusion)']
        if not valid:
            titles[2] += '\nFirst frame: current BEV used as history'
        shared = np.concatenate([x.ravel() for x in maps.values() if x is not None])
        common_limits = np.percentile(shared, [1, 99])
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
        limits = {}
        try:
            for ax, key, title in zip(axes.flat, keys, titles):
                heat = maps[key]
                ax.set_title(title)
                if heat is None:
                    ax.set_facecolor('#eeeeee')
                    ax.text(.5, .5, 'NO VALID HISTORY', transform=ax.transAxes, ha='center')
                else:
                    lo, hi = common_limits if shared_scale else np.percentile(heat, [1, 99])
                    hi = max(float(hi), float(lo) + 1e-8)
                    limits[key] = [float(lo), hi]
                    im = ax.imshow(heat, origin='lower', cmap='viridis', vmin=lo, vmax=hi,
                                   extent=[bounds[0], bounds[3], bounds[1], bounds[4]])
                    fig.colorbar(im, ax=ax, shrink=.8)
                ax.set(xlabel='LiDAR X (m)', ylabel='LiDAR Y (m)',
                       xlim=(bounds[0], bounds[3]), ylim=(bounds[1], bounds[4]), aspect='equal')
            dt = None if old is None else meta['timestamp'] - old['timestamp']
            order = 'warp -> refine -> fuse -> decoder' if before else 'decoder -> warp/refine/fuse'
            fig.suptitle(f"Step {step} | scene {meta['scene_token']} | history_valid={valid} | dt={dt}\n"
                         f"{order} | mean(abs(channels)) | "
                         f"{'shared' if shared_scale else 'per-panel'} 1-99% color scale\n"
                         'Raw history uses previous coordinates; all other panels use current coordinates. '
                         'Warp padding is shown as computed, not masked.')
            fig.savefig(folder / (stem + '.png'), dpi=140)
        finally:
            plt.close(fig)



def save_scene_videos(root, fps):
    """按推理 step 排序，逐张编码六宫格；每个场景独立生成视频。"""
    #* (11.17) 只读取本次 run 的图片，不跨场景拼接、不额外执行模型前向。
    # 固定播放帧率；视频长度由实际处理帧数决定，截断场景也保存已有帧。
    for folder in sorted(root.glob('slot_*/*')):
        frames = sorted(folder.glob('step_*.png'),
                        key=lambda path: int(path.name.split('_')[1]))
        if not frames:
            continue
        first = cv2.imread(str(frames[0]))
        if first is None:
            raise RuntimeError(f'无法读取图片：{frames[0]}')
        height, width = first.shape[:2]
        # MP4 编码使用偶数尺寸；仅补边，不拉伸特征图。
        size = (width + width % 2, height + height % 2)
        path = folder / 'bev_stages.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f'无法创建 MP4 视频：{path}，请检查 OpenCV 视频编码支持')
        try:
            for frame_path in frames:
                frame = cv2.imread(str(frame_path))
                if frame is None or frame.shape[:2] != (height, width):
                    raise RuntimeError(f'图片读取失败或尺寸不一致：{frame_path}')
                frame = cv2.copyMakeBorder(frame, 0, size[1] - height,
                                           0, size[0] - width, cv2.BORDER_CONSTANT,
                                           value=(255, 255, 255))
                writer.write(frame)
        finally:
            writer.release()
        print(f'视频：{path} | {len(frames)} 帧 | {fps:g} FPS | '
              f'{len(frames) / fps:.1f} 秒', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint', help='包含历史融合/RefineNet参数的完整训练权重')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--steps', type=int, default=24)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save-video', action='store_true', help='额外按场景合成MP4视频，默认关闭')
    parser.add_argument('--fps', type=float, default=2.0, help='启用--save-video时的视频播放帧率，默认2；不改变推理采样间隔')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    parser.add_argument('--shared-scale', action='store_true', help='六格共用色阶；默认各格独立显示空间结构')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, default={})
    args = parser.parse_args()
    if args.steps < 1 or args.workers < 0:
        parser.error('steps必须为正数，workers不能为负数')
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("fps必须为有限正数")
    raw = TorchpackConfig()
    raw.load(str(Path(args.config).expanduser().resolve()), recursive=True)
    cfg = Config(recursive_eval(raw))
    cfg.merge_from_dict(args.cfg_options)
    if not cfg.get('streaming', {}).get('enabled', False):
        parser.error('需要流式配置')
    cfg = prepare_streaming_config(cfg)
    cfg.data.workers_per_gpu = args.workers
    if not cfg.model.get('history_fusion') or not cfg.model.get('bev_warp'):
        parser.error('六阶段可视化需要启用 history_fusion 和 bev_warp')
    if not cfg.model.get('cache_enabled', True):
        parser.error('需要启用历史缓存')
    camera = cfg.model.encoders.get('camera')
    if camera:
        camera.backbone.init_cfg = None
        camera.backbone.pop('pretrained', None)
    torch.cuda.set_device(args.device)
    set_random_seed(cfg.get('seed', 0))
    loader = build_stream_val_dataloader(cfg, distributed=False)
    model = build_model(deepcopy(cfg.model))
    if cfg.get('sync_bn'):
        sync_cfg = cfg.sync_bn if isinstance(cfg.sync_bn, dict) else {}
        model = convert_sync_batchnorm(model, exclude=sync_cfg.get('exclude', []))
    #* 4. 严格加载，防止细化/融合层未加载却展示随机参数的输出。
    load_checkpoint(model, str(Path(args.checkpoint).expanduser()), map_location='cpu', strict=True)
    if args.precision == 'fp16':
        wrap_fp16_model(model)
    model.cuda(args.device).eval()
    if model.decoder is None:
        parser.error('当前模型没有decoder，无法展示六个阶段')
    root = Path(args.out_dir).expanduser() / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True)
    print(f'输出目录：{root}\n验证流式推理：每卡 BS=1；decoder前融合={model.fuse_before_decoder}', flush=True)
    #* (11.17) 默认仅保存图片，显式启用时才合成视频；不导出 JSON、配置或特征数组。
    with StageCapture(model) as capture, model.validation_stream(), torch.no_grad():
        for step, data in enumerate(loader):
            capture.clear()
            data = scatter(data, [args.device])[0]
            outputs = model(**data)
            save_frame(root, capture, step,
                       model.bev_warp.point_cloud_range, args.shared_scale)
            print(f'{step + 1}/{min(args.steps, len(loader))}: history_valid={capture.valid}', flush=True)
            del outputs, data
            if step + 1 >= args.steps:
                break
    if args.save_video:
        save_scene_videos(root, args.fps)


if __name__ == '__main__':
    main()
