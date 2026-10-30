import types

import torch
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
from mmcv.runner import (
    DistSamplerSeedHook,
    EpochBasedRunner,
    GradientCumulativeFp16OptimizerHook,
    Hook,
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
    build_runner,
)
from mmdet3d.runner import CustomEpochBasedRunner

from mmdet3d.utils import get_root_logger
from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor


def _force_fpn_lss_fp32(model, nan_debug_cfg, logger):
    """仅在独立 NaN 排查入口中让 decoder.neck(FPN_LSS) 使用 FP32。"""
    if not nan_debug_cfg.get("force_fpn_lss_fp32", False):
        return
    #* 获取FPN_LSS的实例
    neck = model.decoder["neck"]
    if type(neck).__name__ != "FPN_LSS":
        raise TypeError(
            "nan_debug.force_fpn_lss_fp32 only supports decoder.neck=FPN_LSS, "
            f"but got {type(neck).__name__}"
        )
    if getattr(neck, "_nan_debug_force_fp32", False):
        raise RuntimeError("FPN_LSS FP32 debug wrapper has already been installed.")

    #* 保存原来的forward
    original_forward = neck.forward
    def fp32_forward(_self, feats):
        #* 关闭局部 autocast，并把 FPN_LSS 的两个输入显式转为 FP32。
        # 这样只验证 decoder.neck 是否为 FP16 前向溢出的来源；其余模块仍按
        # 原混合精度路径运行，输出 FP32 也可直接交给后续 head 使用。
        output_dtype = feats[0].dtype
        with torch.cuda.amp.autocast(enabled=False):
            fp32_feats = [feat.float() if torch.is_tensor(feat) else feat
                          for feat in feats]
            output = original_forward(fp32_feats)
        # FPN_LSS 内部保持FP32计算，但返回前恢复原输入精度，使后续仍为FP16
        # 的检测头和占据头保持原混合精度路径。
        return output.to(dtype=output_dtype)
    #* 动态替换当前模型实例的forward：
    neck.forward = types.MethodType(fp32_forward, neck)
    neck._nan_debug_force_fp32 = True
    logger.warning(
        "NaN debug: only decoder.neck (%s) is forced to FP32.",
        type(neck).__name__,
    )


class _EvalBeforeResumedTrainingHook(Hook):
    """完整恢复后，在下一轮训练开始前补做一次验证。"""

    def __init__(self, eval_hook):
        self.eval_hook = eval_hook
        self.finished = False

    def before_train_epoch(self, runner):
        if self.finished:
            return
        runner.logger.info(
            "Resume pre-evaluation: evaluating epoch %d checkpoint before epoch %d training.",
            runner.epoch, runner.epoch + 1,
        )
        was_training = runner.model.training
        original_interval = self.eval_hook.interval
        try:
            #* _do_evaluate内部仍会检查evaluation.interval。这里临时设为1，
            # 保证“恢复后先评估”不受常规每N轮评估计划限制；评估后立即恢复。
            self.eval_hook.interval = 1
            self.eval_hook._do_evaluate(runner)
            self.finished = True
        finally:
            self.eval_hook.interval = original_interval
            # multi_gpu_test会切换model.eval()；补评估完成后恢复训练模式，
            # 随后的正式before_train_epoch和训练iteration按原流程执行。
            if was_training:
                runner.model.train()


def train_model_nan_debug(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
):
    logger = get_root_logger()

    #* 此包装发生在 DDP/DataParallel 和 Fp16OptimizerHook 包装模型之前，
    # 只作用于本次独立 NaN 排查训练，不修改原 FPN_LSS 类及普通训练入口。
    _force_fpn_lss_fp32(model, cfg.get("nan_debug", {}), logger)

    # 构建训练 DataLoader。cfg.seed 作为基础种子传入 MMDetection：框架会结合
    # worker id（分布式训练时还会结合 rank）为数据加载进程设置随机种子，并用于
    # 分布式采样器的数据打乱。这样相同 seed 下的数据顺序和随机增强尽量一致。
    # 这里只负责数据侧随机性；是否启用确定性 CUDA 算法由 tools/train.py 设置。
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            num_gpus=None if distributed else 1,
            dist=distributed,
            seed=cfg.seed,
        )
        for ds in dataset
    ]

    # put model on gpus
    find_unused_parameters = cfg.get("find_unused_parameters", False)
    # Sets the `find_unused_parameters` parameter in
    # torch.nn.parallel.DistributedDataParallel
    if distributed:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
    else:
        model = MMDataParallel(
            model.cuda(),
            device_ids=[0],
        )
    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.run_dir,
            logger=logger,
            meta={},
        ),
    )

    if hasattr(runner, "set_dataset"):
        runner.set_dataset(dataset)

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp
    #* ----------------------------------------------------------------------#
    #* (1) 顶层配置config决定启用混合精度还是FP32
    #* (2) @auto_fp16、@force_fp32 决定各方法的计算精度；
    #* (3) Fp16OptimizerHook + GradScaler 负责反向传播的loss缩放
    
    #* 训练开始时Fp16OptimizerHook.before_run() 会执行： wrap_fp16_model(runner.model),
    #* 遍历该模型,把具有fp16装饰器的模块设置为True
    # 精度总开关来自合并后的 cfg，而不是 tools/train.py 中手动转换模型类型。
    # cfg.fp16 为字典：启用 FP16/FP32 混合精度；为 None：不安装 FP16 Hook，走 FP32 路径。
    # FP32 设置：YAML 写 fp16: null，或训练命令追加 --fp16 None；不要传 False。
    # FP16 Hook 在训练开始时启用模型的 fp16_enabled，并负责梯度缩放/反缩放和更新。
    # 具体计算精度还受 auto_fp16、force_fp32 及 MMCV/PyTorch 版本影响，非所有层都用 FP16。
    # 此入口未提供纯 FP16 或 BF16 模式。
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        # 下面两种 Hook 都是 FP16/FP32 混合精度，区别仅在于是否“累积多个 batch 的梯度”。
        # Hook 是 Runner 在训练迭代结束时调用的控制逻辑，不是 AdamW/SGD 优化器本身。
        # 优化器已由上面的 build_optimizer(model, cfg.optimizer) 创建；Hook 负责何时
        # backward、梯度反缩放、裁剪、调用 optimizer.step() 和清理梯度。
        # 一个 iteration 指从 DataLoader 取一个 batch 并完成一次前向/反向；
        # 使用梯度累积后，一个 iteration 不一定对应一次参数更新。
        #
        # 配置例一：普通混合精度，每个 batch 尝试更新一次参数。
        # fp16:
        #   loss_scale: {init_scale: 256.0, growth_interval: 2000}
        # optimizer_config:
        #   grad_clip: {max_norm: 35, norm_type: 2}
        #
        # 配置例二：在上述 optimizer_config 下增加 cumulative_iters: 4，
        # 即每 4 个 batch 累积后尝试更新一次；它不放在 fp16 或 optimizer 节点下。
        # 命令行也可追加 --optimizer_config.cumulative_iters 4。
        # 这里检查的是“键是否存在”，所以设置为 1 仍选择累积 Hook（每步更新）；
        # 需要选择普通 Hook 时应删除该键，不能用 0、None 或 False 关闭累积。
        #
        # 参数展开：**cfg.optimizer_config 传入 grad_clip、cumulative_iters 等；
        # **fp16_cfg 传入 loss_scale 等混合精度参数；distributed 表示是否分布式。
        # 例如累积分支相当于 GradientCumulativeFp16OptimizerHook(
        #     cumulative_iters=4, grad_clip={...}, loss_scale={...}, distributed=True)。
        # 两个字典不能有同名参数；此分支直接调用类构造器，不要额外配置 type 字段。
        #*----------------------------------------------------------#
        #* GradientCumulativeFp16OptimizerHook: 显存仅能放小batch，但需要更大的有效batch，通过梯度积累的方式，用时间换取显存
        if "cumulative_iters" in cfg.optimizer_config:
            # 【混合精度 + 梯度累积】cumulative_iters 必须为正整数 K。
            # 每个小 batch 都做前向，并将 loss / K 后进行缩放和 backward，
            # 梯度累加到参数的 .grad 中，中间不清零，也不更新模型参数。
            # 累积到 K 次后：反缩放 -> 可选梯度裁剪 -> 尝试更新参数 -> 更新 scale
            # -> 清零梯度。当前 MMCV 会在整个训练的最后不足 K 步时按剩余步数归一化并更新。
            # 不需要在模型的 loss 中再手动除以 K，也不需要自己调用 backward/step。
            #
            # 例：每卡 batch=2、4 张卡、K=4，每次完整累积更新约覆盖 2*4*4=32 个样本。
            # 与直接把每卡 batch 改成 8 相比，累积不同时保留 4 个 batch 的激活，
            # 可降低实现相同有效 batch 所需的峰值显存；但不减少处理这些样本的计算量。
            # 这不完全等价于真正的大 batch：BatchNorm 仍按每次前向的 batch 计算统计量。
            # Runner 仍按原 iteration/epoch 计数，学习率计划不会自动按参数更新次数重设。
            # 梯度累积不是修复 NaN 的开关；当前项目 nan_debug 也明确不支持累积 Hook，
            # 使用此分支时应关闭 nan_debug.enabled，或定位问题时先移除 cumulative_iters。
            optimizer_config = GradientCumulativeFp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed
            )
        else:
            # FP16OptimizizerHOOK：
            # 【普通混合精度，不累积】每个 iteration：清零上次梯度 -> 缩放 loss
            # -> backward -> 反缩放梯度 -> 可选裁剪 -> 尝试更新参数 -> 更新 scale。
            # 每卡 batch=B、总卡数=W 时，一次更新通常使用 B*W 个样本。
            # 两种 Hook 都通过 GradScaler 检测梯度中的 Inf/NaN；发现溢出会跳过
            # 本次参数更新，动态缩放模式下还会降低 scale，因此这里称“尝试更新”。
            # loss_scale 只用于反向数值稳定性，不能修复前向已经出现的 Inf/NaN。
            #
            # 注意：下面这一行只是“创建并配置 Hook”，不会立刻执行反向传播。
            # runner.register_training_hooks(...) 注册它之后，Runner 会在每个训练
            # iteration 结束时调用 Fp16OptimizerHook.after_train_iter()。本环境中的
            # 真实实现位于：
            #   mmcv/runner/hooks/optimizer.py -> Fp16OptimizerHook.after_train_iter()
            #
            # 该方法的核心逻辑可简化成下面的伪代码（self.loss_scaler 是torch.cuda.amp.GradScaler）：
            #
            #   # 0. 清除上一轮残留的梯度
            #   runner.model.zero_grad()
            #   runner.optimizer.zero_grad()
            #   *--------------------------------------------------------#
            #   # 1. 取当前缩放系数 S，计算 scaled_loss = loss * S，再从
            #   #    scaled_loss 反向传播。梯度也会相应放大 S 倍，从而减少
            #   #    很小的 FP16 梯度因精度不足而直接变成 0（下溢）的情况。
            #   self.loss_scaler.scale(runner.outputs['loss']).backward() # 作用：把梯度乘以当前缩放系数
            #   *--------------------------------------------------------#
            #   # 2. 将 optimizer 管理的所有梯度除以 S，恢复为本来的数值大小；
            #   #    同时由 GradScaler 检查这些梯度中是否出现 Inf/NaN，并把
            #   #    “是否发现异常”的结果记录在 GradScaler 内部状态中。
            #   self.loss_scaler.unscale_(runner.optimizer) # 作用：把梯度除以当前缩放系数
            #   *--------------------------------------------------------#
            #   # 3. 如果配置了 grad_clip，必须在反缩放之后裁剪真实梯度。
            #   #    否则裁剪的是放大 S 倍后的梯度，裁剪阈值就失去原本含义。
            #   if self.grad_clip is not None:
            #       clip_grad_norm_(model_parameters, **self.grad_clip)
            #   *--------------------------------------------------------#
            #   # 4. 这里调用的是 GradScaler.step(optimizer)，不是直接调用
            #   #    optimizer.step()。GradScaler.step() 先查询第 2 步记录的
            #   #    Inf/NaN 检查结果，再决定是否放行真正的优化器更新：
            #   #    - 梯度全部有限：GradScaler 内部调用 runner.optimizer.step()；
            #   #      此时才由 AdamW/SGD 等真正的优化器更新模型参数。
            #   #    - 任一梯度为 Inf/NaN：GradScaler 不调用
            #   #      runner.optimizer.step()，直接跳过本次参数更新，避免异常
            #   #      梯度污染模型参数。
            #   #    调用关系可理解为：
            #   #      GradScaler.step(runner.optimizer)
            #   #          |-> 梯度正常：runner.optimizer.step()
            #   #          |-> 梯度异常：跳过 runner.optimizer.step()
            #   self.loss_scaler.step(runner.optimizer)
            #   *--------------------------------------------------------#
            #   # 5. update() 根据本轮检查结果更新下一轮使用的 S：
            #   #    - 发现 Inf/NaN：S *= backoff_factor（默认 0.5）；
            #   #    - 连续 growth_interval 轮无异常：S *= growth_factor（默认 2.0）；
            #   #    - 其他正常轮次：S 保持不变。
            #   self.loss_scaler.update(self._scale_update_param)
            #
            # “检测 Inf/NaN、决定是否跳过 optimizer.step、调整 S”的真正实现都在
            # PyTorch GradScaler 内部；DAOcc/MMCV 这里只按上述顺序调用其接口。
            # _scale_update_param 通常为 None，表示动态调整；只有配置静态浮点
            # loss_scale 时，MMCV 才会把固定值传给 update()。
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed
            )
    elif distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
        custom_hooks_config=cfg.get('custom_hooks', None)
    )
    if isinstance(runner, EpochBasedRunner):
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
        eval_hook_cls = DistEvalHook if distributed else EvalHook
        eval_hook = eval_hook_cls(val_dataloader, **eval_cfg)
        runner.register_hook(eval_hook)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    #* 该入口只在完整resume时生效：先使用恢复后的模型完成一次验证，再进入
    # 下一epoch训练。LOAD_FROM只加载权重，不具备“恢复后”的优化器/轮次语义。
    if cfg.get("eval_before_resume", False):
        if not cfg.resume_from:
            raise ValueError("eval_before_resume=True requires resume_from checkpoint.")
        if not validate:
            raise ValueError("eval_before_resume=True requires validate=True.")
        runner.register_hook(
            _EvalBeforeResumedTrainingHook(eval_hook), priority=9
        )

    #* NaN 专用训练入口始终安装独立诊断 Hook；原 train.py 完全不受影响。
    from mmdet3d.utils.nan_debug_persistent import install_persistent_nan_debug
    install_persistent_nan_debug(runner, cfg.get("nan_debug", {}))

    runner.run(data_loaders, [("train", 1)])
