"""完整模型检查：配置替换 → 构建 → 可选权重 → 真实数据 → 两个预测头。"""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import torch


def check_full_model(args):
    #* =================== 1. 使用与训练入口相同的注册、配置加载流程 ===================
    root = Path(args.daocc_root).expanduser().resolve() if args.daocc_root else Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from scripts.run_gauss_daocc import load_config
    from models.GaussDAOcc.head import GaussianOccHead
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_model
    from mmdet3d.utils import recursive_eval
    print('[完整模型 1/4] 加载配置：{}'.format(args.config), flush=True)
    cfg = Config(recursive_eval(load_config(Path(args.config).expanduser())))
    if cfg.model.heads.occ.type != 'GaussianOccHead':
        raise ValueError('配置没有将occ替换为GaussianOccHead')
    if cfg.model.type != 'BEVFusion':
        raise ValueError('此检查面向单帧BEVFusion高斯变体，不混入时序模型')
    model_cfg = deepcopy(cfg.model)
    # 检查流程不额外加载ResNet预训练文件；可选ckpt会在模型构建后统一加载。
    # 不传ckpt时完整网络随机初始化，仅验证可运行性，不验证精度。
    camera = model_cfg.get('encoders', {}).get('camera')
    if camera:
        camera['backbone']['init_cfg'] = None
        if 'pretrained' in camera['backbone']:
            camera['backbone']['pretrained'] = None
    # 隐藏ResNet逐参数初始化的INFO，保留WARNING/ERROR和本脚本的步骤提示。
    import logging
    from mmcv.utils import get_logger
    logger = get_logger('mmcv')
    previous_level = logger.level
    try:
        logger.setLevel(logging.WARNING)
        model = build_model(model_cfg)
    finally:
        logger.setLevel(previous_level)
    head = model.heads['occ']
    if not isinstance(head, GaussianOccHead):
        raise AssertionError('配置类型正确但实际占据头没有替换成功')
    if head.gaussian_aggregation.backend != 'cuda':
        raise ValueError('完整模型必须使用cuda聚合，不能用小网格参考实现替代')
    expected = tuple(head.gaussian_aggregation.shape)
    print('[完整模型 2/4] {} → {}；高斯数={}；占据网格={}'.format(
        type(model).__name__, type(head).__name__, head.gaussian_predictor.anchors.shape[1], expected), flush=True)
    print('保留模块：encoders={}，decoder={}，heads={}'.format(
        list(model.encoders), list(model.decoder), list(model.heads)), flush=True)
    if args.ckpt:
        # 原DAOcc缺失新高斯参数、含旧predicter参数属于预期；mmcv输出具体不匹配键。
        load_checkpoint(model, str(Path(args.ckpt).expanduser()), map_location='cpu', strict=False)
        print('已加载权重；若来自原DAOcc，新高斯预测层仍为随机初始化。', flush=True)
    else:
        print('未指定ckpt：使用随机参数，只检查流程，不代表预测精度。', flush=True)
    if args.build_only:
        print('通过：完整模型构建、配置替换、模块类型检查；未执行真实数据前向。', flush=True)
        return

    #* =================== 2. 加载真实验证数据，沿用原pipeline和坐标变换 ===================
    if not torch.cuda.is_available():
        raise RuntimeError('完整模型前向需要CUDA；可用--build-only先检查配置和构建')
    torch.cuda.set_device(args.device)
    from mmcv.parallel import collate, scatter
    from mmdet3d.datasets import build_dataset
    data_cfg = deepcopy(cfg.data.val)
    data_cfg['test_mode'] = True
    data_cfg.pop('samples_per_gpu', None)
    dataset = build_dataset(data_cfg)
    end = args.sample_index + args.num_samples
    if end > len(dataset):
        raise ValueError('请求样本区间[{}, {})超出验证集大小{}'.format(args.sample_index, end, len(dataset)))
    # 明确以FP32检查，不继承训练fp16配置。保留全部原预测分支。
    model = model.float().cuda(args.device).eval()
    print('[完整模型 3/4] 验证集{}帧，检查索引[{}, {})，FP32、batch=1。'.format(
        len(dataset), args.sample_index, end), flush=True)

    #* =================== 3. 用hook确认高斯头确实执行，不只看注册名称 ===================
    observed = {}
    def record_gaussians(module, inputs, output):
        observed['gaussians'] = tuple(output['means'].shape)
        if not all(torch.isfinite(value).all().item() for value in output.values()):
            raise AssertionError('高斯预测参数出现NaN/Inf')

    def record_head(module, inputs, output):
        observed['head'] = tuple(output.shape)
        if tuple(output.shape) != (1,) + expected + (18,):
            raise AssertionError('占据头输出维度错误：{}'.format(tuple(output.shape)))
        if not torch.isfinite(output).all().item():
            raise AssertionError('占据头输出出现NaN/Inf')
        error = (output.exp().sum(-1)-1).abs().max().item()
        if error > 1e-5:
            raise AssertionError('体素类别概率未归一化：最大误差{}'.format(error))

    handles = [head.gaussian_predictor.register_forward_hook(record_gaussians),
               head.register_forward_hook(record_head)]
    try:
        with torch.no_grad():
            for index in range(args.sample_index, end):
                observed.clear()
                sample = dataset[index]
                if sample is None:
                    raise RuntimeError('验证样本{}的pipeline返回None'.format(index))
                batch = scatter(collate([sample], samples_per_gpu=1), [args.device])[0]

                #* 调用完整model：相机、LiDAR、融合、Decoder、占据头、检测头全部执行。
                predictions = model(**batch)
                torch.cuda.synchronize(args.device)
                if 'gaussians' not in observed or 'head' not in observed: # observed：记录本次前向执行情况的字典，用于确认高斯预测层和占据头确实执行
                    raise AssertionError('完整前向未实际执行高斯预测和占据头')
                if not isinstance(predictions, list) or len(predictions) != 1:
                    raise AssertionError('完整模型返回值不是单帧预测列表')

                result = predictions[0]
                occ = np.asarray(result['occ_pred'])
                if occ.shape != expected or not np.issubdtype(occ.dtype, np.integer):
                    raise AssertionError('最终占据输出的形状或类别类型不符合原接口')
                if occ.min() < 0 or occ.max() >= 18:
                    raise AssertionError('最终占据类别超出0至17')
                if 'object' in model.heads:
                    boxes, scores, labels = result['boxes_3d'], result['scores_3d'], result['labels_3d']
                    if not (len(boxes) == len(scores) == len(labels)):
                        raise AssertionError('检测框、分数、类别数量不一致')
                    if not torch.isfinite(boxes.tensor).all() or not torch.isfinite(scores).all():
                        raise AssertionError('检测结果出现NaN/Inf')
                    print('检测框数量：{}'.format(len(scores)), flush=True)
                print('通过帧{}：高斯{} → 占据头{} → 最终{}，类别范围[{}, {}]'.format(
                    index, observed['gaussians'], observed['head'], occ.shape, occ.min(), occ.max()), flush=True)
                del predictions, result, batch, sample
    finally:
        for handle in handles:
            handle.remove()
    print('[完整模型 4/4] 通过：真实数据完整前向与输出检查。未进行训练反向或精度评估。', flush=True)
