# GaussDAOcc

保留 DAOcc 的相机、LiDAR、融合、Decoder、检测头；将 `BEVOCCHead2D`
替换为 `GaussianOccHead`。这是需要重新训练的研究变体，不是将原权重无损转换为高斯。

## 文件与执行过程

- `../../scripts/run_gauss_daocc.py`：显式继承配置、注册高斯头，调用原 `tools/train.py` 或 `tools/test.py`。
- `../../scripts/check_gauss_daocc_full.py`：完整模型构建、真实数据前向和输出检查。
- `head.py`：沿用原坐标变换和卷积，适配损失及 `get_occ`。
- `geometry.py`：从 BEV 预测中心、尺度、旋转、语义混合权重、17类非空语义。
- `aggregation.py`：批次循环调用本地CUDA内核，产生 `[B,X,Y,Z,18]` 的对数概率。
- `ops/localagg_prob`：从 GaussianFormer 复制的CUDA聚合源码，独立原地编译。
- `../../scripts/check_gauss_daocc.py`：按顺序检查几何、概率、梯度；可选比较CUDA与PyTorch参考实现。
- `../../configs/gauss_daocc/gauss_daocc.yaml`：在原单帧配置上只覆盖占据头。

高斯和体素查询点均在 Occ3D 标签坐标系中，继续使用原模型的 ego/LiDAR/数据增强
坐标转换。默认10000个高斯，输出200×200×16体素；这些是初始实验参数，并非已验证的最优设置。

## 1. 编译本地算子

在训练所用的 `daocc` 环境执行：

```bash
cd /home/wys/wangyushen/forks/CUDA-DAOcc/DAOcc/models/GaussDAOcc/ops/localagg_prob
python setup.py build_ext --inplace
```

扩展生成在本目录的 `local_aggregate_prob/` 内，代码使用包内相对导入，
不依赖 GaussianFormer 原目录，也不需要全局安装同名包。导入模型不会自动编译。

## 2. 检查

```bash
cd /home/wys/wangyushen/forks/CUDA-DAOcc
python DAOcc/scripts/check_gauss_daocc.py
python DAOcc/scripts/check_gauss_daocc.py --cuda
```

第一条仅使用CPU小网格；第二条需要可用GPU与已编译扩展，检查前向和反向数值。
`torch_reference` 有规模限制，仅用于检查，不能作为真实Occ3D推理的隐式回退。

## 3. 从原DAOcc权重初始化并训练

```bash
python DAOcc/scripts/run_gauss_daocc.py train \
  DAOcc/configs/gauss_daocc/gauss_daocc.yaml \
  --daocc-root /home/wys/wangyushen/forks/DAOcc \
  --load_from /home/wys/wangyushen/data/Weights/daocc/occ3d_nus_w_mask/epoch_6_ema.pth \
  --run-dir runs/gauss_daocc
```

原权重的 `heads.occ.predicter.*` 不再使用，新 `heads.occ.gaussian_predictor.*`
缺失属于预期；共享模块及占据头坐标变换、卷积仍可加载。新高斯头必须训练后再评价性能。
训练的随机增强、检测辅助损失、相机掩码等继承原配置，默认不加入时序模块。
可用 `--model.heads.occ.gaussian.anchor_shape '[50,50,4]'` 等原入口覆盖参数调整实验。

## 4. 使用训练后的高斯模型测试

```bash
python DAOcc/scripts/run_gauss_daocc.py test \
  DAOcc/configs/gauss_daocc/gauss_daocc.yaml \
  /实际路径/高斯模型.pth \
  --daocc-root /home/wys/wangyushen/forks/DAOcc \
  --out results/gauss_daocc/predictions.pkl
```

测试参数由原 `tools/test.py` 解释；预测结构与原占据输出兼容。
请通过此入口注册模型并合并配置，不要直接将含 `base_config` 的配置交给原 `tools/test.py`。
现有 `qat/evaluate-model.py`、ONNX导出和TensorRT/C++运行程序尚未为此新头增加专用适配。

## 计算语义与当前边界

- 上游 `localagg_prob` 的占据概率是 `1 - product(1 - gaussian_response)`。
  opacity只参与语义加权，不是关闭高斯占据的门控；本版本保留此语义。
- 合并占据概率与条件语义概率，再取log交给CE。free为第17类；保留掩码与类别权重。
- 尺度有上下限、中心限制在范围内，旋转由单位四元数残差预测；逆协方差直接在设备上构造。
- 原内核固定18类、float32、单样本；适配层逐样本调用支持训练batch>1。
- 保留上游邻域、排序、前后向公式。复制的Python包装添加设备上下文和前后向同步，
  保障原默认流内核的正确调用。同步会影响性能，此版本不能直接作为优化部署速度的结论。
- 上游占据乘积反向在高斯中心精确落在查询点等饱和情况下使用近似公式；
  CUDA数值检查采用非退化样本。实际训练仍应观察梯度与数值稳定性。
- 尚未实现TensorRT插件或独立C++聚合接入；现阶段交付的是PyTorch训练/测试变体。

原始来源及复制后改动见 `ops/localagg_prob/UPSTREAM.txt`。

## 完整模型流程检查

`--full-model` 使用与 `../../scripts/run_gauss_daocc.py` 相同的配置加载、注册和模型构建流程，默认取验证集第0帧：

```bash
python DAOcc/scripts/check_gauss_daocc.py --full-model \
  --daocc-root /home/wys/wangyushen/forks/DAOcc
```

可选 `--ckpt /实际路径/模型.pth`；不传时全模型随机初始化，不加载额外ResNet预训练文件。
原DAOcc权重只能初始化共享部分，高斯头仍需训练。此检查不输出性能指标。

检查会确认实际类型为 `GaussianOccHead`，并通过hook验证真实前向执行了高斯预测层与聚合；
检查各体素概率有限且归一化、输出为200×200×16的整数类别，以及原检测分支返回有效结构。

- `--sample-index 0 --num-samples 2`：连续检查验证集两帧，每次batch=1。
- `--full-model --build-only`：CPU构建完整模型，检查替换是否成功，不读取数据或执行前向。
- `--cuda --full-model`：先执行CUDA算子数值核对，通过后再执行完整模型前向。
- 单独 `--cuda` 保持原来的核心算子检查行为。

完整前向需要可用GPU、已编译的本地聚合扩展和完整验证数据；仅通过构建检查不等于GPU前向通过。
