#!/usr/bin/env bash
# 用法：bash scripts/train_occ3d.sh [--dry-run]
# 环境变量可覆盖下列默认值，例如 GPUS=0,1 MAX_EPOCHS=12 bash scripts/train_occ3d.sh
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 常用参数：batch size 和 workers 均为每张 GPU 的数量。
GPUS=${GPUS:-4,5,6,7}
BATCH_SIZE=${BATCH_SIZE:-4}
WORKERS=${WORKERS:-4}
MAX_EPOCHS=${MAX_EPOCHS:-6}  # 总轮数，不是恢复后额外增加的轮数。
DATA=${DATA:-/home/wangyushen/Data}
CONFIG=${CONFIG:-$PROJECT_ROOT/configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_inin.yaml}
# 相机/双模态训练默认加载 ResNet-50；LiDAR-only 设置 PRETRAIN=None。
# None 表示配置中不存在相机 encoder，脚本不会再注入任何
# model.encoders.camera.* 覆盖项。不能只留空，因为 ${PRETRAIN:-默认值}
# 会把空字符串重新替换为默认路径。
PRETRAIN=${PRETRAIN:-$DATA/Weights/daocc/pretrained/resnet50-0676ba61.pth}
CAMERA_PRETRAIN_ENABLED=1
if [[ $PRETRAIN == None ]]; then
    CAMERA_PRETRAIN_ENABLED=0
fi
RUN_DIR=${RUN_DIR:-$DATA/Outputs/daocc/w_mask/train}
PYTHON=${PYTHON:-python}  # 先激活 DAOcc 环境，或指定该环境的 Python 绝对路径。

# 保存/验证单位由配置by_epoch决定；未指定环境变量时沿用YAML中的interval。
LOG_INTERVAL=${LOG_INTERVAL:-50}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-}
EVAL_INTERVAL=${EVAL_INTERVAL:-}
MAX_ITERS=${MAX_ITERS:-}  # 仅新Runner支持；与epoch预算取较小值。
MAX_KEEP_CKPT=${MAX_KEEP_CKPT:-1}  # epoch模式分别保留；iter模式EMA内嵌，只保留一组文件。
LOAD_FROM=${LOAD_FROM:-None}  # 仅加载模型权重；自动恢复 checkpoint 时优先 resume_from。
EMA_RESUME=${EMA_RESUME:-None}  # 显式指定 EMA 优先于自动匹配同轮文件。

# 可选 NaN/Inf 记录，0 关闭、1 开启。默认关闭，不安装诊断 Hook。
# 示例：NAN_DEBUG=1 RUN_DIR=/新输出目录 bash scripts/train_occ3d.sh
# 已损坏的 checkpoint 不可用于定位首次异常；本脚本仍会自动恢复 RUN_DIR 中的权重。
# 精度设置示例（在 DAOcc 目录执行）：
#   FP32=1 bash scripts/train_occ3d.sh  -> 追加 --fp16 None，关闭 FP16 混合精度。
#   FP32=0 bash scripts/train_occ3d.sh  -> 沿用 YAML；默认 default.yaml 开启 FP16/FP32 混合精度。
# FP32=0 不会强制开启 FP16：若 YAML 中 fp16 为 null，仍使用 FP32。
# 若 YAML 已关闭而想重新开启，请把顶层 fp16 恢复为 default.yaml 中的 loss_scale 字典。
# 这里的“FP16”不是纯半精度；脚本没有纯 FP16/BF16 开关。
# 比较两种精度时请使用不同 RUN_DIR；脚本会自动恢复已有目录中的 checkpoint。
FP32=${FP32:-0}
NAN_DEBUG=${NAN_DEBUG:-0}
NAN_STOP_ON_ERROR=${NAN_STOP_ON_ERROR:-0}  # 1：各 rank 保存后停止；0：继续记录。
NAN_SAVE_BATCH=${NAN_SAVE_BATCH:-1}       # 0：仅保存 JSON 摘要；1：同时保存实际 batch。
NAN_DEBUG_DIR=${NAN_DEBUG_DIR:-$RUN_DIR/nan_debug}
# 开启后每步扫描状态，有额外开销；持续异常时完整 batch 可能占用大量磁盘。

# 分布式启动参数；默认仍为单机。多机时各节点使用相同 MASTER_ADDR/PORT。
DISTRIBUTED_MODULE=${DISTRIBUTED_MODULE:-torch.distributed.run}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-tools/dist_train.py}
LAUNCHER=${LAUNCHER:-pytorch}
STANDALONE=${STANDALONE:-1}  # 1 启用 --standalone；多机设为 0。
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
# NPROC_PER_NODE 在解析 GPU 列表后默认设为 GPU 数量，也可在外部覆盖。

DRY_RUN=0
if [[ $# -eq 1 && $1 == --dry-run ]]; then
    DRY_RUN=1
elif [[ $# -ne 0 ]]; then
    echo "用法：bash $0 [--dry-run]" >&2
    exit 2
fi
fail() { echo "错误：$*" >&2; exit 1; }
[[ $GPUS =~ ^[0-9]+(,[0-9]+)*$ ]] || fail 'GPUS 应为 0,1 这样的 GPU 编号列表'
[[ $BATCH_SIZE =~ ^[1-9][0-9]*$ ]] || fail 'BATCH_SIZE 必须为正整数'
[[ $WORKERS =~ ^(0|[1-9][0-9]*)$ ]] || fail 'WORKERS 必须为非负整数'
[[ $MAX_EPOCHS =~ ^[1-9][0-9]*$ ]] || fail 'MAX_EPOCHS 必须为正整数'
[[ -f $CONFIG ]] || fail "配置不存在：$CONFIG"
command -v "$PYTHON" >/dev/null || fail "找不到 Python：$PYTHON"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
declare -A SEEN_GPUS=()
for gpu in "${GPU_LIST[@]}"; do
    [[ ! ${SEEN_GPUS[$gpu]+yes} ]] || fail "GPU 编号重复：$gpu"
    SEEN_GPUS[$gpu]=1
done
NUM_GPUS=${#GPU_LIST[@]}
NPROC_PER_NODE=${NPROC_PER_NODE:-$NUM_GPUS}
for param in LOG_INTERVAL NNODES NPROC_PER_NODE MASTER_PORT; do
    [[ ${!param} =~ ^[1-9][0-9]*$ ]] || fail "$param 必须为正整数"
done
for param in CHECKPOINT_INTERVAL EVAL_INTERVAL MAX_ITERS; do
    [[ -z ${!param} || ${!param} =~ ^[1-9][0-9]*$ ]] || fail "$param 必须为空或正整数"
done
[[ $MAX_KEEP_CKPT =~ ^[1-9][0-9]*$ ]] || fail 'MAX_KEEP_CKPT 必须为正整数'
[[ $STANDALONE =~ ^[01]$ ]] || fail 'STANDALONE 必须为 0 或 1'
for param in FP32 NAN_DEBUG NAN_STOP_ON_ERROR NAN_SAVE_BATCH; do
    [[ ${!param} =~ ^[01]$ ]] || fail "$param 必须为 0 或 1"
done
[[ $NODE_RANK =~ ^(0|[1-9][0-9]*)$ ]] || fail 'NODE_RANK 必须为非负整数'
(( NPROC_PER_NODE <= NUM_GPUS )) || fail 'NPROC_PER_NODE 不能超过可见 GPU 数量'
(( NODE_RANK < NNODES )) || fail 'NODE_RANK 必须小于 NNODES'
(( MASTER_PORT <= 65535 )) || fail 'MASTER_PORT 必须小于等于 65535'
(( STANDALONE == 0 || NNODES == 1 )) || fail '多机训练请设置 STANDALONE=0'
[[ -f $TRAIN_SCRIPT ]] || fail "训练入口不存在：$TRAIN_SCRIPT"
[[ $LOAD_FROM == None || -f $LOAD_FROM ]] || fail "LOAD_FROM 不存在：$LOAD_FROM"
[[ $EMA_RESUME == None || -f $EMA_RESUME ]] || fail "EMA_RESUME 不存在：$EMA_RESUME"

# 显式RESUME_FROM优先，其次latest，再按数字寻找epoch/iter断点。
# 新旧模式请使用不同RUN_DIR；新Runner会拒绝缺少轮内元数据的旧断点。
RESUME_FROM=${RESUME_FROM:-}
shopt -s nullglob
if [[ -n $RESUME_FROM ]]; then
    [[ -f $RESUME_FROM ]] || fail "RESUME_FROM不存在：$RESUME_FROM"
elif [[ -f $RUN_DIR/latest.pth ]]; then
    RESUME_FROM=$RUN_DIR/latest.pth
else
    LAST_STEP=-1
    checkpoint_kind=''
    for checkpoint in "$RUN_DIR"/epoch_*.pth "$RUN_DIR"/iter_*.pth; do
        name=${checkpoint##*/}
        if [[ -f $checkpoint && $name =~ ^(epoch|iter)_([0-9]+)\.pth$ ]]; then
            kind=${BASH_REMATCH[1]}
            step=$((10#${BASH_REMATCH[2]}))
            [[ -z $checkpoint_kind || $checkpoint_kind == "$kind" ]] || fail '目录混有epoch/iter断点，请显式指定RESUME_FROM'
            checkpoint_kind=$kind
            if (( step > LAST_STEP )); then
                LAST_STEP=$step
                RESUME_FROM=$checkpoint
            fi
        fi
    done
fi

CMD=("$PYTHON" -m "$DISTRIBUTED_MODULE")
if (( STANDALONE )); then
    CMD+=(--standalone)
else
    CMD+=(--node_rank "$NODE_RANK" --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT")
fi
CMD+=(--nnodes="$NNODES" --nproc_per_node="$NPROC_PER_NODE"
    "$TRAIN_SCRIPT" "$CONFIG" --launcher "$LAUNCHER" --run-dir "$RUN_DIR"
    --data.samples_per_gpu "$BATCH_SIZE" --data.workers_per_gpu "$WORKERS"
    --max_epochs "$MAX_EPOCHS" --log_config.interval "$LOG_INTERVAL"
    --max_keep_ckpt "$MAX_KEEP_CKPT" --load_from "$LOAD_FROM")

[[ -z $CHECKPOINT_INTERVAL ]] || CMD+=(--checkpoint_config.interval "$CHECKPOINT_INTERVAL")
[[ -z $EVAL_INTERVAL ]] || CMD+=(--evaluation.interval "$EVAL_INTERVAL")
[[ -z $MAX_ITERS ]] || CMD+=(--runner.max_iters "$MAX_ITERS")

if [[ -n $RESUME_FROM ]]; then
    # 恢复时无需再次加载 ResNet 预训练文件；完整模型由 runner.resume 恢复。
    CMD+=(--resume_from "$RESUME_FROM")
    # Camera-only/双模态关闭初始化，避免恢复后再次读取ResNet文件；
    # LiDAR-only没有camera配置，因此不能注入这条嵌套覆盖项。
    if (( CAMERA_PRETRAIN_ENABLED )); then
        CMD+=(--model.encoders.camera.backbone.init_cfg None)
    fi
    echo "恢复训练：$RESUME_FROM"
    resolved=$(readlink -f -- "$RESUME_FROM")
    name=${resolved##*/}
    EMA_FROM=''
    if [[ $name =~ ^epoch_([0-9]+)\.pth$ ]]; then
        EMA_FROM="$(dirname -- "$resolved")/epoch_${BASH_REMATCH[1]}_ema.pth"
    fi
    if [[ $name =~ ^iter_([0-9]+)\.pth$ ]]; then
        [[ $EMA_RESUME == None ]] || fail 'iter断点已内嵌EMA，请勿另设EMA_RESUME'
        echo 'EMA：从迭代断点内部同步恢复'
    elif [[ $EMA_RESUME != None ]]; then
        echo "恢复指定 EMA：$EMA_RESUME"
    elif [[ -n $EMA_FROM && -f $EMA_FROM ]]; then
        EMA_RESUME=$EMA_FROM
        echo "恢复 EMA：$EMA_FROM"
    else
        echo '提示：没有找到同轮 EMA 文件，EMA 将从恢复后的模型重新初始化。' >&2
    fi
else
    # 若目录只有 EMA 或其他命名的权重，不静默开始新训练覆盖原有输出。
    OTHER_WEIGHTS=("$RUN_DIR"/*.pth)
    (( ${#OTHER_WEIGHTS[@]} == 0 )) || fail '目录中有权重，但没有 epoch_N.pth、iter_N.pth 或有效 latest.pth；无法自动恢复完整训练状态'
    if [[ $LOAD_FROM != None ]]; then
        CMD+=(--resume_from None)
        # 仅在配置包含相机encoder时覆盖其init_cfg；LiDAR-only跳过。
        if (( CAMERA_PRETRAIN_ENABLED )); then
            CMD+=(--model.encoders.camera.backbone.init_cfg None)
        fi
        echo "仅加载模型权重（不恢复优化器和轮数）：$LOAD_FROM"
    else
        CMD+=(--resume_from None)
        if (( CAMERA_PRETRAIN_ENABLED )); then
            [[ -f $PRETRAIN ]] || fail "预训练权重不存在：$PRETRAIN"
            CMD+=(--model.encoders.camera.backbone.init_cfg.checkpoint "$PRETRAIN")
            echo "从相机预训练权重开始：$PRETRAIN"
        else
            # LiDAR-only配置没有camera encoder，从随机初始化的LiDAR分支、
            # decoder和heads开始训练，不访问或注入ResNet预训练参数。
            echo '相机预训练：关闭（PRETRAIN=None，适用于LiDAR-only）'
        fi
    fi
fi

CMD+=(--custom_hooks.resume "$EMA_RESUME")

# Torchpack 将 None 解析为 Python 空值，训练入口据此禁用 FP16 Hook。
if (( FP32 )); then
    CMD+=(--fp16 None)
    echo '精度：FP32（已关闭 FP16 混合精度）'
else
    echo '精度：沿用 YAML 配置'
fi

echo "GPU：$GPUS；每卡 batch：$BATCH_SIZE；总 batch：$((NNODES * NPROC_PER_NODE * BATCH_SIZE))；总 epochs：$MAX_EPOCHS"
echo "输出目录：$RUN_DIR"
printf '执行命令：'; printf ' %q' "${CMD[@]}"; printf '\n'
if (( DRY_RUN )); then
    exit 0
fi
mkdir -p -- "$RUN_DIR"
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONWARNINGS=ignore
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${CMD[@]}"
