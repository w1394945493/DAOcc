#!/usr/bin/env bash
#* NaN 专用分布式训练脚本；不调用普通 tools/dist_train.py 和 apis/train.py。
# 用法：bash scripts/train_occ3d_nan_debug.sh [--dry-run]
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 常用参数：batch size 和 workers 均为每张 GPU 的数量。
GPUS=${GPUS:-4,5,6,7}
BATCH_SIZE=${BATCH_SIZE:-4}
WORKERS=${WORKERS:-4}
MAX_EPOCHS=${MAX_EPOCHS:-6}  # 总轮数，不是恢复后额外增加的轮数。
DATA=${DATA:-/home/wangyushen/Data}
CONFIG=${CONFIG:-$PROJECT_ROOT/configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_inin_spconv2_voxel_fixed_s128_nan_debug.yaml}
PRETRAIN=${PRETRAIN:-$DATA/Weights/daocc/pretrained/resnet50-0676ba61.pth}
RUN_DIR=${RUN_DIR:-$DATA/Outputs/daocc/w_mask/train}
PYTHON=${PYTHON:-python}  # 先激活 DAOcc 环境，或指定该环境的 Python 绝对路径。

# 日志、保存与验证频率（单位沿用训练配置：日志为 iteration，保存/验证为 epoch）。
LOG_INTERVAL=${LOG_INTERVAL:-50}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-1}
EVAL_INTERVAL=${EVAL_INTERVAL:-1}
EVAL_BEFORE_RESUME=${EVAL_BEFORE_RESUME:-0}  # 1：完整恢复后先补评估，再训练下一轮。
MAX_KEEP_CKPT=${MAX_KEEP_CKPT:-1}  # 普通权重和 EMA 权重各自保留的数量。
LOAD_FROM=${LOAD_FROM:-None}  # 仅加载模型权重；自动恢复 checkpoint 时优先 resume_from。
EMA_RESUME=${EMA_RESUME:-None}  # 显式指定 EMA 优先于自动匹配同轮文件。

#* 首次异常后继续逐 iteration 记录；总 loss 连续达到该次数后停止所有 rank。
NAN_CONSECUTIVE=${NAN_CONSECUTIVE:-10}
NAN_MAX_PROBLEMS=${NAN_MAX_PROBLEMS:-50}
NAN_SAVE_BATCH=${NAN_SAVE_BATCH:-1}  # 仅保存首次异常的实际 batch。
NAN_DEBUG_DIR=${NAN_DEBUG_DIR:-$RUN_DIR/nan_debug}
# 已损坏的 checkpoint 不可用于定位首次异常；排查时建议使用新的 RUN_DIR。
# 精度设置示例（在 DAOcc 目录执行）：
#   FP32=1 bash scripts/train_occ3d.sh  -> 追加 --fp16 None，关闭 FP16 混合精度。
#   FP32=0 bash scripts/train_occ3d.sh  -> 沿用 YAML；默认 default.yaml 开启 FP16/FP32 混合精度。
# FP32=0 不会强制开启 FP16：若 YAML 中 fp16 为 null，仍使用 FP32。
# 若 YAML 已关闭而想重新开启，请把顶层 fp16 恢复为 default.yaml 中的 loss_scale 字典。
# 这里的“FP16”不是纯半精度；脚本没有纯 FP16/BF16 开关。
# 比较两种精度时请使用不同 RUN_DIR；脚本会自动恢复已有目录中的 checkpoint。
FP32=${FP32:-0}
# 每步扫描模型状态会带来额外开销，本脚本只用于定位 NaN。

# 分布式启动参数；默认仍为单机。多机时各节点使用相同 MASTER_ADDR/PORT。
DISTRIBUTED_MODULE=${DISTRIBUTED_MODULE:-torch.distributed.run}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-tools/dist_train_nan_debug.py}
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
for param in LOG_INTERVAL CHECKPOINT_INTERVAL EVAL_INTERVAL NNODES NPROC_PER_NODE MASTER_PORT NAN_CONSECUTIVE NAN_MAX_PROBLEMS; do
    [[ ${!param} =~ ^[1-9][0-9]*$ ]] || fail "$param 必须为正整数"
done
[[ $MAX_KEEP_CKPT =~ ^[1-9][0-9]*$ ]] || fail 'MAX_KEEP_CKPT 必须为正整数'
[[ $STANDALONE =~ ^[01]$ ]] || fail 'STANDALONE 必须为 0 或 1'
for param in FP32 NAN_SAVE_BATCH EVAL_BEFORE_RESUME; do
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

# 数字排序，避免 epoch_9 排在 epoch_10 后面；排除 *_ema.pth / best_*.pth。
RESUME_FROM=''
LAST_EPOCH=-1
shopt -s nullglob
for checkpoint in "$RUN_DIR"/epoch_*.pth; do
    name=${checkpoint##*/}
    if [[ -f $checkpoint && $name =~ ^epoch_([0-9]+)\.pth$ ]]; then
        epoch=$((10#${BASH_REMATCH[1]}))
        if (( epoch > LAST_EPOCH )); then
            LAST_EPOCH=$epoch
            RESUME_FROM=$checkpoint
        fi
    fi
done
# 兼容只保留 latest.pth 的目录；断开的软链接不作为有效 checkpoint。
if [[ -z $RESUME_FROM && -f $RUN_DIR/latest.pth ]]; then
    RESUME_FROM=$RUN_DIR/latest.pth
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
    --checkpoint_config.interval "$CHECKPOINT_INTERVAL" --evaluation.interval "$EVAL_INTERVAL"
    --max_keep_ckpt "$MAX_KEEP_CKPT" --load_from "$LOAD_FROM")

if [[ -n $RESUME_FROM ]]; then
    # 恢复时无需再次加载 ResNet 预训练文件；完整模型由 runner.resume 恢复。
    CMD+=(--resume_from "$RESUME_FROM" --model.encoders.camera.backbone.init_cfg None)
    echo "恢复训练：$RESUME_FROM"
    resolved=$(readlink -f -- "$RESUME_FROM")
    name=${resolved##*/}
    EMA_FROM=''
    if [[ $name =~ ^epoch_([0-9]+)\.pth$ ]]; then
        EMA_FROM="$(dirname -- "$resolved")/epoch_${BASH_REMATCH[1]}_ema.pth"
    fi
    if [[ $EMA_RESUME != None ]]; then
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
    (( ${#OTHER_WEIGHTS[@]} == 0 )) || fail '目录中有权重，但没有 epoch_N.pth 或有效 latest.pth；无法自动恢复完整训练状态'
    if [[ $LOAD_FROM != None ]]; then
        CMD+=(--resume_from None --model.encoders.camera.backbone.init_cfg None)
        echo "仅加载模型权重（不恢复优化器和轮数）：$LOAD_FROM"
    else
        [[ -f $PRETRAIN ]] || fail "预训练权重不存在：$PRETRAIN"
        CMD+=(--resume_from None --model.encoders.camera.backbone.init_cfg.checkpoint "$PRETRAIN")
        echo "从预训练权重开始：$PRETRAIN"
    fi
fi

if (( EVAL_BEFORE_RESUME )); then
    [[ -n $RESUME_FROM ]] || fail 'EVAL_BEFORE_RESUME=1 仅可用于检测到完整resume checkpoint的情况'
    # Torchpack把Python布尔字面量解析为bool；独立训练API据此注册一次性评估Hook。
    CMD+=(--eval_before_resume True)
    echo '恢复后流程：先评估恢复的checkpoint，再继续训练下一轮'
fi

CMD+=(--custom_hooks.resume "$EMA_RESUME")

# Torchpack 将 None 解析为 Python 空值，训练入口据此禁用 FP16 Hook。
if (( FP32 )); then
    CMD+=(--fp16 None)
    echo '精度：FP32（已关闭 FP16 混合精度）'
else
    echo '精度：沿用 YAML 配置'
fi

# Torchpack 用 literal_eval 解析 Python 布尔字面量。
nan_save=False
if (( NAN_SAVE_BATCH )); then nan_save=True; fi
CMD+=(--nan_debug.enabled True --nan_debug.out_dir "$NAN_DEBUG_DIR"
    --nan_debug.stop_after_consecutive_nan "$NAN_CONSECUTIVE"
    --nan_debug.max_problem_tensors "$NAN_MAX_PROBLEMS"
    --nan_debug.save_batch "$nan_save")
echo "NaN 诊断：连续 $NAN_CONSECUTIVE 步总 loss 异常后停止；目录：$NAN_DEBUG_DIR；保存首次 batch：$nan_save"

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
