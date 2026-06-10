#!/bin/bash

# Single-node Qwen3-30B-A3B mock-data launch for testing PCT-aware CPU affinity.
#
# The process starts with helper/NCCL threads bound to non-PCT CPUs on the local
# NUMA node. Megatron then pins only the forward/backward dispatch threads to
# the rank-specific PCT CPU pair via --forward-backward-cpu-affinity.

#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH -t 00:30:00

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
PYTHON="${PYTHON:-python}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"
EXIT_INTERVAL="${EXIT_INTERVAL:-20}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
SAVE_DIR="${SAVE_DIR:-/tmp/qwen3-30b-pct-affinity-smoke}"

# Topology assumed by the test platform:
#   local_rank 0..3 -> NUMA 0, local_rank 4..7 -> NUMA 1
#   PCT CPUs per rank are (16 * local_rank), (16 * local_rank + 1)
# Override these masks if the node CPU numbering differs.
HELPER_CPUS_NODE0="${HELPER_CPUS_NODE0:-2-15,18-31,34-47,50-63}"
HELPER_CPUS_NODE1="${HELPER_CPUS_NODE1:-66-79,82-95,98-111,114-127}"

export PYTHON TRAIN_ITERS EXIT_INTERVAL SEQ_LENGTH GLOBAL_BATCH_SIZE MICRO_BATCH_SIZE
export SAVE_DIR HELPER_CPUS_NODE0 HELPER_CPUS_NODE1

srun \
    --ntasks-per-node="${GPUS_PER_NODE}" \
    --gpus-per-task=1 \
    --gpu-bind=single:1 \
    --cpu-bind=none \
    bash -lc '
        NUMA_NODE=$((SLURM_LOCALID / 4))
        PCT0=$((16 * SLURM_LOCALID))
        PCT1=$((16 * SLURM_LOCALID + 1))
        PCT_CPUS="${PCT0},${PCT1}"

        if [ "${NUMA_NODE}" -eq 0 ]; then
            HELPER_CPUS="${HELPER_CPUS_NODE0}"
        else
            HELPER_CPUS="${HELPER_CPUS_NODE1}"
        fi

        echo "local_rank=${SLURM_LOCALID} numa=${NUMA_NODE} pct=${PCT_CPUS} helpers=${HELPER_CPUS}"

        exec numactl --physcpubind="${HELPER_CPUS}" --membind="${NUMA_NODE}" \
            "${PYTHON}" pretrain_gpt.py \
                --mock-data \
                --forward-backward-cpu-affinity "${PCT_CPUS}" \
                --tensor-model-parallel-size 4 \
                --pipeline-model-parallel-size 1 \
                --expert-model-parallel-size 2 \
                --sequence-parallel \
                --use-mcore-models \
                --transformer-impl transformer_engine \
                --bf16 \
                --micro-batch-size "${MICRO_BATCH_SIZE}" \
                --global-batch-size "${GLOBAL_BATCH_SIZE}" \
                --train-iters "${TRAIN_ITERS}" \
                --exit-interval "${EXIT_INTERVAL}" \
                --seq-length "${SEQ_LENGTH}" \
                --max-position-embeddings 40960 \
                --num-layers 48 \
                --hidden-size 2048 \
                --ffn-hidden-size 6144 \
                --num-attention-heads 32 \
                --kv-channels 128 \
                --group-query-attention \
                --num-query-groups 4 \
                --normalization RMSNorm \
                --norm-epsilon 1e-6 \
                --position-embedding-type rope \
                --rotary-percent 1.0 \
                --rotary-base 1000000 \
                --use-rotary-position-embeddings \
                --swiglu \
                --disable-bias-linear \
                --untie-embeddings-and-output-weights \
                --qk-layernorm \
                --num-experts 128 \
                --moe-router-topk 8 \
                --moe-ffn-hidden-size 768 \
                --moe-aux-loss-coeff 0.001 \
                --moe-router-load-balancing-type aux_loss \
                --moe-token-dispatcher-type alltoall \
                --moe-layer-freq 1 \
                --attention-backend flash \
                --attention-dropout 0.0 \
                --hidden-dropout 0.0 \
                --no-masked-softmax-fusion \
                --attention-softmax-in-fp32 \
                --tokenizer-type HuggingFaceTokenizer \
                --tokenizer-model Qwen/Qwen3-30B-A3B \
                --vocab-size 151936 \
                --make-vocab-size-divisible-by 128 \
                --optimizer adam \
                --adam-beta1 0.9 \
                --adam-beta2 0.999 \
                --adam-eps 1e-8 \
                --lr 1e-6 \
                --min-lr 1e-7 \
                --lr-warmup-samples 0 \
                --clip-grad 1.0 \
                --weight-decay 0.01 \
                --save "${SAVE_DIR}" \
                --no-save-optim \
                --no-load-optim
    '
