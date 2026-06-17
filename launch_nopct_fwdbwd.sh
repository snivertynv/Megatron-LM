#!/bin/bash

# Single-node Qwen3-30B-A3B mock-data launch for testing PCT-aware CPU affinity.
#
# The process starts with helper/NCCL threads bound to non-PCT CPUs on the local
# NUMA node. Megatron then pins only the forward/backward dispatch threads to
# the rank-specific PCT CPU pair via --forward-backward-cpu-affinity.

#SBATCH -N 1
#SBATCH -p batch
#SBATCH -A sw_aidot
#SBATCH --ntasks-per-node=8
#SBATCH -t 00:30:00

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
PYTHON="${PYTHON:-python}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"
EXIT_INTERVAL="${EXIT_INTERVAL:-20}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
SAVE_DIR="${SAVE_DIR:-/lustre/fsw/sw_aidot/sniverty/Megatron-LM/qwen3-30b-pct-affinity-smoke}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
# Topology assumed by the test platform:
#   local_rank 0..3 -> NUMA 0, local_rank 4..7 -> NUMA 1
#   PCT CPUs per rank are (16 * local_rank), (16 * local_rank + 1)
# Override these masks if the node CPU numbering differs.
HELPER_CPUS_NODE0="${HELPER_CPUS_NODE0:-2-15,18-31,34-47,50-63}"
HELPER_CPUS_NODE1="${HELPER_CPUS_NODE1:-66-79,82-95,98-111,114-127}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/sw_aidot/sniverty/llmb_workloads_260500/images/nemo:26.06.rc2.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/lustre:/lustre}"
export PYTHON TRAIN_ITERS EXIT_INTERVAL SEQ_LENGTH GLOBAL_BATCH_SIZE MICRO_BATCH_SIZE
export EVAL_INTERVAL SAVE_DIR HELPER_CPUS_NODE0 HELPER_CPUS_NODE1

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-workdir="${PWD}" \
    --ntasks-per-node="${GPUS_PER_NODE}" \
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

        #exec numactl --physcpubind="${PCT_CPUS}" --membind="${NUMA_NODE}" \
        exec numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
            "${PYTHON}" pretrain_gpt.py \
                --mock-data \
                --tensor-model-parallel-size 1 \
                --pipeline-model-parallel-size 1 \
                --expert-model-parallel-size 8 \
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
		--log-interval 1 \
		--eval-interval "${EVAL_INTERVAL}" \
                --eval-iters 0 \
                --untie-embeddings-and-output-weights \
                --qk-layernorm \
                --num-experts 128 \
                --moe-router-topk 8 \
                --moe-ffn-hidden-size 768 \
                --moe-aux-loss-coeff 0.001 \
                --moe-router-load-balancing-type aux_loss \
                --moe-router-force-load-balancing \
                --moe-router-fusion \
                --moe-router-dtype fp32 \
                --moe-token-dispatcher-type flex \
		--moe-flex-dispatcher-backend hybridep \
                --moe-hybridep-num-sms 32 \
                --moe-grouped-gemm \
                --moe-permute-fusion \
                --cuda-graph-impl transformer_engine \
                --cuda-graph-scope moe_router moe_preprocess \
                --moe-layer-freq 1 \
                --attention-backend flash \
                --attention-dropout 0.0 \
                --hidden-dropout 0.0 \
                --tokenizer-type HuggingFaceTokenizer \
                --tokenizer-model Qwen/Qwen3-30B-A3B \
                --vocab-size 151936 \
                --make-vocab-size-divisible-by 128 \
                --optimizer adam \
                --adam-beta1 0.9 \
                --adam-beta2 0.95 \
                --adam-eps 1e-8 \
                --lr 3e-4 \
                --min-lr 3e-5 \
                --lr-decay-style cosine \
                --lr-warmup-iters 10 \
                --lr-warmup-samples 0 \
                --clip-grad 1.0 \
                --weight-decay 0.1 \
                --use-distributed-optimizer \
                --overlap-grad-reduce \
                --overlap-param-gather \
                --use-precision-aware-optimizer \
                --data-parallel-sharding-strategy optim_grads_params \
                --ddp-bucket-size 134217728 \
                --cross-entropy-loss-fusion \
                --recompute-modules core_attn \
                --wgrad-deferral-limit -1 \
                --no-save-optim \
                --no-load-optim
    '
