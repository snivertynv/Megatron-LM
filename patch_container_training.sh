#!/bin/bash
# Applies PCT CPU affinity patches into the container.
# - Copies thread_affinity.py into megatron.training (new standalone file)
# - Patches megatron.bridge.training.pretrain to call initialize_forward_backward_cpu_affinity after setup
# - Patches megatron.bridge.training.train to wrap forward_backward_func with the affinity context
MCORE_TRAINING=$(python3 -c "import megatron.training, os; print(os.path.dirname(megatron.training.__file__))")
BRIDGE_TRAINING=$(python3 -c "import megatron.bridge.training.pretrain, os; print(os.path.dirname(megatron.bridge.training.pretrain.__file__))")
PATCH_SRC="$(dirname "$0")/megatron/training"

echo "[patch_container_training] megatron.training: $MCORE_TRAINING"
echo "[patch_container_training] megatron.bridge.training: $BRIDGE_TRAINING"

# 1. Copy thread_affinity.py (new standalone file, no incompatible deps)
cp -f "$PATCH_SRC/thread_affinity.py" "$MCORE_TRAINING/"
echo "[patch_container_training] thread_affinity.py copied"

# 2. Patch Bridge pretrain.py: call initialize_forward_backward_cpu_affinity after setup()
python3 - "$BRIDGE_TRAINING/pretrain.py" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path, 'r') as f:
    content = f.read()

if 'initialize_forward_backward_cpu_affinity' in content:
    print("[patch_container_training] pretrain.py: already patched")
    sys.exit(0)

anchor = "    pg_collection = setup_output.pg_collection"
if anchor not in content:
    print("[patch_container_training] pretrain.py: anchor not found", file=sys.stderr)
    sys.exit(1)

insertion = """
    _cpu_affinity = __import__('os').environ.get('MEGATRON_FWD_BWD_CPU_AFFINITY')
    if _cpu_affinity is not None:
        from megatron.training.thread_affinity import initialize_forward_backward_cpu_affinity
        initialize_forward_backward_cpu_affinity(_cpu_affinity)"""

content = content.replace(anchor, anchor + insertion, 1)
with open(path, 'w') as f:
    f.write(content)
print("[patch_container_training] pretrain.py: patched")
PYEOF

# 3. Patch Bridge train.py: wrap forward_backward_func call with per-step affinity context
python3 - "$BRIDGE_TRAINING/train.py" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path, 'r') as f:
    content = f.read()

if 'forward_backward_cpu_affinity' in content:
    print("[patch_container_training] train.py: already patched")
    sys.exit(0)

anchor = "        # Forward pass.\n        losses_reduced = forward_backward_func(\n            forward_step_func=forward_step_func,"
if anchor not in content:
    print("[patch_container_training] train.py: anchor not found", file=sys.stderr)
    sys.exit(1)

replacement = """        # Forward pass.
        from megatron.training.thread_affinity import forward_backward_cpu_affinity as _fwd_bwd_affinity
        with _fwd_bwd_affinity():
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step_func,"""

content = content.replace(anchor, replacement, 1)
with open(path, 'w') as f:
    f.write(content)
print("[patch_container_training] train.py: patched")
PYEOF

echo "[patch_container_training] done"
