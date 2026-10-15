#!/usr/bin/env bash
# Protocol jev-chunk-depth-v3.3 (docs/EXPERIMENT_PLAN.v3.3.zh-CN.md, section 7). Submit once from kudu-taught:
#     sbatch ~/jev-v33/code/scripts/a10/v33.sbatch
# The model is not on the cluster: after this job writes ~/jev-v33/run/<jobid>.ready, the operator streams it into
# the node's $TMPDIR with scripts/a10/push_model.py. Completed stages are skipped when the job is resubmitted.
#SBATCH --job-name=jev-v33
#SBATCH --account=pgt
#SBATCH --partition=gecko
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dcs/pg26/<cluster-user>/jev-v33/run/slurm-%j.out

set -euo pipefail
umask 077

WORK="$HOME/jev-v33"
CODE="$WORK/code"
SCR="${TMPDIR:-/tmp}/jev-v33-$SLURM_JOB_ID"
MODEL="$SCR/huginn-0125"
READY="$WORK/run/$SLURM_JOB_ID.ready"
log() { echo "$(date '+%F %T') $*"; }

mkdir -p "$SCR" "$MODEL" "$WORK/run"
cleanup() { rm -rf "$SCR"; rm -f "$READY"; log "scratch removed"; }
trap cleanup EXIT

printf '{"job": "%s", "node": "%s", "scratch": "%s", "model_dir": "%s"}\n' \
    "$SLURM_JOB_ID" "$(hostname -s)" "$SCR" "$MODEL" > "$READY"
log "ready $(cat "$READY")"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
df -h "$SCR" | tail -1

# Environment: uv in node-local scratch, versions pinned to the development machine.
export UV_CACHE_DIR="$SCR/uv-cache" PIP_CACHE_DIR="$SCR/pip-cache" HF_HOME="$SCR/hf"
python3.12 -m venv "$SCR/venv"
"$SCR/venv/bin/python" -m pip install --quiet --upgrade pip uv
"$SCR/venv/bin/uv" pip install --python "$SCR/venv/bin/python" --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu128 -r "$CODE/scripts/a10/requirements-v33.txt"
"$SCR/venv/bin/python" -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'gpus', torch.cuda.device_count())"
log "environment ready"

# Model: wait for the operator's upload (at most 60 minutes), then verify every file.
deadline=$(( $(date +%s) + 3600 ))
until [[ -f "$MODEL/.upload_done" ]]; do
    if (( $(date +%s) > deadline )); then log "model not received within 60 minutes"; exit 3; fi
    sleep 20
done
"$SCR/venv/bin/python" "$CODE/scripts/a10/verify_model.py" "$MODEL" "$CODE/scripts/a10/model_manifest.json"
log "model verified"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export JEV_HUGINN_DIR="$MODEL" JEV_SCRATCH="$SCR/work" PYTHONPATH="$CODE"
cd "$CODE"
"$SCR/venv/bin/python" -u -m experiments.chunk_depth_v33.driver --gpus 3 --nshards 3 --threads 14
log "all stages finished"
