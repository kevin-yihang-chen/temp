#!/usr/bin/env bash
#SBATCH --job-name=be-infovqa-decar-smoke
#SBATCH --partition=q-h800
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h800:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm-infovqa-decar-smoke-%j.out
#SBATCH --mail-user=yihangc@connect.hku.hk
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,STAGE_OUT,INVALID_DEPEND

set -euo pipefail

repo_root=/userhome/cs3/yihangc/Documents/beyond-entropy
python_bin=/userhome/cs3/yihangc/anaconda3/envs/qwen-vl/bin/python

cd "${repo_root}"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "DECAR torch smoke start: $(date --iso-8601=seconds)"
echo "Slurm job: ${SLURM_JOB_ID:-local}"
sha256sum \
  src/beyond_entropy/qwen_backend.py \
  src/beyond_entropy/infographicvqa_decar.py \
  scripts/smoke_infographicvqa_decar_torch.py
"${python_bin}" scripts/smoke_infographicvqa_decar_torch.py \
  --device cuda:0 \
  --nested-oof
echo "DECAR torch smoke end: $(date --iso-8601=seconds)"
