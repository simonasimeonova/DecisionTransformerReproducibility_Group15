#!/usr/bin/env bash
set -euo pipefail
unset PYTHONHOME PYTHONPATH

if [ "$#" -lt 4 ]; then
    echo "Usage: train.sh ENV DATASET K SEED [experiment.py options]" >&2
    exit 2
fi
env_name=$1
dataset=$2
K=$3
seed=$4
shift 4

storage=${DT_STORAGE_DIR:-/home/renku/work/DecisionTransformerReproducibility_Group15/training-storage}
dataset_file="$storage/datasets/$env_name-$dataset-v2.pkl"
[ -f "$dataset_file" ] || { echo "Missing dataset: $dataset_file" >&2; exit 1; }

work=/home/renku/work
mkdir -p "$work/results"
run_dir=$(mktemp -d "$work/results/${env_name}_K${K}_seed${seed}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
ln -s "$storage/datasets" "$run_dir/data"
cd "$run_dir"
printf '%s\n' "env=$env_name dataset=$dataset K=$K seed=$seed extra_args=$*" > arguments.txt
cp /opt/source-commit.txt .

status=0
python -u /opt/decision-transformer/gym/experiment.py \
    --env "$env_name" --dataset "$dataset" --K "$K" --seed "$seed" \
    --device cuda "$@" 2>&1 | tee train.log || status=$?

archive="$work/train-$(basename "$run_dir").tar.gz"
tar --exclude='*/data' -czf "$archive" -C "$work" "results/$(basename "$run_dir")"
cp "$archive" "$storage/"
sync
echo "Saved $storage/$(basename "$archive")"
sleep 300  # Give SWITCHdrive time to upload before the job ends.
exit "$status"
