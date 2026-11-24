#!/usr/bin/env bash
set -euo pipefail
unset PYTHONHOME PYTHONPATH

if [ "$#" -lt 4 ]; then
    echo "Usage: train.sh ENV DATASET K SEED[,SEED...] [experiment.py options]" >&2
    exit 2
fi
env_name=$1
dataset=$2
K=$3
[[ "$4" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "Seeds must be comma-separated numbers" >&2; exit 2; }
IFS=',' read -r -a seeds <<< "$4"
shift 4
seen=""
for seed in "${seeds[@]}"; do
    [[ " $seen " == *" $seed "* ]] && { echo "Duplicate seed: $seed" >&2; exit 2; }
    seen="$seen $seed"
done

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
if [ -f "$repo_dir/gym/experiment.py" ]; then
    experiment_py="$repo_dir/gym/experiment.py"
    source_commit=$(git -C "$repo_dir" rev-parse HEAD)
else
    experiment_py=/opt/decision-transformer/gym/experiment.py
    source_commit=$(cat /opt/source-commit.txt)
fi

storage=${DT_STORAGE_DIR:-/home/renku/work/DecisionTransformerReproducibility_Group15/training-storage}
dataset_file="$storage/datasets/$env_name-$dataset-v2.pkl"
[ -f "$dataset_file" ] || { echo "Missing dataset: $dataset_file" >&2; exit 1; }

work=/home/renku/work
mkdir -p "$work/results"
status=0
for seed in "${seeds[@]}"; do
    run_dir=$(mktemp -d "$work/results/${env_name}_K${K}_seed${seed}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
    ln -s "$storage/datasets" "$run_dir/data"
    cd "$run_dir"
    printf '%s\n' "env=$env_name dataset=$dataset K=$K seed=$seed extra_args=$*" > arguments.txt
    printf '%s\n' "$source_commit" > source-commit.txt

    python -u "$experiment_py" \
        --env "$env_name" --dataset "$dataset" --K "$K" --seed "$seed" \
        --device cuda "$@" 2>&1 | tee train.log || status=$?

    archive="$work/train-$(basename "$run_dir").tar.gz"
    tar --exclude='*/data' -czf "$archive" -C "$work" "results/$(basename "$run_dir")"
    cp "$archive" "$storage/"
    echo "Saved $storage/$(basename "$archive")"
    [ "$status" -eq 0 ] || break
done
sync
sleep 300  # Give SWITCHdrive time to upload before the job ends.
exit "$status"
