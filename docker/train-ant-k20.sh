#!/usr/bin/env bash
set -euo pipefail
unset PYTHONHOME PYTHONPATH

# Keep the training code's default seed (42); explicit --seed overrides it.
# Environment/model/context remain fixed for member C.
args=(--max_iters 10 --num_steps_per_iter 10000 --num_eval_episodes 100
      "$@" --env ant --dataset medium-replay --model_type dt --K 20 --device cuda)
seed=$(python - "${args[@]}" <<'PY'
import argparse
import sys
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--seed', type=int, default=42)
parsed, _ = parser.parse_known_args(sys.argv[1:])
print(parsed.seed)
PY
)
work_dir="${DT_WORK_DIR:-/home/renku/work}"
data_dir="${DT_DATA_DIR:-$work_dir/datasets}"
mkdir -p "$work_dir/results" "$data_dir"
run_dir="$(mktemp -d "$work_dir/results/ant_K20_seed${seed}_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
cd "$run_dir"
exec > >(tee -a train.log) 2>&1
echo "Output directory: $run_dir"
ln -s "$data_dir" data

python - <<'PY'
import torch
print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda, flush=True)
if not torch.cuda.is_available():
    raise SystemExit("No usable CUDA GPU. Select a GPU resource class for this Job.")
x = torch.randn(32, 32, device="cuda")
print("GPU:", torch.cuda.get_device_name(0), "matrix check:", (x @ x).sum().item(), flush=True)
PY
python /opt/decision-transformer/prepare_ant.py "$data_dir"
python -m pip freeze > packages.txt
cp /opt/source-commit.txt source-commit.txt
cp /opt/decision-transformer/gym/experiment.py experiment-source.py

python - "${args[@]}" <<'PY'
import json
import sys
with open("arguments.json", "w") as stream:
    json.dump(sys.argv[1:], stream, indent=2)
PY
exec python -u /opt/decision-transformer/gym/experiment.py "${args[@]}"
