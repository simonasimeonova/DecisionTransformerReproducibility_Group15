#!/usr/bin/env bash
set -euo pipefail
unset PYTHONHOME PYTHONPATH

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
output_dir="$repo_dir/results/evaluation"
archive="$repo_dir/training-storage/evaluation-results.tar.gz"

if [ ! -f "$repo_dir/gym/decision_transformer/evaluation_2/run.py" ]; then
    echo "Evaluation code is missing from $repo_dir" >&2
    exit 1
fi

/usr/local/bin/python - "$repo_dir/training-storage" <<'PY'
import os
import sys
from pathlib import Path
path = Path(sys.argv[1]).resolve()
if not os.path.ismount(path):
    raise SystemExit("Connector is not mounted: " + str(path))
PY

for training_archive in "$repo_dir"/training-storage/*_K*_seed*.tar.gz; do
    [ -f "$training_archive" ] || continue
    run_group="${training_archive##*/}"
    run_group="${run_group#*-}"
    run_group="${run_group%%_seed*}"
    mkdir -p "$repo_dir/results/$run_group"
    tar -xzf "$training_archive" -C "$repo_dir/results/$run_group" --strip-components=1
done

if [ -f "$archive" ]; then
    tar -xzf "$archive" -C "$repo_dir"
fi

cd "$repo_dir/gym"
if [ "$#" -eq 0 ]; then
    set -- --all --claim both
fi
status=0
/usr/local/bin/python -u -m decision_transformer.evaluation_2.run \
    "$@" --output "$output_dir" || status=$?
if [ -d "$output_dir" ]; then
    tar -czf "$repo_dir/evaluation-results.tar.gz" -C "$repo_dir" results/evaluation
    cp "$repo_dir/evaluation-results.tar.gz" "$archive"
    cmp "$repo_dir/evaluation-results.tar.gz" "$archive"
    sync
    echo "Archive copied to $archive; waiting 5 minutes for SWITCHdrive upload."
    sleep 300
fi
exit "$status"
