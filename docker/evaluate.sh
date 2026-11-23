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
    echo "Results available in $archive"
fi
exit "$status"
