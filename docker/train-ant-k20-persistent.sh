#!/usr/bin/env bash
set -euo pipefail
unset PYTHONHOME PYTHONPATH

out="${DT_OUTPUT_DIR:-/home/renku/work/DecisionTransformerReproducibility_Group15/training-storage}"
wait_seconds="${DT_UPLOAD_WAIT_SECONDS:-300}"
[[ "$wait_seconds" =~ ^[0-9]+$ ]] || { echo 'Upload wait must be a nonnegative integer'; exit 1; }
python - "$out" <<'PY'
import os
import sys
import uuid
from pathlib import Path
path = Path(sys.argv[1])
if not os.path.ismount(path.resolve()):
    raise SystemExit('Connector is not mounted: ' + str(path))
marker = path / ('write-check-' + uuid.uuid4().hex + '.txt')
marker.write_text('Ant K=20 training started\n')
if marker.read_text() != 'Ant K=20 training started\n':
    raise SystemExit('Connector write/read check failed')
print('Connector writable:', marker, flush=True)
PY
df -h /home/renku/work "$out"
export DT_WORK_DIR
DT_WORK_DIR=$(mktemp -d /home/renku/work/k20-local-XXXXXX)
status=0
/usr/local/bin/train-ant-k20 "$@" || status=$?
run_dir=$(find "$DT_WORK_DIR/results" -mindepth 1 -maxdepth 1 -type d -print -quit)
[[ -n "$run_dir" ]] || { echo 'No result directory to archive'; exit 1; }
name="k20-$(basename "$run_dir").tar.gz"
tar --exclude='*/data' -C "$DT_WORK_DIR" -czf "$DT_WORK_DIR/$name" results
df -h "$DT_WORK_DIR"
ls -lh "$DT_WORK_DIR/$name"
cp "$DT_WORK_DIR/$name" "$out/$name"
cmp "$DT_WORK_DIR/$name" "$out/$name"
sync
echo "Archive copied to mount: $out/$name"
echo "Waiting $wait_seconds seconds for upload. Verify and download the archive in SWITCHdrive now."
sleep "$wait_seconds"
echo 'Upload grace period finished. Mount readback does not verify remote persistence.'
exit "$status"
