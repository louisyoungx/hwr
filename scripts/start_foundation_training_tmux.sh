#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "usage: $0 RUN_ID [--resume]" >&2
}

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  usage
  exit 2
fi

run_id="$1"
resume=false
if [[ "$#" -eq 2 ]]; then
  if [[ "$2" != "--resume" ]]; then
    usage
    exit 2
  fi
  resume=true
fi
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]; then
  echo "foundation run id contains unsupported characters" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
repository_root="$(cd "$script_dir/.." && pwd)"
python_binary="$repository_root/.venv/bin/python"
notification_wrapper="$script_dir/run_training_with_lark_notify.sh"
output_root="${HWR_FOUNDATION_OUTPUT_ROOT:-$repository_root/runs/foundation-world-model}"
log_root="${HWR_FOUNDATION_LOG_ROOT:-$repository_root/logs/foundation-world-model}"
readiness="${HWR_FOUNDATION_DEVELOPMENT_READY:-$repository_root/artifacts/development-ready.json}"
model_root="${HWR_FOUNDATION_MODEL_ROOT:-$repository_root/models/foundation}"
device="${HWR_FOUNDATION_DEVICE:-mps}"
teacher_device="${HWR_FOUNDATION_TEACHER_DEVICE:-mps}"
session_name="hwr-foundation-$run_id"
log_path="$log_root/$run_id.log"

for executable in tmux lark-cli; do
  if ! command -v "$executable" >/dev/null 2>&1; then
    echo "$executable is required before detached foundation training" >&2
    exit 127
  fi
done
if [[ ! -x "$python_binary" || ! -x "$notification_wrapper" ]]; then
  echo "foundation Python runtime or notification wrapper is unavailable" >&2
  exit 127
fi
if [[ ! -f "$readiness" ]]; then
  echo "development-ready report does not exist: $readiness" >&2
  exit 1
fi
if [[ ! -d "$model_root" ]]; then
  echo "foundation model directory does not exist: $model_root" >&2
  exit 1
fi
if tmux has-session -t "=$session_name" 2>/dev/null; then
  echo "tmux session already exists: $session_name" >&2
  exit 1
fi

training_command=(
  "$python_binary"
  -m hwr.apps.train_foundation_world_model
  --run-id "$run_id"
  --output-root "$output_root"
  --device "$device"
  --foundation-device "$teacher_device"
  --development-ready "$readiness"
  --model-root "$model_root"
)
if [[ "$resume" == true ]]; then
  training_command+=(--resume)
fi

mkdir -p "$log_root"
tmux new-session -d -s "$session_name" -c "$repository_root" \
  env "HWR_TRAINING_RUN_ROOT=$output_root" \
  "$notification_wrapper" "$run_id" "$log_path" \
  "${training_command[@]}"

printf 'tmux session: %s\nrun: %s/%s\nlog: %s\n' \
  "$session_name" "$output_root" "$run_id" "$log_path"
