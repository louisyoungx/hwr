#!/usr/bin/env bash

set -uo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "usage: $0 RUN_ID LOG_PATH COMMAND [ARG ...]" >&2
  exit 2
fi

run_id="$1"
log_path="$2"
shift 2
script_dir="$(cd "$(dirname "$0")" && pwd)"
notification_script="$script_dir/send_lark_agent_message.sh"

mkdir -p "$(dirname "$log_path")"
started_at="$(date '+%Y-%m-%d %H:%M:%S %Z')"
source_commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
run_root="${HWR_TRAINING_RUN_ROOT:-runs/bimanual-rl}"

printf 'training run: %s\nstarted: %s\nsource commit: %s\ncommand:' \
  "$run_id" "$started_at" "$source_commit" | tee -a "$log_path"
printf ' %q' "$@" | tee -a "$log_path"
printf '\n' | tee -a "$log_path"

"$@" 2>&1 | tee -a "$log_path"
training_status="${PIPESTATUS[0]}"

ended_at="$(date '+%Y-%m-%d %H:%M:%S %Z')"
run_path="$run_root/$run_id"
episode_path="$run_path/episodes.jsonl"
checkpoint_path="$run_path/training-checkpoint.pt"
latest_path="$run_path/latest.json"
if [[ -f "$latest_path" ]]; then
  checkpoint_relative="$(python3 -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["training_checkpoint"])' \
    "$latest_path" 2>/dev/null || true)"
  candidate="$run_path/$checkpoint_relative/training-state.pt"
  if [[ -n "$checkpoint_relative" && -f "$candidate" ]]; then
    checkpoint_path="$candidate"
  fi
fi
episode_count=0
checkpoint_sha256="missing"
if [[ -f "$episode_path" ]]; then
  episode_count="$(wc -l < "$episode_path" | tr -d ' ')"
fi
if [[ -f "$checkpoint_path" ]]; then
  checkpoint_sha256="$(shasum -a 256 "$checkpoint_path" | awk '{print $1}')"
fi

if [[ "$training_status" -eq 0 ]]; then
  outcome="Training completed"
else
  outcome="Training exited abnormally"
fi
message="$(printf '%s\nRun: %s\nStatus Code: %s\nEpisode Count: %s\nCheckpoint: %s\nCheckpoint SHA-256: %s\nSource Commit: %s\nEnd Time: %s\nLog: %s\nRun Directory: %s\nPlease restart the Codex task; I will continue from this result.' \
  "$outcome" "$run_id" "$training_status" "$episode_count" \
  "$checkpoint_path" "$checkpoint_sha256" "$source_commit" "$ended_at" "$log_path" "$run_path")"
idempotency_key="$(printf '%s:%s' "$run_id" "$ended_at" | shasum -a 256 | cut -c1-32)"
"$notification_script" \
  --idempotency-key "$idempotency_key" \
  "$message" >>"$log_path" 2>&1
notify_status="$?"

if [[ "$training_status" -ne 0 ]]; then
  exit "$training_status"
fi
exit "$notify_status"
