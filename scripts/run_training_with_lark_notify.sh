#!/usr/bin/env bash

set -uo pipefail

if [[ "$#" -lt 5 ]]; then
  echo "usage: $0 RUN_ID RECIPIENT_OPEN_ID LOG_PATH COMMAND [ARG ...]" >&2
  exit 2
fi

run_id="$1"
recipient_open_id="$2"
log_path="$3"
shift 3

mkdir -p "$(dirname "$log_path")"
started_at="$(date '+%Y-%m-%d %H:%M:%S %Z')"
source_commit="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

printf 'training run: %s\nstarted: %s\nsource commit: %s\ncommand:' \
  "$run_id" "$started_at" "$source_commit" | tee -a "$log_path"
printf ' %q' "$@" | tee -a "$log_path"
printf '\n' | tee -a "$log_path"

"$@" 2>&1 | tee -a "$log_path"
training_status="${PIPESTATUS[0]}"

ended_at="$(date '+%Y-%m-%d %H:%M:%S %Z')"
run_path="runs/bimanual-rl/$run_id"
episode_path="$run_path/episodes.jsonl"
checkpoint_path="$run_path/training-checkpoint.pt"
episode_count=0
checkpoint_sha256="missing"
if [[ -f "$episode_path" ]]; then
  episode_count="$(wc -l < "$episode_path" | tr -d ' ')"
fi
if [[ -f "$checkpoint_path" ]]; then
  checkpoint_sha256="$(shasum -a 256 "$checkpoint_path" | awk '{print $1}')"
fi

if [[ "$training_status" -eq 0 ]]; then
  outcome="训练已完成"
else
  outcome="训练异常退出"
fi
message="$(printf '%s\nRun: %s\n状态码: %s\nEpisode 记录数: %s\nCheckpoint SHA-256: %s\n源码提交: %s\n结束时间: %s\n日志: %s\n运行目录: %s\n请重启 Codex 任务，我会从该结果继续。' \
  "$outcome" "$run_id" "$training_status" "$episode_count" \
  "$checkpoint_sha256" "$source_commit" "$ended_at" "$log_path" "$run_path")"
idempotency_key="$(printf '%s:%s' "$run_id" "$ended_at" | shasum -a 256 | cut -c1-32)"
lark_cli="$(command -v lark-cli || true)"
notify_status=127

if [[ -n "$lark_cli" ]]; then
  for attempt in 1 2 3; do
    if LARKSUITE_CLI_NO_UPDATE_NOTIFIER=1 \
      LARKSUITE_CLI_NO_SKILLS_NOTIFIER=1 \
      "$lark_cli" im +messages-send \
        --as bot \
        --user-id "$recipient_open_id" \
        --text "$message" \
        --idempotency-key "$idempotency_key" >>"$log_path" 2>&1; then
      notify_status=0
      break
    fi
    notify_status="$?"
    if [[ "$attempt" -lt 3 ]]; then
      sleep "$((attempt * 10))"
    fi
  done
else
  printf 'lark-cli is unavailable; notification was not sent\n' >>"$log_path"
fi

if [[ "$training_status" -ne 0 ]]; then
  exit "$training_status"
fi
exit "$notify_status"
