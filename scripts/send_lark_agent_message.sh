#!/usr/bin/env bash

set -uo pipefail

default_recipient_open_id="ou_663a48636b9cd51d4a4aec323de37703"
recipient_open_id="${HWR_LARK_RECIPIENT_OPEN_ID:-$default_recipient_open_id}"
idempotency_key=""
retry_base_seconds="${HWR_LARK_RETRY_BASE_SECONDS:-10}"

usage() {
  echo "usage: $0 [--recipient OPEN_ID] [--idempotency-key KEY] MESSAGE" >&2
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --recipient)
      if [[ "$#" -lt 2 ]]; then
        usage
        exit 2
      fi
      recipient_open_id="$2"
      shift 2
      ;;
    --idempotency-key)
      if [[ "$#" -lt 2 ]]; then
        usage
        exit 2
      fi
      idempotency_key="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      echo "unknown option: $1" >&2
      usage
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ "$#" -ne 1 || -z "$1" ]]; then
  usage
  exit 2
fi

message="$1"
lark_cli="$(command -v lark-cli || true)"
if [[ -z "$lark_cli" ]]; then
  echo "lark-cli is unavailable; notification was not sent" >&2
  exit 127
fi

if [[ -z "$idempotency_key" ]]; then
  idempotency_key="$(printf '%s:%s:%s:%s' \
    "$recipient_open_id" "$message" "$(date +%s)" "$$" \
    | shasum -a 256 | cut -c1-32)"
fi

notify_status=1
for attempt in 1 2 3; do
  if LARKSUITE_CLI_NO_UPDATE_NOTIFIER=1 \
    LARKSUITE_CLI_NO_SKILLS_NOTIFIER=1 \
    "$lark_cli" im +messages-send \
      --as bot \
      --user-id "$recipient_open_id" \
      --text "$message" \
      --idempotency-key "$idempotency_key"; then
    exit 0
  else
    notify_status="$?"
  fi
  if [[ "$attempt" -lt 3 && "$retry_base_seconds" -gt 0 ]]; then
    sleep "$((attempt * retry_base_seconds))"
  fi
done

exit "$notify_status"
