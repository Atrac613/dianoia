#!/usr/bin/env bash
# Qwen3 through ModelStore on a connected iPhone, end to end: the phone
# downloads Qwen's Q8_0 GGUF from Hugging Face, imports it, and answers the
# qwen3 mode's prompts on Metal and on the CPU (lib/qwen3_check.dart).
# Prints the phone's record; tool/device_bench/results/iphone_qwen3_<spec>.json
# keeps it.
#
#   tool/device_bench/run_ios_qwen3.sh [DEVICE_ID]
#   SPEC=1.7b tool/device_bench/run_ios_qwen3.sh
#   REINSTALL=1 tool/device_bench/run_ios_qwen3.sh   # download again
#   VERIFY_COST=1 tool/device_bench/run_ios_qwen3.sh # n-token pass times
#     of 0.6B and 1.7B, for speculative decoding (qwen3VerifyCost)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP="$ROOT/tool/device_bench"
BUNDLE_ID=com.noguwo.dartllmlab.lfm3Bench
SPEC="${SPEC:-0.6b}"
REINSTALL="${REINSTALL:+true}"

DEVICE="${1:-$(xcrun devicectl list devices 2>/dev/null |
  awk '/iPhone/ && /connected/ && / physical/ {for (i = 1; i <= NF; i++) if ($i ~ /^[0-9A-F]{8}-[0-9A-F]{16}$/) print $i}' |
  head -1)}"
[[ -n "$DEVICE" ]] || { echo "no connected iPhone" >&2; exit 1; }
container=(--device "$DEVICE" --domain-type appDataContainer --domain-identifier "$BUNDLE_ID")

echo "== build and install on $DEVICE"
(cd "$APP" && flutter build ios --release >/dev/null)
xcrun devicectl device install app --device "$DEVICE" "$APP/build/ios/iphoneos/Runner.app" >/dev/null

staging="$(mktemp -d)"
trap 'rm -rf "${staging:?}"' EXIT
printf '{"mode": "qwen3", "spec": "%s", "reinstall": %s, "verify_cost": %s, "build": "%s"}\n' \
  "$SPEC" "${REINSTALL:-false}" "$([[ -n "${VERIFY_COST:-}" ]] && echo true || echo false)" "$(git -C "$ROOT" describe --always --dirty)" >"$staging/bench_config.json"
xcrun devicectl device copy to "${container[@]}" --source "$staging/bench_config.json" \
  --destination Documents/bench_config.json >/dev/null

started() { python3 -I -c "import json,sys; print(json.load(open(sys.argv[1])).get('started', ''))" "$1" 2>/dev/null || true; }
pull() {
  xcrun devicectl device copy from "${container[@]}" --source Documents/bench/qwen3-latest.json \
    --destination "$1" >/dev/null 2>&1
}
previous=""
if pull "$staging/previous.json"; then previous="$(started "$staging/previous.json")"; fi

echo "== run (unlock the phone if it is locked)"
launched=""
for _ in $(seq 1 180); do
  if xcrun devicectl device process launch --device "$DEVICE" --terminate-existing "$BUNDLE_ID" >/dev/null 2>&1; then
    launched=1
    break
  fi
  sleep 10
done
[[ -n "$launched" ]] || { echo "the phone stayed locked" >&2; exit 1; }

mkdir -p "$APP/results"
out="$APP/results/iphone_qwen3_$SPEC${VERIFY_COST:+_verify_cost}.json"
for _ in $(seq 1 160); do
  sleep 15
  pull "$staging/latest.json" || continue
  [[ "$(started "$staging/latest.json")" != "$previous" ]] || continue
  if grep -q '"finished": true' "$staging/latest.json"; then
    cp "$staging/latest.json" "$out"
    cat "$out"
    exit 0
  fi
done
echo "no finished record after 40 min" >&2
pull "$staging/latest.json" && cat "$staging/latest.json"
exit 1
