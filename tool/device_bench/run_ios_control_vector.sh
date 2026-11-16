#!/usr/bin/env bash
# A control vector on a connected iPhone's CPU and GPU, against this Mac.
#
# Packs models/LFM3-1.7B's own Q4 cache into a device bundle (once; the
# weights a vector from tool/control_vector_extract.dart is measured on),
# builds and installs tool/device_bench, copies the bundle (if it is not
# there yet), the vector and the prompt files into the app's Documents,
# launches the app in its control_vector mode (retrying until the phone is
# unlocked), waits for its record, runs the same checks on this Mac
# (tool/control_vector_bench.dart) and compares them. Records hold token
# ids, counts and rates, never reply text; refusal prompts give counts only.
# Writes tool/device_bench/results/iphone_control_vector.txt.
#
#   tool/device_bench/run_ios_control_vector.sh VECTOR PROMPTS [REFUSALS] [DEVICE_ID]
#   MODE=project SCALE=1 FIRST=8 LAST= tool/device_bench/run_ios_control_vector.sh ...
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAB="${LAB:-$ROOT/../dart-llm-lab}"
DART="${DART:-dart}"
CLI="$ROOT/build/cli/bundle/bin/dianoia"
APP="$ROOT/tool/device_bench"
BUNDLE_ID=com.noguwo.dartllmlab.lfm3Bench
BUNDLE=LFM3-1.7B-Q4
PACKED="$ROOT/build/device_bench/$BUNDLE"
OUT="$ROOT/build/device_bench/iphone/control_vector"
MODE="${MODE:-project}"
SCALE="${SCALE:-1}"
FIRST="${FIRST:-8}"
LAST="${LAST:-}"

VECTOR="${1:?usage: run_ios_control_vector.sh VECTOR PROMPTS [REFUSALS] [DEVICE_ID]}"
PROMPTS="${2:?usage: run_ios_control_vector.sh VECTOR PROMPTS [REFUSALS] [DEVICE_ID]}"
REFUSALS="${3:-}"
DEVICE="${4:-$(xcrun devicectl list devices 2>/dev/null |
  awk '/iPhone/ && /(connected|available)/ && / physical/ {for (i = 1; i <= NF; i++) if ($i ~ /^[0-9A-F]{8}-[0-9A-F]{16}$/) print $i}' |
  head -1)}"
[[ -n "$DEVICE" ]] || { echo "no connected iPhone" >&2; exit 1; }
container=(--device "$DEVICE" --domain-type appDataContainer --domain-identifier "$BUNDLE_ID")

(cd "$ROOT" && "$DART" build cli -o "$ROOT/build/cli" >/dev/null)
if [[ ! -f "$PACKED/lfm3-q4.cache" ]]; then
  "$CLI" pack --model="$LAB/models/LFM3-1.7B" --out="$PACKED" --weights=q4
fi

echo "== build and install on $DEVICE"
(cd "$APP" && flutter build ios --release >/dev/null)
xcrun devicectl device install app --device "$DEVICE" "$APP/build/ios/iphoneos/Runner.app" >/dev/null

if ! xcrun devicectl device info files "${container[@]}" --subdirectory "Documents/$BUNDLE" 2>/dev/null |
  grep -q "lfm3-q4.cache"; then
  echo "== copy the model bundle ($(du -sh "$PACKED" | cut -f1))"
  xcrun devicectl device copy to "${container[@]}" --source "$PACKED" --destination "Documents/$BUNDLE" >/dev/null
fi
staging="$(mktemp -d)"
trap 'rm -rf "${staging:?}"' EXIT
mkdir -p "$staging/cvec"
cp "$VECTOR" "$staging/cvec/vector.safetensors"
cp "$PROMPTS" "$staging/cvec/prompts.txt"
refusals_json=null
if [[ -n "$REFUSALS" ]]; then
  cp "$REFUSALS" "$staging/cvec/refusals.txt"
  refusals_json='"cvec/refusals.txt"'
fi
xcrun devicectl device copy to "${container[@]}" --source "$staging/cvec" --destination Documents/cvec >/dev/null
printf '{"mode": "control_vector", "bundle": "%s", "weights": "q4", "vector": "cvec/vector.safetensors", "vector_mode": "%s", "scale": %s, "first_layer": %s, "last_layer": %s, "prompts": "cvec/prompts.txt", "refusals": %s}\n' \
  "$BUNDLE" "$MODE" "$SCALE" "$FIRST" "${LAST:-null}" "$refusals_json" >"$staging/bench_config.json"
xcrun devicectl device copy to "${container[@]}" --source "$staging/bench_config.json" \
  --destination Documents/bench_config.json >/dev/null

# A record from an earlier run may still be there: wait for a new one.
started() { sed -n 's/^  "started": "\(.*\)",$/\1/p' "$1" 2>/dev/null | head -1; }
record=Documents/bench/control-vector-latest.json
mkdir -p "$OUT"
previous=""
if xcrun devicectl device copy from "${container[@]}" --source "$record" \
  --destination "$staging/previous.json" >/dev/null 2>&1; then
  previous="$(started "$staging/previous.json")"
fi

echo "== run (unlock the phone if it is locked)"
launched=""
for _ in $(seq 1 180); do
  if xcrun devicectl device process launch --device "$DEVICE" --terminate-existing "$BUNDLE_ID" >/dev/null 2>&1; then
    launched="$(date -u +%Y-%m-%dT%H:%M:%S)"
    break
  fi
  sleep 10
done
[[ -n "$launched" ]] || { echo "the phone stayed locked" >&2; exit 1; }
finished=""
for _ in $(seq 1 160); do
  sleep 15
  xcrun devicectl device copy from "${container[@]}" --source "$record" \
    --destination "$OUT/latest.json" >/dev/null 2>&1 || continue
  [[ "$(started "$OUT/latest.json")" != "$previous" ]] || continue
  if grep -q '"finished": true' "$OUT/latest.json"; then
    finished=1
    break
  fi
done
[[ -n "$finished" ]] || { echo "no finished record after 40 min" >&2; exit 1; }

echo "== the same checks on this Mac"
layers="$FIRST-$LAST"
args=("$PACKED" "$VECTOR" "$PROMPTS" --mode="$MODE" --scale="$SCALE" --layers="$layers"
  --out="$ROOT/build/device_bench/mac_control_vector.json" --compare="$OUT/latest.json")
[[ -z "$REFUSALS" ]] || args+=(--refusals="$REFUSALS")
mkdir -p "$APP/results"
{
  echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ), launched $launched UTC, $BUNDLE (q4), $MODE scale $SCALE layers $layers"
  (cd "$ROOT" && "$DART" run tool/control_vector_bench.dart "${args[@]}" 2>/dev/null)
} | tee "$APP/results/iphone_control_vector.txt"
