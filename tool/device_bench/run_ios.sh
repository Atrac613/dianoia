#!/usr/bin/env bash
# LFM3 through dianoia on a connected iPhone, end to end.
#
# Prepares the device bundle (once), builds and installs
# tool/device_bench, copies the bundle and the reference token ids into
# the app's Documents (the bundle only if it is not there yet), launches the
# app (retrying until the phone is unlocked), waits for its record, copies
# the results back and checks them against the same runs on this Mac.
# Writes tool/device_bench/results/iphone_<weights>.txt.
#
#   tool/device_bench/run_ios.sh [DEVICE_ID]   # default: the first connected iPhone
#   WEIGHTS=q4 tool/device_bench/run_ios.sh    # Liquid's QAD Q4_0 instead of Q8
#
# Q8 packs models/LFM3-1.7B; Q4 imports
# models/LFM3-1.7B-QAD-GGUF/LFM3-1.7B-QAD-Q4_0.gguf (LiquidAI/LFM3-1.7B-GGUF).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Models, the transformers reference runs and the Python comparison tools
# stay in dart-llm-lab.
LAB="${LAB:-$ROOT/../dart-llm-lab}"
PY="$LAB/venv/bin/python"
DART="${DART:-dart}"
CLI="$ROOT/build/cli/bundle/bin/dianoia"
[[ -x "$PY" ]] || { echo "missing $PY (run the lab's scripts/setup.sh)" >&2; exit 1; }

# LIFECYCLE_CYCLES=N: the app only loads, uses and closes the model N times
# (lib/lifecycle_check.dart); print what it recorded and stop.
lifecycle_report() {
  "$PY" - "$1" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
rows = r.get("lifecycle", [])
for row in rows:
    print(f"  cycle {row['cycle']}: {row['buffers_open']} GPU buffers open, "
          f"{row['buffers_closed']} after close, footprint {row['footprint_mb']:.0f} MB, "
          f"{row['seconds']:.1f} s")
if rows:
    settled, later = rows[0], rows[1:]
    leaked = any(row["buffers_closed"] != settled["buffers_closed"] for row in later)
    growth = later[-1]["footprint_mb"] - settled["footprint_mb"] if later else 0
    print(f"buffers after close stay at {settled['buffers_closed']}: {'NO' if leaked else 'yes'}; "
          f"footprint {growth:+.0f} MB after cycle 0; "
          f"closed model refuses to run: {r.get('closed_model_refused')}")
PY
}

BUNDLE_ID=com.noguwo.dartllmlab.lfm3Bench
WEIGHTS="${WEIGHTS:-q8}"
MODEL="$LAB/models/LFM3-1.7B"
GGUF="$LAB/models/LFM3-1.7B-QAD-GGUF/LFM3-1.7B-QAD-Q4_0.gguf"
case "$WEIGHTS" in
  q8) BUNDLE=LFM3-1.7B-device ;;
  q4) BUNDLE=LFM3-1.7B-QAD-Q4 ;;
  *) echo "WEIGHTS must be q8 or q4" >&2; exit 2 ;;
esac
PACKED="$LAB/models/$BUNDLE"
RUNS="$LAB/runs/lfm3"
OUT="$ROOT/build/device_bench/iphone/$WEIGHTS"
MAC="$ROOT/build/device_bench/mac_$WEIGHTS"
APP="$ROOT/tool/device_bench"

DEVICE="${1:-$(xcrun devicectl list devices 2>/dev/null |
  awk '/iPhone/ && /connected/ && / physical/ {for (i = 1; i <= NF; i++) if ($i ~ /^[0-9A-F]{8}-[0-9A-F]{16}$/) print $i}' |
  head -1)}"
[[ -n "$DEVICE" ]] || { echo "no connected iPhone" >&2; exit 1; }
container=(--device "$DEVICE" --domain-type appDataContainer --domain-identifier "$BUNDLE_ID")

(cd "$ROOT" && "$DART" build cli -o "$ROOT/build/cli" >/dev/null)
if [[ ! -f "$PACKED/lfm3-$WEIGHTS.cache" ]]; then
  if [[ "$WEIGHTS" == q8 ]]; then
    "$CLI" pack --model="$MODEL" --out="$PACKED"
  else
    [[ -f "$GGUF" ]] || { echo "download $GGUF from LiquidAI/LFM3-1.7B-GGUF first" >&2; exit 1; }
    "$CLI" import-gguf --gguf="$GGUF" --model="$MODEL" --out="$PACKED"
  fi
fi

echo "== build and install on $DEVICE"
(cd "$APP" && flutter build ios --release >/dev/null)
xcrun devicectl device install app --device "$DEVICE" "$APP/build/ios/iphoneos/Runner.app" >/dev/null

if ! xcrun devicectl device info files "${container[@]}" --subdirectory "Documents/$BUNDLE" 2>/dev/null |
  grep -q "lfm3-$WEIGHTS.cache"; then
  echo "== copy the model bundle ($(du -sh "$PACKED" | cut -f1))"
  xcrun devicectl device copy to "${container[@]}" --source "$PACKED" --destination "Documents/$BUNDLE" >/dev/null
fi
staging="$(mktemp -d)"
trap 'rm -rf "${staging:?}"' EXIT
for c in tools long; do
  mkdir -p "$staging/ref/$c"
  cp "$RUNS/ref_$c/ids.i32" "$RUNS/ref_$c/gen.i32" "$staging/ref/$c/"
done
xcrun devicectl device copy to "${container[@]}" --source "$staging/ref" --destination Documents/ref >/dev/null
cp "$ROOT/example/tool_cases.json" "$staging/tool_cases.json"
xcrun devicectl device copy to "${container[@]}" --source "$staging/tool_cases.json" \
  --destination Documents/tool_cases.json >/dev/null
printf '{"bundle": "%s", "weights": "%s", "lifecycle_cycles": %s}\n' "$BUNDLE" "$WEIGHTS" \
  "${LIFECYCLE_CYCLES:-0}" >"$staging/bench_config.json"
xcrun devicectl device copy to "${container[@]}" --source "$staging/bench_config.json" \
  --destination Documents/bench_config.json >/dev/null

# A record from an earlier run may still be there: wait for a new one.
started() { "$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('started', ''))" "$1" 2>/dev/null || true; }
mkdir -p "$OUT"
previous=""
if xcrun devicectl device copy from "${container[@]}" --source Documents/bench/latest.json \
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
for _ in $(seq 1 120); do
  sleep 15
  xcrun devicectl device copy from "${container[@]}" --source Documents/bench/latest.json \
    --destination "$OUT/latest.json" >/dev/null 2>&1 || continue
  [[ "$(started "$OUT/latest.json")" != "$previous" ]] || continue
  if grep -q '"finished": true' "$OUT/latest.json"; then
    finished=1
    break
  fi
done
[[ -n "$finished" ]] || { echo "no finished record after 30 min" >&2; exit 1; }
if [[ "${LIFECYCLE_CYCLES:-0}" -gt 0 ]]; then
  lifecycle_report "$OUT/latest.json"
  exit 0
fi
for d in gpu_tools gpu_long cpu_tools; do
  xcrun devicectl device copy from "${container[@]}" --source "Documents/out/$WEIGHTS/$d" --destination "$OUT/$d" >/dev/null
done

echo "== the same teacher-forced runs on this Mac"
for c in tools long; do
  "$CLI" run --model="$PACKED" --ids="$RUNS/ref_$c/ids.i32" --force="$RUNS/ref_$c/gen.i32" \
    --out="$MAC/gpu_$c" --all-logits-max=128 --max-seq=2048 --weights="$WEIGHTS" \
    --gpu >/dev/null
done
"$CLI" run --model="$PACKED" --ids="$RUNS/ref_tools/ids.i32" --force="$RUNS/ref_tools/gen.i32" \
  --out="$MAC/cpu_tools" --all-logits-max=128 --max-seq=2048 --weights="$WEIGHTS" \
  --native-threads=6 >/dev/null

mkdir -p "$APP/results"
{
  echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ), launched $launched UTC, $BUNDLE ($WEIGHTS)"
  "$PY" - "$OUT/latest.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
d = r["device"]
print(f"device: {d['gpu']}, {r['os']}, {d['physical_memory'] / 2**30:.1f} GiB, "
      f"max buffer {d['max_buffer_length'] / 2**30:.1f} GiB, "
      f"GPU working set {d['recommended_working_set'] / 2**30:.1f} GiB, "
      f"{d['weight_segments']} weight buffer(s)")
def asks(label, runs):
    for i, a in enumerate(runs, 1):
        print(f"  {label} ask {i}: prompt {a['prompt_tok_per_s']:6.1f} tok/s, decode {a['decode_tok_per_s']:5.1f} tok/s, "
              f"total {a['total_s']:5.2f} s, thermal {a['thermal_before']}->{a['thermal_after']}, "
              f"footprint {a['footprint_mb']} MB, {a['tool_calls']}")
asks("GPU", r["gpu"]["ask"])
for c in ("tools", "long"):
    f = r["gpu"].get(f"forced_{c}")
    if f:
        print(f"  GPU forced {c}: {f['seq_len']} prompt tokens at {f['prompt_tok_per_s']:.1f} tok/s, "
              f"decode {f['decode_tok_per_s']:.1f} tok/s")
chat = r["gpu"].get("chat")
if chat:
    print(f"  GPU chat: system and tools {chat['prefix_tokens']} tokens; reuse gives the from-scratch logits: "
          f"{'bit for bit' if chat['identical_logits'] else 'NO'}")
    print(f"    cold first token {chat['cold']['ttft_s'] * 1000:.0f} ms ({chat['cold']['prompt_tokens']} tokens), "
          f"warm-up {chat['warm_up_s'] * 1000:.0f} ms, warm first token {chat['warm_ttft_median_s'] * 1000:.0f} ms median, "
          f"tool round trip {chat['tool_round_trip']['ttft_s'] * 1000:.0f} ms, tool calls right {chat['passed']}/{chat['cases']}")
    if "state_bytes" in chat:
        print(f"    state file {chat['state_bytes'] / 1e6:.1f} MB, saved in {chat['state_save_s'] * 1000:.0f} ms, "
              f"loaded in {chat['state_load_s'] * 1000:.0f} ms (right after saving)")
for k, v in r["cpu"].items():
    if k.startswith("threads_"):
        asks(f"CPU {k.replace('threads_', 'x')}", v["ask"])
EOF
  echo "iPhone GPU against Mac GPU (tools, long):"
  "$PY" "$LAB/tools/lfm3_compare.py" "$MAC/gpu_tools" "$OUT/gpu_tools" | grep -E "^(OK|FAILED)|L23|logits|greedy" | sed 's/^/  /' || true
  "$PY" "$LAB/tools/lfm3_compare.py" "$MAC/gpu_long" "$OUT/gpu_long" | grep -E "^(OK|FAILED)|L23|logits|greedy" | sed 's/^/  /' || true
  echo "iPhone CPU against Mac CPU (tools), bitwise:"
  for f in hidden.f32 logits.f32 gen.i32 gen_logits.f32; do
    if cmp -s "$MAC/cpu_tools/$f" "$OUT/cpu_tools/$f"; then echo "  $f identical"; else echo "  $f differs"; fi
  done
  echo "KL against float32 (tools):"
  for d in gpu_tools cpu_tools; do
    "$PY" "$LAB/tools/lfm3_quality.py" "$RUNS/ref_tools" "$OUT/$d" | sed "s/^/  ${d%_tools}: /"
  done
} | tee "$APP/results/iphone_$WEIGHTS.txt"
