#!/usr/bin/env bash
# LFM3 through dianoia on the GPU (Vulkan) and CPU of a connected Android
# device, end to end, as run_ios.sh does for the iPhone.
#
# Builds the bench app's APK with Gradle (no Android Studio needed: any JDK
# 17+), installs it, pushes the model bundle (once), the reference token ids
# and the tool cases into the app's external files directory, launches the
# app, waits for its record, pulls the results and checks them against the
# same runs on this Mac. Writes tool/device_bench/results/android_<weights>.txt
# (with SKIP_CPU=1, the GPU alone: android_<weights>_gpu.txt).
# Q8_VECTOR_ACCEPTANCE=1 focuses on a repeated vector-load comparison and
# enables vector loads for asks, chat/state reuse, and tools/long dumps,
# with direct scalar controls. It requires thermal status 0 at launch.
# Q8_VECTOR_DEFAULT_ACCEPTANCE=1 runs the same gate using automatic
# selection instead of forcing the candidate, and checks a factory probe.
#
#   tool/device_bench/run_android.sh            # QAD Q4, the first adb device
#   WEIGHTS=q8 ANDROID_SERIAL=... tool/device_bench/run_android.sh
#
# The device must stay awake and unlocked while the app runs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Models, the transformers reference runs and the Python comparison tools
# stay in dart-llm-lab.
LAB="${LAB:-$ROOT/../dart-llm-lab}"
PY="$LAB/venv/bin/python"
DART="${DART:-dart}"
CLI="$ROOT/build/cli/bundle/bin/dianoia"
[[ -x "$PY" ]] || { echo "missing $PY (run the lab's scripts/setup.sh)" >&2; exit 1; }
GPU_SUSTAINED_SECONDS="${GPU_SUSTAINED_SECONDS:-0}"
[[ "$GPU_SUSTAINED_SECONDS" =~ ^(0|[1-9][0-9]*)$ && "$GPU_SUSTAINED_SECONDS" -le 3600 ]] || {
  echo "GPU_SUSTAINED_SECONDS must be 0..3600" >&2; exit 2;
}
if [[ "${Q8_VECTOR_DEFAULT_ACCEPTANCE:-0}" == 1 ]]; then
  Q8_VECTOR_ACCEPTANCE=1
fi

PACKAGE=com.noguwo.dartllmlab.lfm3_bench
WEIGHTS="${WEIGHTS:-q4}"
MODEL="$LAB/models/LFM3-1.7B"
GGUF="$LAB/models/LFM3-1.7B-QAD-GGUF/LFM3-1.7B-QAD-Q4_0.gguf"
case "$WEIGHTS" in
  q8) BUNDLE=LFM3-1.7B-device ;;
  q4) BUNDLE=LFM3-1.7B-QAD-Q4 ;;
  *) echo "WEIGHTS must be q8 or q4" >&2; exit 2 ;;
esac
PACKED="$LAB/models/$BUNDLE"
RUNS="$LAB/runs/lfm3"
OUT="$ROOT/build/device_bench/android/$WEIGHTS"
MAC="$ROOT/build/device_bench/mac_$WEIGHTS"
APP="$ROOT/tool/device_bench"
REMOTE="/sdcard/Android/data/$PACKAGE/files"

[[ -n "$(adb devices | awk 'NR > 1 && $2 == "device"')" ]] || { echo "no adb device" >&2; exit 1; }
JAVA_HOME="${JAVA_HOME:-$(/usr/libexec/java_home 2>/dev/null || true)}"
[[ -n "$JAVA_HOME" ]] || { echo "set JAVA_HOME to a JDK 17+" >&2; exit 1; }
export JAVA_HOME

(cd "$ROOT" && "$DART" build cli -o "$ROOT/build/cli" >/dev/null)
if [[ ! -f "$PACKED/lfm3-$WEIGHTS.cache" ]]; then
  if [[ "$WEIGHTS" == q8 ]]; then
    "$CLI" pack --model="$MODEL" --out="$PACKED"
  else
    [[ -f "$GGUF" ]] || { echo "download $GGUF from LiquidAI/LFM3-1.7B-GGUF first" >&2; exit 1; }
    "$CLI" import-gguf --gguf="$GGUF" --model="$MODEL" --out="$PACKED"
  fi
fi

echo "== build and install on $(adb shell getprop ro.product.model | tr -d '\r')"
(cd "$APP" && flutter pub get >/dev/null && cd android && ./gradlew -q assembleRelease -Ptarget-platform=android-arm64 >/dev/null)
adb install -r "$APP/build/app/outputs/flutter-apk/app-release.apk" >/dev/null

# Android gives the app its external files directory only if the app asks
# for it (MainActivity does at start); one that adb created first belongs to
# the shell user and the app cannot use it. Recreate it through the app.
owner="$(adb shell "stat -c %U '$REMOTE' 2>/dev/null" | tr -d '\r')"
if [[ "$owner" != u0_a* ]]; then
  adb shell rm -rf "/sdcard/Android/data/$PACKAGE"
  adb shell am start -W -n "$PACKAGE/.MainActivity" >/dev/null
  for _ in $(seq 1 30); do
    [[ "$(adb shell "stat -c %U '$REMOTE' 2>/dev/null" | tr -d '\r')" == u0_a* ]] && break
    sleep 1
  done
  adb shell am force-stop "$PACKAGE"
fi
remote_size() { adb shell "stat -c %s '$1' 2>/dev/null" | tr -d '\r'; }
for f in "$PACKED"/*; do
  name="$(basename "$f")"
  if [[ "$(remote_size "$REMOTE/$BUNDLE/$name")" != "$(stat -f %z "$f")" ]]; then
    echo "== push $BUNDLE/$name ($(du -h "$f" | cut -f1))"
    adb push "$f" "$REMOTE/$BUNDLE/$name" >/dev/null
  fi
done
staging="$(mktemp -d)"
observer_pid=""
cleanup() {
  if [[ -n "$observer_pid" ]]; then
    kill "$observer_pid" 2>/dev/null || true
    wait "$observer_pid" 2>/dev/null || true
  fi
  rm -rf "${staging:?}"
}
trap cleanup EXIT
for c in tools long; do
  mkdir -p "$staging/ref/$c"
  cp "$RUNS/ref_$c/ids.i32" "$RUNS/ref_$c/gen.i32" "$staging/ref/$c/"
done
cp "$ROOT/example/tool_cases.json" "$staging/"
# SKIP_CPU=1 measures the GPU only, for quicker GPU work.
printf '{"bundle": "%s", "weights": "%s", "skip_cpu": %s, "q8_vector_acceptance": %s, "q8_vector_default_acceptance": %s, "gpu_sustained_seconds": %s}\n' "$BUNDLE" "$WEIGHTS" \
  "$([[ "${SKIP_CPU:-0}" == 1 ]] && echo true || echo false)" \
  "$([[ "${Q8_VECTOR_ACCEPTANCE:-0}" == 1 ]] && echo true || echo false)" \
  "$([[ "${Q8_VECTOR_DEFAULT_ACCEPTANCE:-0}" == 1 ]] && echo true || echo false)" \
  "$GPU_SUSTAINED_SECONDS" >"$staging/bench_config.json"
adb push "$staging/ref" "$REMOTE/" >/dev/null
adb push "$staging/tool_cases.json" "$staging/bench_config.json" "$REMOTE/" >/dev/null

# A record from an earlier run may still be there: wait for a new one.
started() { "$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('started', ''))" "$1" 2>/dev/null || true; }
is_finished() { "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("finished") is True)' "$1" 2>/dev/null || true; }
mkdir -p "$OUT"
previous=""
if adb pull "$REMOTE/bench/latest.json" "$staging/previous.json" >/dev/null 2>&1; then
  previous="$(started "$staging/previous.json")"
fi
thermal() { adb shell dumpsys thermalservice 2>/dev/null | tr -d '\r' | awk -F': ' '/Thermal Status/ {print $2; exit}'; }
thermal_before="$(thermal)"
if [[ "${Q8_VECTOR_ACCEPTANCE:-0}" == 1 && "$thermal_before" != 0 ]]; then
  echo "Q8 vector acceptance requires thermal status 0, got '$thermal_before'; let the device cool and retry" >&2
  exit 1
fi

echo "== run (keep the device awake and unlocked)"
if [[ "$GPU_SUSTAINED_SECONDS" -gt 0 ]]; then
  "$PY" "$LAB/tools/android_gpu_observe.py" --out "$OUT/sustained_telemetry.jsonl" \
    --seconds "$((GPU_SUSTAINED_SECONDS + 180))" &
  observer_pid=$!
fi
adb shell input keyevent KEYCODE_WAKEUP >/dev/null
adb shell am start -S -n "$PACKAGE/.MainActivity" >/dev/null
launched="$(date -u +%Y-%m-%dT%H:%M:%S)"
finished=""
wait_minutes=$((GPU_SUSTAINED_SECONDS / 60 + 10))
[[ "$wait_minutes" -ge 40 ]] || wait_minutes=40
for _ in $(seq 1 "$((wait_minutes * 4))"); do
  sleep 15
  adb pull "$REMOTE/bench/latest.json" "$OUT/latest.json" >/dev/null 2>&1 || continue
  [[ "$(started "$OUT/latest.json")" != "$previous" ]] || continue
  if [[ "$(is_finished "$OUT/latest.json")" == True ]]; then
    finished=1
    break
  fi
done
if [[ -z "$finished" ]]; then
  adb pull "$REMOTE/bench/error.txt" "$OUT/error.txt" >/dev/null 2>&1 && cat "$OUT/error.txt" >&2
  echo "no finished record after $wait_minutes min" >&2
  exit 1
fi
thermal_after="$(thermal)"
rm -rf "${OUT:?}"/cpu_tools "${OUT:?}"/gpu_tools "${OUT:?}"/gpu_long
if [[ "${Q8_VECTOR_ACCEPTANCE:-0}" == 1 ]]; then
  rm -rf "${OUT:?}"/scalar_tools "${OUT:?}"/scalar_long
fi
dumps=(gpu_tools gpu_long)
[[ "$GPU_SUSTAINED_SECONDS" -gt 0 ]] && dumps+=(reference_tools reference_long)
[[ "${Q8_VECTOR_ACCEPTANCE:-0}" == 1 ]] && dumps+=(scalar_tools scalar_long)
[[ "${SKIP_CPU:-0}" == 1 ]] || dumps+=(cpu_tools)
for d in "${dumps[@]}"; do
  adb pull "$REMOTE/out/$WEIGHTS/$d" "$OUT/" >/dev/null 2>&1 || true
done

echo "== CPU-int8 quality controls on this Mac"
for c in tools long; do
  "$CLI" run --model="$PACKED" --ids="$RUNS/ref_$c/ids.i32" --force="$RUNS/ref_$c/gen.i32" \
    --out="$MAC/cpu_$c" --all-logits-max=128 --max-seq=2048 --weights="$WEIGHTS" \
    --native-threads=6 --native-inputs=int8 >/dev/null
done

mkdir -p "$APP/results"
{
  echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ), launched $launched UTC, $BUNDLE ($WEIGHTS)"
  echo "device: $(adb shell getprop ro.product.manufacturer | tr -d '\r') $(adb shell getprop ro.product.model | tr -d '\r')," \
    "SoC $(adb shell getprop ro.soc.model | tr -d '\r'), Android $(adb shell getprop ro.build.version.release | tr -d '\r')," \
    "thermal status $thermal_before -> $thermal_after"
  "$PY" - "$OUT/latest.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
d = r.get("device", {})
mhz = sorted({k // 1000 for k in d.get("max_khz", []) if k}, reverse=True)
print(f"cpu: {d.get('cores')} cores, top MHz {mhz}, parts {d.get('cpu_parts')}, "
      f"{(d.get('mem_total_kb') or 0) / 2**20:.1f} GiB")
if "gpu" in d:
    print(f"gpu: {d['gpu']}, max buffer {d['max_buffer_length'] / 2**30:.1f} GiB, "
          f"{d['weight_segments']} weight buffer(s)")
if r.get("gpu_error"):
    print(f"gpu: not used ({r['gpu_error']})")
def chat_lines(label, chat):
    print(f"  {label} chat: system and tools {chat['prefix_tokens']} tokens; reuse gives the from-scratch logits: "
          f"{'bit for bit' if chat['identical_logits'] else 'NO'}")
    print(f"    cold first token {chat['cold']['ttft_s'] * 1000:.0f} ms ({chat['cold']['prompt_tokens']} tokens), "
          f"warm-up {chat['warm_up_s'] * 1000:.0f} ms, warm first token {chat['warm_ttft_median_s'] * 1000:.0f} ms median, "
          f"tool round trip {chat['tool_round_trip']['ttft_s'] * 1000:.0f} ms, tool calls right {chat['passed']}/{chat['cases']}")
    print(f"    state file {chat['state_bytes'] / 1e6:.1f} MB, saved in {chat['state_save_s'] * 1000:.0f} ms, "
          f"loaded in {chat['state_load_s'] * 1000:.0f} ms (right after saving)")
g = r.get("gpu")
if g:
    for i, a in enumerate(g.get("ask", []), 1):
        gpu_ms = a.get("decode_gpu_ms_per_token")
        split = (f", decode GPU {gpu_ms:.1f} ms/token (encoding {a['decode_encode_ms_per_token']:.2f})"
                 if gpu_ms else "")
        print(f"  GPU ask {i}: prompt {a['prompt_tok_per_s']:6.1f} tok/s, decode {a['decode_tok_per_s']:5.1f} tok/s, "
              f"total {a['total_s']:5.2f} s{split}, rss {a['footprint_mb']} MB, {a['tool_calls']}")
    sustained = g.get("sustained")
    if sustained:
        import statistics
        samples = sustained["samples"]
        print(f"  Sustained fixed-context run: {sustained.get('elapsed_s'):.1f}s, {len(samples)} blocks, "
              f"context {sustained['prompt_tokens']}->{sustained['context_end']}, {sustained['decode_steps_per_block']} steps/block")
        for automatic in (False, True):
            group = [s for s in samples if s["automatic"] == automatic and s['profiled']]
            med = lambda key: statistics.median(s[key] for s in group)
            print(f"  Sustained auto={automatic}: GPU {med('gpu_ms_per_token'):.2f} ms/token; "
                  f"Q4 {med('q4_ms_per_token'):.2f}, Q8 {med('q8_ms_per_token'):.2f}, "
                  f"other {med('other_ms_per_token'):.2f}; wall {med('wall_ms_per_token'):.2f}")
            plain = [s for s in samples if s['automatic'] == automatic and not s['profiled']]
            if plain:
                print(f"  Sustained auto={automatic}, unprofiled: GPU {statistics.median(s['gpu_ms_per_token'] for s in plain):.2f}, "
                      f"wall {statistics.median(s['wall_ms_per_token'] for s in plain):.2f} ms/token")
        for name, subset in (("first minute", [s for s in samples if s["elapsed_s"] <= 60]),
                             ("last minute", [s for s in samples if s["elapsed_s"] >= sustained['elapsed_s'] - 60])):
            subset = [s for s in subset if s["automatic"] and not s['profiled']]
            if subset:
                print(f"  Sustained {name}, auto unprofiled: GPU median {statistics.median(s['gpu_ms_per_token'] for s in subset):.2f} ms/token")
    for c in ("tools", "long"):
        f = g.get(f"forced_{c}")
        if f:
            print(f"  GPU forced {c}: {f['seq_len']} prompt tokens at {f['prompt_tok_per_s']:.1f} tok/s, "
                  f"decode {f['decode_tok_per_s']:.1f} tok/s")
    if (g.get("profile") or {}).get("copy_gb_per_s"):
        print(f"  GPU copy bandwidth: {g['profile']['copy_gb_per_s']:.1f} GB/s (read plus written)")
    for phase, prof in (g.get("profile") or {}).items():
        if not isinstance(prof, dict) or not prof:
            continue
        total = sum(v["ms_per_token"] for v in prof.values())
        parts = sorted(prof.items(), key=lambda kv: -kv[1]["ms_per_token"])
        print(f"  GPU profile, {phase}: {total:.1f} ms/token: " +
              ", ".join(f"{k} {v['ms_per_token']:.2f} ({v['dispatches_per_token']:.0f}x)" for k, v in parts))
    sweep = (g.get("profile") or {}).get("decode_sweep", [])
    if sweep:
        import statistics
        for rows in (0, 1, 2, 4):
            samples = [s for s in sweep if s["rows"] == rows]
            gpu_ms = [s["gpu_ms_per_token"] for s in samples]
            wall_ms = [s["wall_ms_per_token"] for s in samples]
            identical = all(s["differing_logits"] == 0 and s["non_finite"] == 0 for s in samples)
            print(f"  Decode sweep rows {rows} ({len(samples)} runs): GPU median {statistics.median(gpu_ms):.2f} "
                  f"ms/token, range {min(gpu_ms):.2f}-{max(gpu_ms):.2f}; wall median {statistics.median(wall_ms):.2f} "
                  f"ms/token; logits vs old GEMM {'bit-identical' if identical else 'FAILED'}")
    q8_groups = (g.get("profile") or {}).get("q8_group_sweep", [])
    if q8_groups:
        import statistics
        for group in sorted({s["q8_workgroup"] for s in q8_groups}):
            samples = [s for s in q8_groups if s["q8_workgroup"] == group]
            total = [s["gpu_ms_per_token"] for s in samples]
            q8 = [s["q8_ms_per_token"] for s in samples]
            wall = [s["wall_ms_per_token"] for s in samples]
            identical = all(s["differing_logits"] == 0 and s["non_finite"] == 0 for s in samples)
            print(f"  Q8 workgroup {group}, full decode: GPU median {statistics.median(total):.2f} ms/token, "
                  f"Q8 alone {statistics.median(q8):.2f}, wall {statistics.median(wall):.2f}; "
                  f"logits vs old GEMM {'bit-identical' if identical else 'FAILED'}")
    q8_matrices = g.get("q8_workgroup_sweep", [])
    if q8_matrices:
        import statistics
        for group in sorted({s["q8_workgroup"] for s in q8_matrices}):
            samples = [s for s in q8_matrices if s["q8_workgroup"] == group]
            times = [s["gpu_ms_per_dispatch"] for s in samples]
            identical = all(s["differing_outputs"] == 0 and s["non_finite"] == 0 for s in samples)
            print(f"  Q8 workgroup {group}, isolated LM head: GPU median {statistics.median(times):.3f} "
                  f"ms/dispatch, range {min(times):.3f}-{max(times):.3f} ({len(samples)} runs); "
                  f"vs old GEMM {'bit-identical' if identical else 'FAILED'}")
    vector_full = (g.get("profile") or {}).get("q8_vector_sweep", [])
    vector_matrix = g.get("q8_vector_sweep", [])
    if vector_full or vector_matrix:
        import statistics
        for vector in (False, True):
            samples = [s for s in vector_full if s["q8_vector"] == vector]
            if samples:
                total = [s["gpu_ms_per_token"] for s in samples]
                q8 = [s["q8_ms_per_token"] for s in samples]
                wall = [s["wall_ms_per_token"] for s in samples]
                identical = all(s["differing_logits"] == 0 and s["non_finite"] == 0 for s in samples)
                print(f"  Q8 vector={vector}, full decode ({len(samples)} runs): GPU median {statistics.median(total):.2f} "
                      f"ms/token, Q8 alone {statistics.median(q8):.2f}, wall {statistics.median(wall):.2f}; "
                      f"logits vs old GEMM {'bit-identical' if identical else 'FAILED'}")
            samples = [s for s in vector_matrix if s["q8_vector"] == vector]
            if samples:
                times = [s["gpu_ms_per_dispatch"] for s in samples]
                identical = all(s["differing_outputs"] == 0 and s["non_finite"] == 0 for s in samples)
                print(f"  Q8 vector={vector}, isolated LM head ({len(samples)} runs): GPU median {statistics.median(times):.3f} "
                      f"ms/dispatch, range {min(times):.3f}-{max(times):.3f}; "
                      f"vs old GEMM {'bit-identical' if identical else 'FAILED'}")
    matrix_sweep = g.get("matrix_sweep", [])
    if matrix_sweep:
        import statistics
        for name in dict.fromkeys(s["matrix"] for s in matrix_sweep):
            for rows in (0, 1, 2, 4):
                samples = [s for s in matrix_sweep if s["matrix"] == name and s["decode_rows"] == rows]
                times = [s["gpu_ms_per_dispatch"] for s in samples]
                identical = all(s["differing_outputs"] == 0 and s["non_finite"] == 0 for s in samples)
                clocks = sorted({s.get("gpu_cur_freq_raw_after") or "unavailable" for s in samples})
                first = samples[0]
                print(f"  Matrix {name} {first['format']} [{first['matrix_rows']}, {first['matrix_cols']}], "
                      f"rows {rows}: GPU median {statistics.median(times):.3f} ms/dispatch, "
                      f"range {min(times):.3f}-{max(times):.3f} ({len(samples)} runs x {first['repetitions']}); "
                      f"vs old GEMM {'bit-identical' if identical else 'FAILED'}; clock raw {clocks}")
    if g.get("chat"):
        chat_lines("GPU", g["chat"])
    acceptance = g.get("vector_acceptance")
    if g.get("q8_vector_default_acceptance"):
        print(f"  Scoped vector default eligible: {g.get('q8_vector_head_default_eligible')}, "
              f"factory probe: {g.get('q8_vector_default_probe')}")
    if acceptance:
        chat = acceptance.get("chat", {})
        print(f"  Vector acceptance chat: scalar {chat.get('scalar_passed')}/{chat.get('cases')}, "
              f"vector {chat.get('vector_passed')}/{chat.get('cases')}, "
              f"bit-identical cases {chat.get('identical_cases')}/{chat.get('cases')}")
        for case in ("tools", "long"):
            print(f"  Vector against scalar, {case}: {acceptance.get(case)}")
for k, v in r.get("cpu", {}).items():
    if not k.startswith("threads_"):
        continue
    for i, a in enumerate(v["ask"], 1):
        print(f"  CPU {k.replace('threads_', 'x')} ask {i}: prompt {a['prompt_tok_per_s']:6.1f} tok/s, "
              f"decode {a['decode_tok_per_s']:5.1f} tok/s, total {a['total_s']:5.2f} s, rss {a['footprint_mb']} MB, "
              f"{a['tool_calls']}")
if r.get("cpu", {}).get("chat"):
    chat_lines(f"CPU x{r['cpu']['best_threads']}", r["cpu"]["chat"])
EOF
  echo "Quality gates against float32, bounded by same-weights Mac CPU-int8 controls:"
  for c in tools long; do
    "$PY" "$LAB/tools/lfm3_quality.py" "$RUNS/ref_$c" "$OUT/gpu_$c" --baseline "$MAC/cpu_$c"
  done
  if [[ -d "$OUT/cpu_tools" ]]; then
    "$PY" "$LAB/tools/lfm3_quality.py" "$RUNS/ref_tools" "$OUT/cpu_tools" --baseline "$MAC/cpu_tools"
  fi
} | tee "$APP/results/android_$WEIGHTS$([[ "${SKIP_CPU:-0}" == 1 ]] && echo _gpu).txt"

if [[ "$GPU_SUSTAINED_SECONDS" -gt 0 ]]; then
  "$PY" - "$OUT/latest.json" "$GPU_SUSTAINED_SECONDS" <<'EOF'
import json, sys, math
r = json.load(open(sys.argv[1]))
assert r.get('finished') and not r.get('gpu_error')
g = r['gpu']; s = g['sustained']; samples = s['samples']
assert s.get('finished') and not s.get('failed') and s['elapsed_s'] >= int(sys.argv[2])
assert samples and len(samples) % 4 == 0
for i, x in enumerate(samples):
    assert x['automatic'] == [False, True, True, False][i % 4]
    assert x['differing_final_logits'] == x['non_finite'] == 0
    if x['profiled']:
        assert x['q4_dispatches_per_token'] == 153
        assert x['vector_head_dispatches_per_token'] == int(x['automatic'])
        assert x['scalar_head_dispatches_per_token'] == int(not x['automatic'])
    assert all(x[k] is None or math.isfinite(x[k]) for k in ('gpu_ms_per_token', 'q4_ms_per_token', 'q8_ms_per_token', 'other_ms_per_token', 'wall_ms_per_token'))
for case in ('tools', 'long'):
    checks = g['sustained_controls'][case]
    assert set(checks) == {'hidden.f32', 'logits.f32', 'gen.i32', 'gen_logits.f32'} and all(checks.values())
print('Sustained gate passed: duration, selected adopted kernels, finite timings, final logits in every block, and tools/long old-GEMM controls')
EOF
fi

if [[ "${Q8_VECTOR_ACCEPTANCE:-0}" == 1 ]]; then
  "$PY" - "$OUT/latest.json" "${Q8_VECTOR_DEFAULT_ACCEPTANCE:-0}" <<'EOF'
import json, sys
from collections import Counter
r = json.load(open(sys.argv[1]))
assert r.get("finished") and not r.get("gpu_error"), r.get("gpu_error")
g = r["gpu"]
assert g["q8_vector_acceptance"] is True
if sys.argv[2] == "1":
    assert g["q8_vector_default_acceptance"] is True
    assert g["q8_vector_head_default_eligible"] is True
    assert g["q8_vector_default_probe"] == {"vector_dispatches": 1, "scalar_dispatches": 0}
a = g["vector_acceptance"]
c = a["chat"]
assert c["cases"] == c["scalar_passed"] == c["vector_passed"] == c["identical_cases"] == 16, c
assert g["chat"]["passed"] == g["chat"]["cases"] == 16 and g["chat"]["identical_logits"]
for case in ("tools", "long"):
    assert set(a[case]) == {"hidden.f32", "logits.f32", "gen.i32", "gen_logits.f32"}
    assert all(a[case].values()), (case, a[case])
for samples, diff in ((g["q8_vector_sweep"], "differing_outputs"),
                      (g["profile"]["q8_vector_sweep"], "differing_logits")):
    assert Counter(s["q8_vector"] for s in samples) == {False: 6, True: 6}
    assert all(s[diff] == 0 and s["non_finite"] == 0 for s in samples)
print("Q8 vector acceptance passed: paired chat logits, candidate chat/state reuse, tools/long dumps, and repeat sweeps")
EOF
fi
