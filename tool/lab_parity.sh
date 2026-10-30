#!/usr/bin/env bash
# Check that dianoia computes exactly what dart-llm-lab computes.
#
# dianoia started as an import of the lab's engine, and while it is cleaned
# up into a library every change must keep the arithmetic: this builds the
# lab's CLI and kernels and dianoia's CLI bundle, runs both on the lab's
# reference prompts (runs/lfm3/ref_<case>/ids.i32), and requires the dumps
# (hidden states, logits, greedy tokens) to match byte for byte, on the CPU
# Q8 kernels and on the GPU with Q8 and Q4 weights. macOS on Apple silicon
# only (Metal).
#
#   tool/lab_parity.sh [chat long]
#   LAB=/path/to/dart-llm-lab MODEL=/path/to/LFM3-1.7B tool/lab_parity.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB="${LAB:-$ROOT/../dart-llm-lab}"
MODEL="${MODEL:-$LAB/models/LFM3-1.7B}"
OUT="$ROOT/build/lab_parity"
CASES=("$@")
((${#CASES[@]})) || CASES=(chat long)

for c in "${CASES[@]}"; do
  if [[ ! -f "$LAB/runs/lfm3/ref_$c/ids.i32" ]]; then
    echo "missing $LAB/runs/lfm3/ref_$c (run the lab's scripts/lfm3_check.sh)" >&2
    exit 1
  fi
done

mkdir -p "$OUT/lab"
# The lab's own build commands (scripts/build_native.sh, build_metal.sh).
clang -x c -O3 -mcpu=native -ffp-contract=off -fno-fast-math -shared -fPIC \
  -DACCELERATE_NEW_LAPACK "$LAB/stage5_native/c/lab_native.c" \
  -framework Accelerate -o "$OUT/lab/liblab_native.dylib"
clang -x objective-c -fobjc-arc -O2 -shared -fPIC \
  "$LAB/stage5_native/metal/lab_metal.m" -framework Metal -framework Foundation \
  -o "$OUT/lab/liblab_metal.dylib"
(cd "$LAB/lfm3" && dart pub get >/dev/null && dart compile exe bin/lfm3.dart -o "$OUT/lab/lfm3" >/dev/null)
(cd "$ROOT" && dart build cli -o "$OUT/dianoia" >/dev/null)

LAB_CLI=("$OUT/lab/lfm3")
LAB_NATIVE=(--native-lib="$OUT/lab/liblab_native.dylib")
DN_CLI=("$OUT/dianoia/bundle/bin/dianoia")

status=0
compare() {
  for f in hidden.f32 logits.f32 gen.i32 gen_logits.f32; do
    if cmp -s "$1/$f" "$2/$f"; then
      echo "  $f identical"
    else
      echo "  $f DIFFERS" >&2
      status=1
    fi
  done
}

for c in "${CASES[@]}"; do
  ref="$LAB/runs/lfm3/ref_$c"
  max_new="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['max_new_tokens'])" "$ref/meta.json")"
  common=(run --model="$MODEL" --ids="$ref/ids.i32" --max-new="$max_new")

  echo "== $c: CPU Q8 kernels"
  "${LAB_CLI[@]}" "${common[@]}" --out="$OUT/$c/lab_q8" --weights=q8 "${LAB_NATIVE[@]}" >/dev/null
  "${DN_CLI[@]}" "${common[@]}" --out="$OUT/$c/dianoia_q8" --weights=q8 >/dev/null
  compare "$OUT/$c/lab_q8" "$OUT/$c/dianoia_q8"

  for w in q8 q4; do
    echo "== $c: GPU, $w weights"
    "${LAB_CLI[@]}" "${common[@]}" --out="$OUT/$c/lab_gpu_$w" --weights="$w" "${LAB_NATIVE[@]}" \
      --gpu-lib="$OUT/lab/liblab_metal.dylib" >/dev/null
    "${DN_CLI[@]}" "${common[@]}" --out="$OUT/$c/dianoia_gpu_$w" --weights="$w" --gpu >/dev/null
    compare "$OUT/$c/lab_gpu_$w" "$OUT/$c/dianoia_gpu_$w"
  done
done
exit $status
