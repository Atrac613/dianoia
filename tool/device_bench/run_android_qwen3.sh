#!/usr/bin/env bash
# Qwen3 through ModelStore on a connected Android device, end to end, as
# run_ios_qwen3.sh does for the iPhone: the device downloads Qwen's Q8_0
# GGUF from Hugging Face into internal storage, imports it, and answers the
# qwen3 mode's prompts on Vulkan and on the CPU (lib/qwen3_check.dart).
# Prints the record; tool/device_bench/results/android_qwen3_<spec>.json
# keeps it. The device must stay awake and unlocked while the app runs.
#
#   tool/device_bench/run_android_qwen3.sh
#   SPEC=1.7b ANDROID_SERIAL=... tool/device_bench/run_android_qwen3.sh
#   REINSTALL=1 tool/device_bench/run_android_qwen3.sh   # download again
#   GPU_PROFILE=1 tool/device_bench/run_android_qwen3.sh # GPU time and memory
#     by kernel and kind instead of the checks (qwen3GpuProfile)
#   GPU_SHAPES=1 tool/device_bench/run_android_qwen3.sh  # the profile, then every
#     decode matrix kind timed in each layout (benchmarkDecodeShapes)
#   VERIFY_COST=1 tool/device_bench/run_android_qwen3.sh # n-token pass times
#     of 0.6B and 1.7B, for speculative decoding (qwen3VerifyCost)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP="$ROOT/tool/device_bench"
PACKAGE=com.noguwo.dartllmlab.lfm3_bench
REMOTE="/sdcard/Android/data/$PACKAGE/files"
SPEC="${SPEC:-0.6b}"
REINSTALL="${REINSTALL:+true}"

[[ -n "$(adb devices | awk 'NR > 1 && $2 == "device"')" ]] || { echo "no authorized adb device" >&2; exit 1; }
JAVA_HOME="${JAVA_HOME:-$(/usr/libexec/java_home 2>/dev/null || true)}"
[[ -n "$JAVA_HOME" ]] || { echo "set JAVA_HOME to a JDK 17+" >&2; exit 1; }
export JAVA_HOME

echo "== build and install on $(adb shell getprop ro.product.model | tr -d '\r')"
(cd "$APP" && flutter pub get >/dev/null && cd android && ./gradlew -q assembleRelease -Ptarget-platform=android-arm64 >/dev/null)
adb install -r "$APP/build/app/outputs/flutter-apk/app-release.apk" >/dev/null

# The external files directory exists once the app has started.
if [[ "$(adb shell "stat -c %U '$REMOTE' 2>/dev/null" | tr -d '\r')" != u0_a* ]]; then
  adb shell rm -rf "/sdcard/Android/data/$PACKAGE"
  adb shell am start -W -n "$PACKAGE/.MainActivity" >/dev/null
  for _ in $(seq 1 30); do
    [[ "$(adb shell "stat -c %U '$REMOTE' 2>/dev/null" | tr -d '\r')" == u0_a* ]] && break
    sleep 1
  done
  adb shell am force-stop "$PACKAGE"
fi

staging="$(mktemp -d)"
trap 'rm -rf "${staging:?}"' EXIT
flag() { [[ -n "${1:-}" ]] && echo true || echo false; }
if [[ -n "${VERIFY_COST:-}" && -n "${GPU_PROFILE:-}${GPU_SHAPES:-}" ]]; then
  echo "VERIFY_COST is a run of its own: unset GPU_PROFILE and GPU_SHAPES" >&2
  exit 1
fi
printf '{"mode": "qwen3", "spec": "%s", "reinstall": %s, "gpu_profile": %s, "gpu_shapes": %s, "verify_cost": %s, "build": "%s"}\n' \
  "$SPEC" "${REINSTALL:-false}" "$(flag "${GPU_PROFILE:-}${GPU_SHAPES:-}")" "$(flag "${GPU_SHAPES:-}")" "$(flag "${VERIFY_COST:-}")" \
  "$(git -C "$ROOT" describe --always --dirty)" >"$staging/bench_config.json"
adb push "$staging/bench_config.json" "$REMOTE/" >/dev/null

started() { python3 -I -c "import json,sys; print(json.load(open(sys.argv[1])).get('started', ''))" "$1" 2>/dev/null || true; }
pull() { adb pull "$REMOTE/bench/qwen3-latest.json" "$1" >/dev/null 2>&1; }
previous=""
if pull "$staging/previous.json"; then previous="$(started "$staging/previous.json")"; fi

echo "== run (keep the device awake and unlocked)"
adb shell input keyevent KEYCODE_WAKEUP >/dev/null
adb shell am start -S -n "$PACKAGE/.MainActivity" >/dev/null

mkdir -p "$APP/results"
# VERIFY_COST times both sizes, whatever SPEC installs first.
if [[ -n "${VERIFY_COST:-}" ]]; then
  out="$APP/results/android_qwen3_verify_cost.json"
else
  out="$APP/results/android_qwen3_$SPEC${GPU_PROFILE:+_profile}${GPU_SHAPES:+_shapes}.json"
fi
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
adb pull "$REMOTE/bench/error.txt" "$staging/error.txt" >/dev/null 2>&1 && cat "$staging/error.txt" >&2
echo "no finished record after 40 min" >&2
pull "$staging/latest.json" && cat "$staging/latest.json"
exit 1
