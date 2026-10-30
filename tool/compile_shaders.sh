#!/usr/bin/env bash
# Compile the Vulkan compute shaders (src/vulkan/shaders/*.comp) to SPIR-V
# word arrays in src/vulkan/spirv/, which lab_vulkan.c includes.
#
# The arrays are committed so that building the package (hook/build.dart)
# needs only a C compiler, never glslc. Run this after editing a shader;
# test/shaders_test.dart fails while src/vulkan/spirv/SOURCES.sha256 does
# not match the shader sources.
#
#   tool/compile_shaders.sh
#   GLSLC=/path/to/glslc tool/compile_shaders.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHADERS="$ROOT/src/vulkan/shaders"
OUT="$ROOT/src/vulkan/spirv"

if [[ -z "${GLSLC:-}" ]]; then
  # The newest NDK's glslc; the NDK ships it for x86_64 hosts only.
  NDK_ROOT="${ANDROID_NDK:-}"
  if [[ -z "$NDK_ROOT" ]]; then
    SDK="${ANDROID_HOME:-$HOME/Library/Android/sdk}"
    NDK_ROOT="$(find "$SDK/ndk" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -V | tail -1)"
  fi
  host="$(uname -s | tr '[:upper:]' '[:lower:]')"
  GLSLC="$NDK_ROOT/shader-tools/$host-x86_64/glslc"
fi
if [[ ! -x "$GLSLC" ]]; then
  echo "glslc not found (set GLSLC or ANDROID_NDK): $GLSLC" >&2
  exit 1
fi

mkdir -p "$OUT"
rm -f "$OUT"/*.inc
for shader in "$SHADERS"/*.comp; do
  name="$(basename "$shader" .comp)"
  "$GLSLC" -O --target-env=vulkan1.2 -mfmt=c -o "$OUT/$name.inc" "$shader"
done

# Sorted by file name, as test/shaders_test.dart recomputes it.
(cd "$SHADERS" && LC_ALL=C ls -1 -- *.comp *.glsl | LC_ALL=C sort | while read -r f; do
  printf '%s  %s\n' "$(shasum -a 256 "$f" | cut -d' ' -f1)" "$f"
done) > "$OUT/SOURCES.sha256"
echo "$("$GLSLC" --version | head -1)" > "$OUT/GLSLC_VERSION"
echo "compiled $(ls "$OUT"/*.inc | wc -l | tr -d ' ') shaders into $OUT"
