#!/usr/bin/env bash
# Copy a model bundle into the example app's data, where the app looks for
# it (lib/src/chat_protocol.dart, ModelLocation).
#
#   example/tool/install_model.sh ios       BUNDLE_DIR [DEVICE_ID]
#   example/tool/install_model.sh android   BUNDLE_DIR
#   example/tool/install_model.sh simulator BUNDLE_DIR
#
# BUNDLE_DIR is the LFM3-1.7B-QAD-Q4 directory written by
# `dianoia import-gguf`. Install the app and start it once first, so that
# its data directory exists. On an iPhone the Files app or Finder can copy
# the bundle into the app's documents just as well.
set -euo pipefail

PLATFORM="${1:?ios, android or simulator}"
BUNDLE="${2:?the model bundle directory}"
NAME=LFM3-1.7B-QAD-Q4
IOS_ID=dev.atrac613.dianoiaExample
ANDROID_ID=dev.atrac613.dianoia_example
[[ -f "$BUNDLE/lfm3-q4.cache" ]] || { echo "$BUNDLE has no lfm3-q4.cache" >&2; exit 1; }

case "$PLATFORM" in
  ios)
    DEVICE="${3:-$(xcrun devicectl list devices 2>/dev/null |
      awk '/iPhone/ && /connected/ && / physical/ {for (i = 1; i <= NF; i++) if ($i ~ /^[0-9A-F]{8}-[0-9A-F]{16}$/) print $i}' |
      head -1)}"
    [[ -n "$DEVICE" ]] || { echo "no connected iPhone" >&2; exit 1; }
    xcrun devicectl device copy to --device "$DEVICE" --domain-type appDataContainer \
      --domain-identifier "$IOS_ID" --source "$BUNDLE" --destination "Documents/$NAME" >/dev/null
    ;;
  android)
    REMOTE="/sdcard/Android/data/$ANDROID_ID/files"
    # A directory adb creates belongs to the shell user, and the app cannot
    # read it; the app creates its own on first start.
    owner="$(adb shell "stat -c %U '$REMOTE' 2>/dev/null" | tr -d '\r')"
    [[ "$owner" == u0_a* ]] || { echo "start the app once first" >&2; exit 1; }
    for f in "$BUNDLE"/*; do
      adb push "$f" "$REMOTE/$NAME/$(basename "$f")" >/dev/null
    done
    ;;
  simulator)
    container="$(xcrun simctl get_app_container booted "$IOS_ID" data)"
    mkdir -p "$container/Documents/$NAME"
    cp "$BUNDLE"/* "$container/Documents/$NAME/"
    ;;
  *)
    echo "unknown platform $PLATFORM" >&2
    exit 2
    ;;
esac
echo "copied $BUNDLE to the $PLATFORM app; restart the app"
