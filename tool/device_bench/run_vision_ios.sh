#!/usr/bin/env bash
# Measure a prebuilt VL bundle on a specific physical iPhone. Local device
# testing only; does not publish models or upload an App Store build.
# VL_BUNDLE=/path/to/pack-vl IMAGE=/path/to/card.png tool/device_bench/run_vision_ios.sh DEVICE_ID
# BACKEND=cpu|gpu, RUNS=3, EXPECTED='...', PROMPT='...', WAIT_SECONDS=600.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP="$ROOT/tool/device_bench"
DEVICE="${1:?pass the physical device identifier}"
VL_BUNDLE="${VL_BUNDLE:?set VL_BUNDLE to the pack-vl output}"
IMAGE="${IMAGE:?set IMAGE to a local PNG/JPEG}"
FLUTTER="${FLUTTER:-flutter}"
BACKEND="${BACKEND:-gpu}"
wait_seconds="${WAIT_SECONDS:-600}"
[[ "$wait_seconds" =~ ^[1-9][0-9]*$ ]] || { echo 'WAIT_SECONDS must be positive' >&2; exit 2; }
[[ "${RUNS:-3}" =~ ^[1-9][0-9]*$ ]] || { echo 'RUNS must be positive' >&2; exit 2; }
OUT="${OUT:-$ROOT/build/device_bench/vision/iphone-$BACKEND-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$BACKEND" == cpu || "$BACKEND" == gpu ]] || { echo 'BACKEND must be cpu or gpu' >&2; exit 2; }
[[ -f "$VL_BUNDLE/vl_bundle.json" && -f "$IMAGE" ]] || { echo 'bundle manifest or image missing' >&2; exit 2; }
mkdir -p "$OUT"
BUNDLE_ID=com.noguwo.dartllmlab.lfm3Bench
container=(--device "$DEVICE" --domain-type appDataContainer --domain-identifier "$BUNDLE_ID")
(cd "$APP" && "$FLUTTER" pub get && "$FLUTTER" build ios --release)
xcrun devicectl --timeout 60 device install app --device "$DEVICE" "$APP/build/ios/iphoneos/Runner.app"
manifest_sha="$(shasum -a 256 "$VL_BUNDLE/vl_bundle.json" | awk '{print $1}')"
remote_bundle="vision-bundle-${manifest_sha:0:12}"
if xcrun devicectl --timeout 30 device copy from "${container[@]}" \
  --source "Documents/$remote_bundle/vl_bundle.json" --destination "$OUT/remote-manifest.json" >/dev/null 2>&1 &&
  cmp -s "$OUT/remote-manifest.json" "$VL_BUNDLE/vl_bundle.json"; then
  echo 'Reusing device VL bundle; the loader will verify all payload hashes.'
else
  xcrun devicectl --timeout 600 device copy to "${container[@]}" \
    --source "$VL_BUNDLE" --destination "Documents/$remote_bundle"
fi
xcrun devicectl --timeout 30 device copy to "${container[@]}" --source "$IMAGE" --destination Documents/vision-input.image
# Preserve the earlier text benchmark configuration. A missing configuration
# is restored as {}, the text benchmark's defaults.
if ! xcrun devicectl --timeout 30 device copy from "${container[@]}" \
  --source Documents/bench_config.json --destination "$OUT/previous-config.json" >/dev/null 2>&1; then
  echo '{}' > "$OUT/previous-config.json"
fi
restore() {
  if ! xcrun devicectl --timeout 30 device copy to "${container[@]}" \
    --source "$OUT/previous-config.json" --destination Documents/bench_config.json >/dev/null 2>&1; then
    echo "Could not restore the previous benchmark configuration; its backup is $OUT/previous-config.json" >&2
    return 1
  fi
}
trap restore EXIT
previous=''
if xcrun devicectl --timeout 30 device copy from "${container[@]}" \
  --source Documents/bench/vision-latest.json --destination "$OUT/previous-result.json" >/dev/null 2>&1; then
  previous="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["started"])' "$OUT/previous-result.json")"
fi
python3 - "$OUT/config.json" "$remote_bundle" "$BACKEND" "${RUNS:-3}" \
  "${PROMPT:-画像に書かれた文字を、そのまま書き起こしてください。}" "${EXPECTED:-東京 123}" "$ROOT" <<'PY'
import hashlib,json,subprocess,sys
out,bundle,backend,runs,prompt,expected,root=sys.argv[1:]
config=dict(mode='vision',bundle=bundle,image='vision-input.image',backend=backend,
            runs=int(runs),prompt=prompt,expected_text=expected,threads=4,
            context_length=512,max_new_tokens=32)
assert config['runs'] > 0
config['build']=dict(mode='release',git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                    dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)),
                    vision_bench_sha256=hashlib.sha256(open(root+'/lib/src/vision_bench.dart','rb').read()).hexdigest())
with open(out,'w') as f: json.dump(config,f,ensure_ascii=False)
PY
xcrun devicectl --timeout 30 device copy to "${container[@]}" --source "$OUT/config.json" --destination Documents/bench_config.json
xcrun devicectl --timeout 30 device process launch --device "$DEVICE" --terminate-existing "$BUNDLE_ID"
wait_start=$SECONDS
deadline=$((wait_start + wait_seconds))
while ((SECONDS < deadline)); do
  sleep 5
  if ! xcrun devicectl --timeout 15 device copy from "${container[@]}" \
    --source Documents/bench/vision-latest.json --destination "$OUT/latest.json" >/dev/null 2>&1; then
    continue
  fi
  status="$(python3 - "$OUT/latest.json" "$previous" <<'PY'
import json,sys
try:
 r=json.load(open(sys.argv[1]))
 print('old' if r['started']==sys.argv[2] else 'done' if r.get('finished') else r.get('phase','running'))
except (ValueError,OSError): print('reading')
PY
)"
  echo "$((SECONDS - wait_start)) s: $status"
  if [[ "$status" == done ]]; then
    python3 - "$OUT/latest.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); print('passed:',r.get('passed'))
for turn in r.get('turns',[]):
 print(turn['case'],repr(turn['text']),'first text',turn['first_text_s'],'s; decode',turn['decode_tokens_per_s'],'tokens/s')
if not r.get('passed'): raise SystemExit(1)
PY
    echo "Result: $OUT/latest.json"
    exit 0
  fi
done
echo "No completed record; inspect $OUT/latest.json and device crash diagnostics. Do not treat an incomplete run as a performance result." >&2
exit 1
