#!/usr/bin/env bash
# Builds the app and runs its command-line checks on the test images (no camera, no window).
# Results: logs/selftest.txt, logs/*.json, and logs/compare.txt (Swift vs the Python/ONNX engine).
cd "$(dirname "$0")/.."
[ -d /Applications/Xcode.app ] && export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
mkdir -p logs
{ sw_vers; xcodebuild -version; uname -m; sysctl -n machdep.cpu.brand_string; } > logs/system.txt 2>&1
echo "building..."
if ! xcodebuild -project FaceHandTrack.xcodeproj -target FaceHandTrack -configuration Release \
       SYMROOT="$PWD/build" build > logs/build.txt 2>&1; then
  { echo "BUILD FAILED"; grep -E "error:" logs/build.txt | sort -u; } > logs/selftest.txt
  cat logs/selftest.txt
  exit 1
fi
BIN=build/Release/FaceHandTrack.app/Contents/MacOS/FaceHandTrack
PY=../onnx/.venv/bin/python   # the ONNX version's environment: makes the test images and the Python reference
if [ ! -d tests/sequence ] && [ -x "$PY" ]; then
  echo "making test images..."
  "$PY" tests/make_test_frames.py > logs/make_frames.txt 2>&1
fi
echo "running checks..."
{
  "$BIN" --selftest-hands tests/hand_canvases logs/hands.json
  "$BIN" --selftest-sequence tests/sequence logs/sequence.json
  "$BIN" --selftest-sequence tests/faces logs/faces.json
  "$BIN" --benchmark logs/benchmark.json
  echo "SELFTEST DONE"
} > logs/selftest.txt 2>&1
tail -3 logs/selftest.txt
if [ -x "$PY" ]; then
  echo "comparing with the Python (ONNX) engine..."
  "$PY" tests/compare_selftest.py logs/hands.json logs/sequence.json logs/faces.json > logs/compare.txt 2>&1
  tail -5 logs/compare.txt
fi
