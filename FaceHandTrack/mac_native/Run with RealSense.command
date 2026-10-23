#!/usr/bin/env bash
# Double-click in Finder: builds FaceHandTrack and runs it on the Intel RealSense colour (2-D) stream.
# macOS shows the camera's colour sensor to apps only through Intel's SDK (librealsense), and lets the SDK
# take the camera over USB only as administrator, so this asks for your Mac password once per launch.
set -uo pipefail
cd "$(dirname "$0")"
[ -d /Applications/Xcode.app ] && export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
mkdir -p logs
if ! ls /opt/homebrew/lib/librealsense2*.dylib >/dev/null 2>&1 && ! ls /usr/local/lib/librealsense2*.dylib >/dev/null 2>&1; then
  echo "Intel's RealSense SDK is missing. Install it once with:  brew install librealsense"
  exit 1
fi
if pgrep -xq realsense-viewer; then
  echo "realsense-viewer is running and holds the camera: quit it, then double-click this again."
  exit 1
fi
echo "Building FaceHandTrack..."
if ! xcodebuild -project FaceHandTrack.xcodeproj -target FaceHandTrack -configuration Release \
       SYMROOT="$PWD/build" build > logs/build.txt 2>&1; then
  grep -E "error:" logs/build.txt | sort -u | head -20
  echo "Build failed. Full log: logs/build.txt"
  exit 1
fi
rm -rf FaceHandTrack.app && cp -R build/Release/FaceHandTrack.app .
echo
echo "Starting FaceHandTrack on the RealSense colour camera. Type your Mac password if asked (it is not shown)."
echo "Quit the app (Cmd+Q) to stop; the camera then works as usual again."
sudo FaceHandTrack.app/Contents/MacOS/FaceHandTrack --realsense 2>&1 | tee logs/realsense_run.txt
