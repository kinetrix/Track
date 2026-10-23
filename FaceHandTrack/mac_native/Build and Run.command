#!/usr/bin/env bash
# Double-click in Finder: builds the Mac app with Xcode's command-line tools, then starts it.
# (Or open FaceHandTrack.xcodeproj in Xcode and press Run.)
set -uo pipefail
cd "$(dirname "$0")"
[ -d /Applications/Xcode.app ] && export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
mkdir -p logs
echo "Building FaceHandTrack (the first build also compiles the Core ML models, about a minute)..."
if xcodebuild -project FaceHandTrack.xcodeproj -target FaceHandTrack -configuration Release \
     SYMROOT="$PWD/build" build > logs/build.txt 2>&1; then
  rm -rf FaceHandTrack.app && cp -R build/Release/FaceHandTrack.app .
  echo "Built FaceHandTrack.app (next time just double-click it). Starting it now."
  echo "macOS asks once for camera access: click Allow."
  open FaceHandTrack.app
else
  grep -E "error:" logs/build.txt | sort -u | head -20
  echo
  echo "Build failed. Full log: logs/build.txt"
fi
