#!/usr/bin/env bash
# Builds the app if needed, then prints every camera macOS offers and its video formats (logs/cameras.txt).
cd "$(dirname "$0")/.."
[ -d /Applications/Xcode.app ] && export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
mkdir -p logs
xcodebuild -project FaceHandTrack.xcodeproj -target FaceHandTrack -configuration Release \
  SYMROOT="$PWD/build" build > logs/build.txt 2>&1 || { echo "BUILD FAILED (logs/build.txt)" | tee logs/cameras.txt; exit 1; }
build/Release/FaceHandTrack.app/Contents/MacOS/FaceHandTrack --list-cameras 2>&1 | tee logs/cameras.txt
