#!/usr/bin/env bash
# Starts the built app from Terminal so its console output goes to logs/run.txt
# (camera access is then asked for Terminal instead of FaceHandTrack). Quit the app to stop.
cd "$(dirname "$0")/.."
APP=build/Release/FaceHandTrack.app
[ -d "$APP" ] || APP=FaceHandTrack.app
mkdir -p logs
"$APP/Contents/MacOS/FaceHandTrack" > logs/run.txt 2>&1
