#!/usr/bin/env bash
# Debug helper: runs the viewer and saves (frame before, frame after, landmarks) each time a tracked hand is lost.
cd "$(dirname "$0")" && ./start.sh --debug-hands recordings/lost
