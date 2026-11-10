#!/bin/bash
# Creates .venv in this folder and installs pyrealsense2 + OpenCV (Apple Silicon macOS).
set -e
cd "$(dirname "$0")"

python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install numpy opencv-python

if ./.venv/bin/pip install pyrealsense2; then
  echo "Installed pyrealsense2"
else
  echo "Official wheel not available for this Mac/Python; trying pyrealsense2-macosx"
  ./.venv/bin/pip install pyrealsense2-macosx
fi

./.venv/bin/python -c "import pyrealsense2 as rs; print('pyrealsense2 OK, version', rs.__version__)"
echo "Setup done."
