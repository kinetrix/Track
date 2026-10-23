#!/usr/bin/env bash
# One-time setup on Linux (OrbStack machine, Ubuntu/Debian PC, robot computer).
# Uses its own environment folder (.venv-linux) so it never touches the Mac's .venv,
# even when this folder is shared between macOS and Linux.
set -euo pipefail
cd "$(dirname "$0")"
if ! python3 -c 'import venv, ensurepip' >/dev/null 2>&1; then
  echo "Installing python3-venv (needs sudo)..."
  sudo apt-get update -y && sudo apt-get install -y python3-venv python3-pip
fi
python3 -m venv .venv-linux
.venv-linux/bin/python -m pip install -q --upgrade pip
# headless OpenCV: no window system needed (the picture goes to a browser instead)
.venv-linux/bin/python -m pip install "numpy>=1.24" "onnxruntime>=1.20" "opencv-python-headless>=4.9"
.venv-linux/bin/python -c "import onnxruntime as o, cv2, platform; print('Linux', platform.machine(), '| ONNX Runtime', o.__version__, o.get_available_providers(), '| OpenCV', cv2.__version__)"
echo "Setup done. Run: .venv-linux/bin/python app/headless_viewer.py --help"
