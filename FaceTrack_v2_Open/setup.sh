#!/usr/bin/env bash
# One-time setup: a private Python environment in .venv with numpy, OpenCV and ONNX Runtime.
# Everything stays inside this folder (no admin rights, nothing installed system-wide).
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3 /opt/homebrew/bin/python3 /usr/local/bin/python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY="$(command -v "$c")"; break
  fi
done

if [ -n "$PY" ]; then
  echo "Using $PY ($("$PY" --version))"
  "$PY" -m venv .venv
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
else
  # macOS ships Python 3.9, which is too old for current ONNX Runtime. Get a private 3.12 with uv.
  echo "Python 3.10+ not found. Downloading a private Python 3.12 into .tools/ (uv, by Astral)..."
  mkdir -p .tools
  export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
  if [ ! -x .tools/uv ]; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$PWD/.tools" sh
  fi
  .tools/uv venv --python 3.12 .venv
  .tools/uv pip install --python .venv/bin/python -r requirements.txt
fi
.venv/bin/python -c "import onnxruntime as o, cv2; print('ONNX Runtime', o.__version__, o.get_available_providers()); print('OpenCV', cv2.__version__)"
echo "Setup done. Start with ./start.sh (or double-click 'Start FaceTrack.command')."
