#!/usr/bin/env bash
# One-time setup for re-exporting the models: PyTorch environment + EdgeTAM code and checkpoint.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install -U pip wheel setuptools
.venv/bin/python -m pip install -r requirements.txt
if [ ! -d EdgeTAM ]; then
  git clone --depth 1 https://github.com/facebookresearch/EdgeTAM.git   # code + checkpoints/edgetam.pt (Apache-2.0)
fi
./get_models.sh
echo "Ready. Next:  .venv/bin/python export_onnx.py && .venv/bin/python validate_onnx.py"
