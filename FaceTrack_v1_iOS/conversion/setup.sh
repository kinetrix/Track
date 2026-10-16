#!/usr/bin/env bash
# One-time setup for re-running the conversions (the converted models already ship in ios/FaceTrack/Models).
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements.txt
if [ ! -d EdgeTAM ]; then
  git clone --depth 1 https://github.com/facebookresearch/EdgeTAM.git   # code + checkpoints/edgetam.pt (Apache-2.0)
fi
echo "Ready. Next:  source .venv/bin/activate && python convert_yunet.py && python export_edgetam.py"
