#!/usr/bin/env bash
# One-time setup on Ubuntu 22.04 / 24.04 (x86_64 PC or laptop).
#   ./setup.sh             everything (asks for your password once, for apt and the camera rules)
#   ./setup.sh --cpu       force the CPU-only PyTorch build even if an NVIDIA GPU is present
#   ./setup.sh --no-udev   skip the RealSense udev rules
# Python packages go into .venv inside this folder; nothing else is installed system-wide
# apart from a few apt packages (if missing) and the RealSense udev rules file.
set -euo pipefail
cd "$(dirname "$0")"
CPU_ONLY=0; UDEV=1
for a in "$@"; do
  case "$a" in
    --cpu) CPU_ONLY=1 ;;
    --no-udev) UDEV=0 ;;
    *) echo "unknown option $a"; exit 1 ;;
  esac
done

echo "== 1/6 system packages"
need=()
python3 -c 'import venv, ensurepip' >/dev/null 2>&1 || need+=(python3-venv python3-pip)
command -v git >/dev/null || need+=(git)
command -v curl >/dev/null || need+=(curl)
[ -e /usr/lib/x86_64-linux-gnu/libGL.so.1 ] || need+=(libgl1)
if [ ${#need[@]} -gt 0 ]; then
  echo "installing: ${need[*]} (needs sudo)"
  sudo apt-get update -y && sudo apt-get install -y "${need[@]}"
fi
python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ needed (Ubuntu 22.04 or newer)"'

echo "== 2/6 Python environment (.venv)"
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip

echo "== 3/6 PyTorch"
cuda_ok() { .venv/bin/python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; }
if [ "$CPU_ONLY" = 0 ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  .venv/bin/python -m pip install torch torchvision            # the standard Linux build includes CUDA
  # The newest build needs a recent driver; with an older driver, builds for older CUDA versions still work.
  # cu118 (last in PyTorch 2.7.1) runs on any driver from 450.80 on, e.g. the 470 series.
  for spec in "cu128 torch torchvision" "cu126 torch torchvision" "cu118 torch==2.7.1 torchvision==0.22.1"; do
    cuda_ok && break
    set -- $spec
    cu=$1; shift
    echo "this PyTorch build cannot use your driver; trying the $cu build"
    .venv/bin/python -m pip uninstall -y torch torchvision >/dev/null 2>&1 || true
    .venv/bin/python -m pip install "$@" --index-url "https://download.pytorch.org/whl/$cu" || true
  done
  cuda_ok || echo "WARNING: PyTorch still cannot use the GPU (update the NVIDIA driver: sudo ubuntu-drivers install). The app will run on the CPU."
else
  echo "No working NVIDIA driver (or --cpu): installing the smaller CPU-only build"
  .venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "== 4/6 other packages"
.venv/bin/python -m pip install -r requirements.txt

echo "== 5/6 EdgeTAM code + checkpoint (Meta, Apache-2.0)"
if [ ! -f third_party/EdgeTAM/checkpoints/edgetam.pt ]; then
  rm -rf third_party/EdgeTAM
  git clone --depth 1 https://github.com/facebookresearch/EdgeTAM.git third_party/EdgeTAM
fi

echo "== 6/6 RealSense camera access"
RULES=/etc/udev/rules.d/99-realsense-libusb.rules
if [ "$UDEV" = 1 ] && [ ! -f "$RULES" ]; then
  echo "installing the RealSense udev rules (lets the app open the camera without sudo; needs sudo)"
  curl -fsSL https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules \
    | sudo tee "$RULES" >/dev/null
  sudo udevadm control --reload-rules && sudo udevadm trigger
  echo "unplug and replug the camera once"
fi

.venv/bin/python - <<'PY'
import cv2, onnx2torch, torch
print("PyTorch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "running on the CPU")
print("OpenCV", cv2.__version__)
try:
    import pyrealsense2 as rs
    devs = [d.get_info(rs.camera_info.name) + " (serial " + d.get_info(rs.camera_info.serial_number) + ")"
            for d in rs.context().query_devices()]
    print("RealSense:", ", ".join(devs) if devs else "none plugged in right now")
except Exception as e:
    print("RealSense SDK not usable:", e)
PY
echo "Setup done. Start with ./run.sh"
