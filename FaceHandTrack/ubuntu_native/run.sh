#!/usr/bin/env bash
# Start the viewer. Extra arguments are passed through, e.g. ./run.sh --camera realsense:1 --no-mirror
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || ./setup.sh
exec .venv/bin/python app/viewer.py "$@"
