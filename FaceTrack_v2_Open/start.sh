#!/usr/bin/env bash
# Run the Mac viewer. Extra arguments are passed through, e.g. ./start.sh --accel cpu
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || ./setup.sh
exec .venv/bin/python app/mac_viewer.py "$@"
