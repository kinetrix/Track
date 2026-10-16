#!/usr/bin/env bash
# Debug helper: replays recordings/session through the engine on this Mac and writes recordings/replay_log.txt
cd "$(dirname "$0")" && .venv/bin/python tests/replay.py recordings/session --start 130 --end 200 > recordings/replay_log.txt 2>&1; echo done >> recordings/replay_log.txt
