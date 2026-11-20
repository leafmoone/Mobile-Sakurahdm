#!/usr/bin/env bash
# Usage: train_hdm.sh CONFIG.yaml [--stage OVERLAY.yaml ...] [--resume CHECKPOINT]
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m sakurahdm.train_hdm "$@"
