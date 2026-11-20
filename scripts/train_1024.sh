#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/scripts/train_hdm.sh" "$ROOT/configs/hdm-base.yaml" --stage "$ROOT/configs/hdm-1024.yaml" "$@"
