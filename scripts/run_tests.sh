#!/usr/bin/env bash
# Simple test runner that sets PYTHONPATH to the repository root
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
pytest "$@"
