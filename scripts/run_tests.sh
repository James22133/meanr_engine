#!/usr/bin/env bash
# Simple test runner that sets PYTHONPATH to the repository root
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

# Warn if key packages are missing
missing=()
for pkg in hmmlearn pykalman pandas numpy; do
    python - <<EOF
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("$pkg") else 1)
EOF
    if [ $? -ne 0 ]; then
        missing+=("$pkg")
    fi
done

if [ ${#missing[@]} -ne 0 ]; then
    echo "WARNING: Missing packages: ${missing[*]}"
    echo "Run 'pip install -r requirements.txt' first."
fi

pytest "$@"
