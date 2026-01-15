#!/bin/bash

# Helper script to run HIL scripts with the correct Python environment
# Usage: ./run_hil.sh mixed|de|bo

VENV_PYTHON="/home/waleed/python-env/bin/python"

case "$1" in
    mixed)
        echo "Running Mixed HIL (DE vs BO)..."
        $VENV_PYTHON mixed_hil_pid/scripts/mixed_hil.py
        ;;
    de)
        echo "Running DE HIL..."
        $VENV_PYTHON mixed_hil_pid/scripts/de_hil.py
        ;;
    bo)
        echo "Running BO HIL..."
        $VENV_PYTHON mixed_hil_pid/scripts/bo_hil.py
        ;;
    *)
        echo "Usage: $0 {mixed|de|bo}"
        echo "  mixed - Run Mixed HIL (DE vs BO)"
        echo "  de    - Run DE-only HIL"
        echo "  bo    - Run BO-only HIL"
        exit 1
        ;;
esac
