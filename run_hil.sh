#!/bin/bash

# Helper script to run HIL scripts with the correct Python environment
# Usage: ./run_hil.sh mixed|de|bo [num_experiments]

VENV_PYTHON="/home/waleed/python-env/bin/python"
NUM_EXP="${2:-1}"  # Default to 1 if not provided

case "$1" in
    mixed)
        echo "Running Mixed HIL (DE vs BO)..."
        $VENV_PYTHON mixed_hil_pid/scripts/mixed_hil_rerun.py $NUM_EXP
        ;;
    de)
        echo "Running DE HIL..."
        $VENV_PYTHON mixed_hil_pid/scripts/de_hil_rerun.py $NUM_EXP
        ;;
    bo)
        echo "Running BO HIL..."
        $VENV_PYTHON mixed_hil_pid/scripts/bo_hil_rerun.py $NUM_EXP
        ;;
    *)
        echo "Usage: $0 {mixed|de|bo} [num_experiments]"
        echo "  mixed - Run Mixed HIL (DE vs BO)"
        echo "  de    - Run DE-only HIL"
        echo "  bo    - Run BO-only HIL"
        echo "  num_experiments - Number of experiments to run (default: 1)"
        exit 1
        ;;
esac
