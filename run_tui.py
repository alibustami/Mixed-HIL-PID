#!/usr/bin/env python3
"""
Quick launcher script for the HIL-PID TUI.

Usage: python run_tui.py
   or: ./run_tui.py
"""

import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from mixed_hil_pid.apps.tui import main

if __name__ == "__main__":
    main()
