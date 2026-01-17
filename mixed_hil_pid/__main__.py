#!/usr/bin/env python3
"""
Entry point for the mixed_hil_pid package.

This module allows the package to be run as:
    python -m mixed_hil_pid

By default, this launches the TUI (Text User Interface) for configuring
and running PID optimization experiments.

For more information about the TUI, see TUI_README.md
"""

from mixed_hil_pid.apps.tui import main

if __name__ == '__main__':
    main()
