# Mixed HIL-PID Optimization Suite - TUI

## 🚀 Professional Text User Interface

A beautiful terminal interface for running and configuring PID optimization experiments.

### Features

- **🎯 Approach Selection**: Choose between 5 different optimization approaches
  - Mixed HIL (DE vs BO) - Human-guided comparison
  - DE HIL - Human-guided Differential Evolution
  - BO HIL - Human-guided Bayesian Optimization
  - DE Autorun - Automated DE benchmarking
  - BO Autorun - Automated BO benchmarking

- **⚙️ Configuration Editor**: Edit all optimization parameters in real-time
  - PID bounds (Kp, Ki, Kd)
  - Simulation settings
  - Performance targets
  - Optimization parameters

- **📊 Live Configuration Preview**: View current settings at a glance

### Installation

Install the TUI dependency:
```bash
pip install textual
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### Usage

**Option 1: Using the launcher script**
```bash
/home/waleed/python-env/bin/python run_tui.py
```

**Option 2: As a Python module**
```bash
/home/waleed/python-env/bin/python -m mixed_hil_pid.apps.tui
```

### Keyboard Shortcuts

- **Tab**: Switch between tabs
- **Arrow Keys**: Navigate
- **Enter**: Select/Activate
- **E**: Edit configuration
- **Q**: Quit
- **Ctrl+C**: Force quit
- **Ctrl+S**: Save config (in editor)
- **Esc**: Go back (in editor)

### Screenshots

The TUI features:
- Clean, modern design with color-coded approaches
- Tabbed interface for approaches and configuration
- Interactive configuration editor with validation
- One-click launch of any optimization approach
- Real-time status notifications

### Architecture

```
mixed_hil_pid/apps/
├── __init__.py
└── tui.py          # Main TUI application

Entry points:
├── run_tui.py      # Quick launcher
└── python -m mixed_hil_pid.apps.tui
```

### Approach Cards

1. **🔄 Mixed HIL** (Primary variant)
   - Compare DE and BO side-by-side
   - 4 feedback options

2. **🔷 DE HIL** (Success variant)
   - Human-guided DE
   - Accept/Reject feedback

3. **🔶 BO HIL** (Warning variant)
   - Human-guided BO
   - Accept/Reject feedback

4. **⚡ DE Autorun** (Default variant)
   - Automated DE runs
   - No human interaction

5. **⚡ BO Autorun** (Default variant)
   - Automated BO runs
   - No human interaction

---

**Made with ❤️ using [Textual](https://textual.textualize.io/)**
