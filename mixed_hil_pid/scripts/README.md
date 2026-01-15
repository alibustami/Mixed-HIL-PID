# Human-in-the-Loop PID Optimization Scripts

This directory contains three main executable scripts for PID controller optimization using human feedback.

## 📁 Scripts Overview

### 1. **mixed_hil.py** - Mixed HIL (DE vs BO)
Compares candidates from both Differential Evolution (DE) and Bayesian Optimization (BO) side-by-side.

**Run:**
```bash
# Option 1: Using helper script
./run_hil.sh mixed

# Option 2: Direct with venv Python
/home/waleed/python-env/bin/python mixed_hil_pid/scripts/mixed_hil.py
```

**GUI Options:**
- **Prefer A (DE)** - Learn preference toward DE solution
- **Prefer B (BO)** - Learn preference toward BO solution  
- **TIE (Refine)** - Both good, converge toward average
- **REJECT Both** - Both bad, expand search space

**Output:** `logs/mixed/mixed_YYYYMMDD-HHMMSS/`

---

### 2. **de_hil.py** - DE-Only HIL
Uses only Differential Evolution with human feedback.

**Run:**
```bash
# Using helper script
./run_hil.sh de

# Or direct
/home/waleed/python-env/bin/python mixed_hil_pid/scripts/de_hil.py
```

**GUI Options:**
- **ACCEPT (Refine)** - Good solution, refine around it
- **REJECT (Expand)** - Bad solution, expand search

**Output:** `logs/DE_HIL/de_hil_YYYYMMDD-HHMMSS/`

---

### 3. **bo_hil.py** - BO-Only HIL
Uses only Bayesian Optimization with human feedback.

**Run:**
```bash
# Using helper script
./run_hil.sh bo

# Or direct
/home/waleed/python-env/bin/python mixed_hil_pid/scripts/bo_hil.py
```

**GUI Options:**
- **ACCEPT (Refine)** - Good solution, refine around it
- **REJECT (Expand)** - Bad solution, expand search

**Output:** `logs/BO_HIL/bo_hil_YYYYMMDD-HHMMSS/`

---

## 🛠️ Supporting Modules

- **config.py** - All configuration constants
- **metrics.py** - PID performance calculations
- **logging_utils.py** - CSV and pickle logging
- **simulation.py** - PyBullet simulation wrapper
- **preference_model.py** - Preference learning (Mixed HIL only)
- **gui.py** - GUI components

---

## 🎯 Configuration

Edit `config.py` to change:
- PID bounds: `PID_BOUNDS`
- Performance targets: `PID_MAX_OVERSHOOT_PCT`, `PID_MAX_RISE_TIME`, etc.
- Simulation steps: `SIMULATION_STEPS`
- Max iterations: `MAX_ITERATIONS`

---

## 📊 Output Files

Each run creates:
- `iteration_log.csv` - Detailed iteration data
- `iteration_log.pkl` - History data for analysis
- `config.yaml` - Run configuration
- `best_results.json` - Best solution found

---

## ⚙️ Installation

**Install dependencies:**
```bash
pip install -r ../requirements.txt
```

Or manually:
```bash
pip install pybullet numpy matplotlib seaborn scipy scikit-learn
```

**Requirements:**
- Python 3.7+
- PyBullet - Robot simulation
- NumPy - Numerical computing
- Matplotlib, Seaborn - Visualization
- scikit-learn, scipy - Bayesian optimization
- tkinter - GUI (usually pre-installed)
