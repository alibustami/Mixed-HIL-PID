#!/usr/bin/env python3
"""
BO HIL Rerun: Execute Multiple BO HIL Experiments

Runs multiple human-in-the-loop PID optimization experiments using Bayesian Optimization
with user feedback to refine or expand the search space.

Each experiment consists of multiple iterations. This script allows you to run N experiments
sequentially, with state management for resumption if interrupted.

Usage: 
    python bo_hil_rerun.py [num_experiments]
    
Example:
    python bo_hil_rerun.py 5  # Run 5 experiments
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mixed_hil_pid.config_loader import load_config
from mixed_hil_pid.simulator import SimulationManager
from mixed_hil_pid.experiment_executor import ExperimentExecutor
from mixed_hil_pid.utils import load_state, save_state, resolve_state, generate_timestamp


# Load configuration
CONFIG = load_config()

# State management path
STATE_PATH = Path("logs/BO_HIL") / "rerun_state.json"


def main(num_experiments=1):
    """
    Run multiple BO HIL experiments.
    
    Args:
        num_experiments: Number of experiments to run
    """
    # Create simulation manager
    sim_config = {
        "target_yaw_deg": CONFIG['target_yaw_deg'], 
        "simulation_steps": CONFIG['simulation_steps'], 
        "dt": CONFIG['dt'],
        "pid_output_limit": CONFIG['pid_output_limit'], 
        "pid_sat_penalty": CONFIG['pid_sat_penalty'],
        "pid_strict_output_limit": CONFIG['pid_strict_output_limit'], 
        "pid_sat_hard_penalty": CONFIG['pid_sat_hard_penalty'],
    }
    sim = SimulationManager(sim_config)
    sim.start()
    
    try:
        # Resolve state for resumption
        batch_id, start_run = resolve_state(STATE_PATH, "BO", num_experiments)
        save_state(STATE_PATH, "BO", batch_id, start_run - 1, num_experiments)
        
        if start_run > 1:
            print(f"Resuming batch {batch_id} from experiment {start_run}/{num_experiments}")
        
        # Create executor
        executor = ExperimentExecutor(sim)
        
        # Run experiments
        for run_idx in range(start_run, num_experiments + 1):
            print(f"\n{'='*60}\n  BO HIL: Experiment {run_idx}/{num_experiments}\n{'='*60}")
            
            best_path = executor.run_bo_hil_experiment(run_idx, num_experiments, batch_id)
            save_state(STATE_PATH, "BO", batch_id, run_idx, num_experiments)
            
            print(f"\nExperiment {run_idx} completed. Best results: {best_path}")
        
        print(f"\n✓ All {num_experiments} BO HIL experiments completed!")
        
    finally:
        sim.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple BO HIL experiments")
    parser.add_argument(
        "num_experiments", 
        type=int, 
        nargs="?", 
        default=1,
        help="Number of experiments to run (default: 1)"
    )
    
    args = parser.parse_args()
    
    if args.num_experiments < 1:
        print("Error: num_experiments must be at least 1")
        sys.exit(1)
    
    main(args.num_experiments)
