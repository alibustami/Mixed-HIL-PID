#!/usr/bin/env python3
"""
DE HIL Rerun: Execute Multiple DE HIL Experiments

Runs multiple human-in-the-loop PID optimization experiments using Differential Evolution
with user feedback to refine or expand the search space.

Each experiment consists of multiple iterations. This script allows you to run N experiments
sequentially, with state management for resumption if interrupted.

Usage: 
    python de_hil_rerun.py [num_experiments]
    
Example:
    python de_hil_rerun.py 5  # Run 5 experiments
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mixed_hil_pid.config_loader import load_config, get_robot_config
from mixed_hil_pid.simulator import SimulationManager
from mixed_hil_pid.experiment_executor import ExperimentExecutor
from mixed_hil_pid.utils import load_state, save_state, resolve_state, generate_timestamp


# Load configuration
CONFIG = load_config()

# State management path (will be set per robot type)
STATE_PATH = None


def main(num_experiments=1, robot_type="husky"):
    """
    Run multiple DE HIL experiments.
    
    Args:
        num_experiments: Number of experiments to run
        robot_type: Robot to use ('husky' or 'ackermann')
    """
    # Get robot configuration
    robot_config = get_robot_config(CONFIG, robot_type)
    
    # Create simulation manager
    sim_config = {
        "target_yaw_deg": CONFIG['target_yaw_deg'], 
        "simulation_steps": CONFIG['simulation_steps'], 
        "dt": CONFIG['dt'],
        "pid_output_limit": robot_config['pid_output_limit'],  # Robot-specific limit
        "pid_sat_penalty": CONFIG['pid_sat_penalty'],
        "pid_strict_output_limit": CONFIG['pid_strict_output_limit'], 
        "pid_sat_hard_penalty": CONFIG['pid_sat_hard_penalty'],
    }
    sim = SimulationManager(sim_config, robot_config)
    sim.start()
    
    try:
        # Set up robot-specific state path
        global STATE_PATH
        STATE_PATH = Path(f"logs/{robot_type}_logs/DE_HIL_{robot_type}") / "rerun_state.json"
        
        # Resolve state for resumption
        batch_id, start_run = resolve_state(STATE_PATH, "DE", num_experiments)
        save_state(STATE_PATH, "DE", batch_id, start_run - 1, num_experiments)
        
        if start_run > 1:
            print(f"Resuming batch {batch_id} from experiment {start_run}/{num_experiments}")
        
        # Create executor
        executor = ExperimentExecutor(sim)
        
        # Run experiments
        for run_idx in range(start_run, num_experiments + 1):
            print(f"\n{'='*60}\n  DE HIL ({robot_type}): Experiment {run_idx}/{num_experiments}\n{'='*60}")
            
            best_path = executor.run_de_hil_experiment(run_idx, num_experiments, batch_id, robot_type)
            save_state(STATE_PATH, "DE", batch_id, run_idx, num_experiments)
            
            print(f"\nExperiment {run_idx} completed. Best results: {best_path}")
        
        print(f"\n✓ All {num_experiments} DE HIL experiments completed!")
        
    finally:
        sim.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple DE HIL experiments")
    parser.add_argument(
        "num_experiments", 
        type=int, 
        nargs="?", 
        default=1,
        help="Number of experiments to run (default: 1)"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="husky",
        choices=["husky", "ackermann"],
        help="Robot type to use (default: husky)"
    )
    
    args = parser.parse_args()
    
    if args.num_experiments < 1:
        print("Error: num_experiments must be at least 1")
        sys.exit(1)
    
    main(args.num_experiments, args.robot)
