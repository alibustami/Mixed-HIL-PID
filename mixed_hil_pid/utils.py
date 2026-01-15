"""
Utility functions and classes for HIL PID optimization.

This module contains common utilities, helper functions, and shared classes
including preference learning model and state management for experiment resumption.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone


# --- Preference Learning Model ---

class PreferenceModel:
    """
    A preference learning model that maintains weights over the parameter space.
    
    The model learns from pairwise preferences (preferred vs. other) and updates
    internal weights to bias future anchor points toward preferred regions.
    """
    
    def __init__(self, bounds, lr=0.3):
        """
        Initialize the preference model.
        
        Args:
            bounds: List of (min, max) tuples for each parameter
            lr: Learning rate for weight updates
        """
        self.bounds = np.array(bounds, dtype=float)
        self.lr = float(lr)
        self.weights = np.random.rand(len(self.bounds))

    def _normalize(self):
        """Normalize weights to [0, 1] range."""
        self.weights = np.clip(self.weights, 0.0, 1.0)

    def anchor_params(self):
        """
        Generate anchor parameters based on current weights.
        
        Returns:
            Array of parameter values interpolated using current weights
        """
        min_b = self.bounds[:, 0]
        max_b = self.bounds[:, 1]
        return min_b + self.weights * (max_b - min_b)

    def update_towards(self, preferred, other):
        """
        Update weights to move toward preferred parameters.
        
        Args:
            preferred: Array of preferred parameter values
            other: Array of other (non-preferred) parameter values
            
        Returns:
            Gap vector (other - preferred) in normalized space
        """
        preferred = np.array(preferred, dtype=float)
        other = np.array(other, dtype=float)
        min_b = self.bounds[:, 0]
        span = self.bounds[:, 1] - min_b

        # Normalize to [0, 1]
        pref_norm = (preferred - min_b) / (span + 1e-9)
        other_norm = (other - min_b) / (span + 1e-9)
        gap = other_norm - pref_norm

        # Update weights toward preferred
        self.weights = self.weights + self.lr * (pref_norm - self.weights)
        self._normalize()
        
        return gap


# --- Candidate Comparison Helper ---

def better_candidate(fit, viol, current_best):
    """
    Check if candidate is better than current best.
    
    Prefers feasible solutions (violation <= 0) over infeasible ones.
    Among feasible solutions, prefers lower fitness.
    Among infeasible solutions, prefers lower violation.
    
    Args:
        fit: Fitness value of new candidate
        viol: Constraint violation of new candidate  
        current_best: Dictionary with "fit" and "violation" keys, or None
        
    Returns:
        Boolean indicating whether new candidate is better
    """
    if current_best is None:
        return True
    cur_fit, cur_viol = float(current_best["fit"]), float(current_best["violation"])
    a_feas, b_feas = (viol <= 0.0), (cur_viol <= 0.0)
    if a_feas and not b_feas:
        return True
    if b_feas and not a_feas:
        return False
    return fit < cur_fit if (a_feas and b_feas) else viol < cur_viol


# --- State Management for Experiment Resumption ---

def load_state(state_path):
    """
    Load saved autorun state from file.
    
    Args:
        state_path: Path object to state file
        
    Returns:
        State dictionary or None if file doesn't exist
    """
    try:
        return json.loads(state_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_state(state_path, optimizer, batch_id, last_completed_run, total_runs):
    """
    Save autorun state for resumption.
    
    Args:
        state_path: Path object to state file
        optimizer: Optimizer name (e.g., "DE", "BO", "Mixed")
        batch_id: Batch identifier string
        last_completed_run: Index of last completed run
        total_runs: Total number of runs in this batch
    """
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "optimizer": optimizer,
        "total_runs": total_runs,
        "batch_id": batch_id,
        "last_completed": int(last_completed_run),
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    state_path.write_text(json.dumps(state, indent=2))


def resolve_state(state_path, optimizer, total_runs):
    """
    Determine batch ID and starting run index.
    
    Checks if there's an existing incomplete batch that can be resumed.
    If resumption is possible, returns the existing batch ID and next run index.
    Otherwise, generates a new batch ID and starts from run 1.
    
    Args:
        state_path: Path object to state file
        optimizer: Optimizer name to match (e.g., "DE", "BO", "Mixed")
        total_runs: Total number of runs expected
        
    Returns:
        Tuple of (batch_id, start_run_index)
    """
    state = load_state(state_path)
    if state and state.get("optimizer") == optimizer:
        batch_id = state.get("batch_id")
        last_completed = max(int(state.get("last_completed", 0)), 0)
        if batch_id and (last_completed + 1) <= total_runs:
            return batch_id, last_completed + 1
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"), 1


# --- Timestamp Generation ---

def generate_timestamp():
    """
    Generate UTC timestamp string for experiment naming.
    
    Returns:
        Timestamp string in format YYYYMMDD-HHMMSS
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")