"""
Metrics calculation module for PID controller performance.

This module provides functions to calculate and validate PID controller
performance metrics such as overshoot, rise time, and settling time.
"""

import numpy as np


def calculate_metrics(history, target_val):
    """
    Calculate performance metrics from a PID response history.
    
    Args:
        history: Dictionary with 'time' and 'actual' arrays
        target_val: Target setpoint value
        
    Returns:
        Dictionary containing:
            - overshoot: Percentage overshoot beyond target
            - rise_time: Time to rise from 10% to 90% of target
            - settling_time: Time to settle within 5% of target
    """
    time_arr = np.array(history["time"], dtype=float)
    actual_arr = np.array(history["actual"], dtype=float)

    # Calculate overshoot
    max_val = float(np.max(actual_arr))
    overshoot = 0.0
    if max_val > target_val:
        overshoot = ((max_val - target_val) / target_val) * 100.0

    # Calculate rise time (10% to 90%)
    try:
        t_10_idx = np.where(actual_arr >= 0.1 * target_val)[0][0]
        t_90_idx = np.where(actual_arr >= 0.9 * target_val)[0][0]
        rise_time = float(time_arr[t_90_idx] - time_arr[t_10_idx])
    except IndexError:
        rise_time = -1.0

    # Calculate settling time (within 5% tolerance)
    tolerance = 0.05 * target_val
    upper_bound = target_val + tolerance
    lower_bound = target_val - tolerance
    out_of_bounds = np.where((actual_arr > upper_bound) | (actual_arr < lower_bound))[0]

    if len(out_of_bounds) == 0:
        settling_time = 0.0
    elif out_of_bounds[-1] == len(actual_arr) - 1:
        settling_time = -1.0  # Never settled
    else:
        settling_time = float(time_arr[out_of_bounds[-1] + 1])

    return {
        "overshoot": overshoot,
        "rise_time": rise_time,
        "settling_time": settling_time
    }


def meets_pid_targets(metrics, max_overshoot=None, max_rise_time=None, max_settling_time=None):
    """
    Check if PID performance metrics meet target criteria.
    
    Args:
        metrics: Dictionary with overshoot, rise_time, settling_time
        max_overshoot: Maximum allowed overshoot percentage (None = no limit)
        max_rise_time: Maximum allowed rise time (None = no limit)
        max_settling_time: Maximum allowed settling time (None = no limit)
        
    Returns:
        Boolean indicating whether all targets are met
    """
    if max_overshoot is not None and metrics["overshoot"] > max_overshoot:
        return False
    
    if max_rise_time is not None:
        if metrics["rise_time"] <= 0 or metrics["rise_time"] > max_rise_time:
            return False
    
    if max_settling_time is not None:
        if metrics["settling_time"] <= 0 or metrics["settling_time"] > max_settling_time:
            return False
    
    return True


def violation_from_sat(sat_info, output_limit):
    """
    Calculate constraint violation from saturation info.
    
    Args:
        sat_info: Dictionary with 'max_abs_raw_output' key
        output_limit: Maximum allowed output value
        
    Returns:
        Violation value (<= 0 is feasible, > 0 is infeasible)
    """
    return float(sat_info["max_abs_raw_output"] - output_limit)


def calculate_violation(sat_info, output_limit, metrics=None, 
                        max_rise_time=None, max_settling_time=None, max_overshoot=None,
                        max_sat_fraction=None):
    """
    Calculate comprehensive constraint violation including time-based constraints.
    
    This is the CORRECT function to use for feasibility checking. It includes:
    - Saturation constraint (output limit OR saturation fraction)
    - Rise time constraint
    - Settling time constraint  
    - Overshoot constraint
    
    Args:
        sat_info: Dictionary with 'max_abs_raw_output' and 'sat_fraction' keys
        output_limit: Maximum allowed output value
        metrics: Dictionary with performance metrics (overshoot, rise_time, settling_time)
        max_rise_time: Maximum allowed rise time (None = no limit)
        max_settling_time: Maximum allowed settling time (None = no limit)
        max_overshoot: Maximum allowed overshoot percentage (None = no limit)
        max_sat_fraction: Maximum allowed saturation fraction (None = use strict output limit)
        
    Returns:
        Violation value (<= 0 means ALL constraints satisfied, > 0 means at least one violated)
        The violation is the MAXIMUM of all individual constraint violations.
    """
    violations = []
    
    # Saturation constraint
    # If max_sat_fraction provided, we use that (so brief limit overruns are allowed)
    # Otherwise, we use strict max_abs_raw_output check
    if max_sat_fraction is not None:
        current_frac = sat_info.get("sat_fraction", 0.0)
        sat_violation = float(current_frac - max_sat_fraction)
    else:
        sat_violation = float(sat_info["max_abs_raw_output"] - output_limit)
        
    violations.append(sat_violation)
    
    # Time-based constraints (only if metrics provided)
    if metrics is not None:
        # Rise time constraint
        if max_rise_time is not None:
            rt = metrics["rise_time"]
            if rt == -1:
                # Failed to rise = strict violation
                violations.append(100.0)
            elif rt > 0:
                violations.append(float(rt - max_rise_time))
        
        # Settling time constraint
        if max_settling_time is not None:
            st = metrics["settling_time"]
            if st == -1:
                # Failed to settle = strict violation
                violations.append(100.0)
            elif st > 0:
                violations.append(float(st - max_settling_time))
        
        # Overshoot constraint
        if max_overshoot is not None:
            overshoot_violation = float(metrics["overshoot"] - max_overshoot)
            violations.append(overshoot_violation)
    
    # Return worst (maximum) violation
    return float(max(violations))

