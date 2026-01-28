"""
Logging utilities for HIL PID optimization (shared by all approaches).

This module handles all logging operations including CSV logs,
pickle history files, and configuration dumping for Mixed HIL, DE HIL, and BO HIL.
"""

import csv
import json
import pickle
from datetime import datetime, timezone


def init_logger(config_data, log_path, pkl_path, config_path):
    """
    Initialize logging directory and files.
    
    Args:
        config_data: Dictionary of configuration parameters to save
        log_path: Path object for CSV log file
        pkl_path: Path object for pickle history file
        config_path: Path object for config file
    """
    # Create experiment directory
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path.write_text(json.dumps(config_data, indent=2))

    # Initialize CSV log
    if not log_path.exists():
        with log_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "iteration",
                "choice",
                "cand_a_kp",
                "cand_a_ki",
                "cand_a_kd",
                "fit_a",
                "viol_a",
                "max_raw_a",
                "sat_frac_a",
                "overshoot_a",
                "rise_time_a",
                "settling_time_a",
                "target_ok_a",
                "cand_b_kp",
                "cand_b_ki",
                "cand_b_kd",
                "fit_b",
                "viol_b",
                "max_raw_b",
                "sat_frac_b",
                "overshoot_b",
                "rise_time_b",
                "settling_time_b",
                "target_ok_b",
                "de_mutation",
                "de_best_fit",
                "de_best_viol",
                "de_pop_std",
                "best_overall_fit",
                "bo_span_kp",
                "bo_span_ki",
                "bo_span_kd",
                "pref_weights",
                "gap_note",
                "iter_seconds",
            ])

    # Initialize pickle file
    if not pkl_path.exists():
        with pkl_path.open("wb") as f:
            pickle.dump([], f)


def log_iteration(
    log_path,
    iteration,
    choice_label,
    cand_a, fit_a, viol_a, sat_a, metrics_a, target_ok_a,
    cand_b, fit_b, viol_b, sat_b, metrics_b, target_ok_b,
    de_mutation, de_best_fit, de_best_viol, de_pop_std,
    best_overall_fit, bo_spans, pref_weights,
    gap_note="",
    iter_seconds=0.0
):
    """
    Log a single iteration to CSV file.
    
    Args:
        log_path: Path object for CSV log file
        iteration: Iteration number
        choice_label: User choice label (prefer_de, prefer_bo, etc.)
        cand_a: DE candidate parameters
        fit_a: DE fitness
        viol_a: DE constraint violation
        sat_a: DE saturation info dictionary
        metrics_a: DE performance metrics
        target_ok_a: Whether DE met targets
        cand_b: BO candidate parameters
        fit_b: BO fitness
        viol_b: BO constraint violation
        sat_b: BO saturation info dictionary
        metrics_b: BO performance metrics
        target_ok_b: Whether BO met targets
        de_mutation: Current DE mutation factor
        de_best_fit: DE best fitness
        de_best_viol: DE best violation
        de_pop_std: DE population standard deviation
        best_overall_fit: Overall best fitness
        bo_spans: BO search space spans
        pref_weights: Preference model weights
        gap_note: Additional notes
        iter_seconds: Iteration duration
    """
    ts = datetime.now(timezone.utc).isoformat()
    row = [
        ts,
        int(iteration),
        str(choice_label),
        float(cand_a[0]), float(cand_a[1]), float(cand_a[2]),
        float(fit_a),
        float(viol_a),
        float(sat_a["max_abs_raw_output"]),
        float(sat_a["sat_fraction"]),
        float(metrics_a["overshoot"]),
        float(metrics_a["rise_time"]),
        float(metrics_a["settling_time"]),
        int(bool(target_ok_a)),
        float(cand_b[0]), float(cand_b[1]), float(cand_b[2]),
        float(fit_b),
        float(viol_b),
        float(sat_b["max_abs_raw_output"]),
        float(sat_b["sat_fraction"]),
        float(metrics_b["overshoot"]),
        float(metrics_b["rise_time"]),
        float(metrics_b["settling_time"]),
        int(bool(target_ok_b)),
        float(de_mutation),
        float(de_best_fit),
        float(de_best_viol),
        float(de_pop_std),
        float(best_overall_fit),
        float(bo_spans[0]),
        float(bo_spans[1]),
        float(bo_spans[2]),
        "|".join(f"{w:.4f}" for w in pref_weights),
        str(gap_note),
        float(iter_seconds),
    ]
    with log_path.open("a", newline="") as f:
        csv.writer(f).writerow(row)


def append_histories_pickle(pkl_path, records):
    """
    Append iteration history records to pickle file.
    
    Args:
        pkl_path: Path object for pickle file
        records: List of dictionaries to append
    """
    try:
        with pkl_path.open("rb") as f:
            existing = pickle.load(f)
    except FileNotFoundError:
        existing = []
    with pkl_path.open("wb") as f:
        pickle.dump(existing, f)


def save_trace(trace_path, trace_data):
    """
    Save simulation trace data to consistency CSV file.

    Args:
        trace_path: Path object for the trace CSV file
        trace_data: List of dictionaries containing trace data
    """
    if not trace_data:
        return

    # Create directory if needed
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    # basic validation to get headers
    headers = list(trace_data[0].keys())

    with trace_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(trace_data)
