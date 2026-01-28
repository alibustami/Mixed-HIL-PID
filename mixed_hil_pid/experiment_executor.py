"""
Experiment Executor for HIL PID Optimization.

This module contains the unified ExperimentExecutor class that handles
running single experiments for all optimization approaches (Mixed HIL,
BO HIL, DE HIL, BO Vanilla, DE Vanilla).
"""

import json
import time
from pathlib import Path

import numpy as np

from mixed_hil_pid.config_loader import (
    get_pid_bounds,
    get_robot_config,
    load_config,
)

# Load configuration at module level
_CONFIG = load_config()
PID_BOUNDS = get_pid_bounds(_CONFIG)
from mixed_hil_pid.gui import MixedHILGUI, SingleHILGUI
from mixed_hil_pid.logger import (
    append_histories_pickle,
    init_logger,
    log_iteration,
    save_trace,
)
from mixed_hil_pid.metrics import (
    calculate_metrics,
    calculate_violation,
    meets_pid_targets,
    violation_from_sat,
)
from mixed_hil_pid.optimizers.bayesian_optimizer import (
    ConstrainedBayesianOptimizer,
)
from mixed_hil_pid.optimizers.differential_evolution import (
    DifferentialEvolutionOptimizer,
)
from mixed_hil_pid.utils import PreferenceModel, better_candidate


class ExperimentExecutor:
    """
    Unified executor for running single experiments across all optimization approaches.

    This class extracts the core experiment logic from each script and provides
    a consistent interface for running experiments regardless of the optimization method.
    """

    def __init__(self, sim):
        """
        Initialize the experiment executor.

        Args:
            sim: SimulationManager instance (already started)
        """
        self.sim = sim

    def run_mixed_hil_experiment(
        self, run_index, total_runs, batch_id, robot_type="husky"
    ):
        """
        Run a single Mixed HIL experiment.

        Args:
            run_index: Index of this run (1-based)
            total_runs: Total number of runs
            batch_id: Batch identifier string
            robot_type: Robot type ('husky' or 'ackermann')

        Returns:
            Path to best results JSON file
        """
        # Setup paths
        run_suffix = f"_run{run_index:02d}" if total_runs > 1 else ""
        exp_dir = (
            Path(f"logs/{robot_type}_logs/MixedHIL_{robot_type}")
            / f"MixedHIL_{batch_id}{run_suffix}"
        )
        log_path = exp_dir / "iteration_log.csv"
        pkl_path = exp_dir / "iteration_log.pkl"
        config_path = exp_dir / "config.yaml"
        best_path = exp_dir / "best_results.json"

        # Initialize algorithms
        pref_model = PreferenceModel(PID_BOUNDS, lr=_CONFIG["preference_lr"])
        de = DifferentialEvolutionOptimizer(
            bounds=PID_BOUNDS,
            pop_size=6,
            mutation_factor=_CONFIG["base_mutation"],
        )
        de.population[0] = np.clip(
            pref_model.anchor_params(), de.bounds[:, 0], de.bounds[:, 1]
        )
        bo = ConstrainedBayesianOptimizer(
            bounds=PID_BOUNDS, pof_min=_CONFIG["bo_pof_min"]
        )
        gui = MixedHILGUI()

        # Initialize logging
        config_data = {
            "optimizer": "Mixed",
            "batch_id": batch_id,
            "run_index": run_index,
            "total_runs": total_runs,
            "pid_bounds": PID_BOUNDS,
            "simulation_steps": _CONFIG["simulation_steps"],
            "dt": _CONFIG["dt"],
            "base_mutation": _CONFIG["base_mutation"],
            "preference_lr": _CONFIG["preference_lr"],
        }
        init_logger(config_data, log_path, pkl_path, config_path)

        # Get robot-specific output limit
        # Get robot-specific constraints
        robot_config = get_robot_config(_CONFIG, robot_type)
        pid_output_limit = robot_config["pid_output_limit"]
        max_overshoot = robot_config.get("pid_max_overshoot_pct", 5)
        max_rise = robot_config.get("pid_max_rise_time", 2)
        max_settle = robot_config.get("pid_max_settling_time", 5)
        max_sat_frac = robot_config.get("max_sat_fraction", None)

        def fitness_wrapper(p):
            fit, hist, sat = self.sim.evaluate(p, return_history=True)
            metrics = calculate_metrics(hist, _CONFIG["target_yaw_deg"])
            viol = calculate_violation(
                sat, pid_output_limit, metrics,
                max_rise, max_settle, max_overshoot, max_sat_frac
            )
            return fit, viol

        # Warm-start BO
        for cand in de.population:
            fit, viol = fitness_wrapper(cand)
            bo.update(cand, fit, viol)

        best_record, iteration = None, 0
        try:
            while iteration < _CONFIG["max_iterations"]:
                iteration += 1
                iter_start = time.time()

                # Generate candidates
                cand_a, fit_a_fast, viol_a_fast = de.evolve(fitness_wrapper)
                cand_b = bo.propose_location()
                
                # Evaluate BO candidate with full constraints for update
                fit_b_fast, viol_b_fast = fitness_wrapper(cand_b)
                bo.update(cand_b, fit_b_fast, viol_b_fast)
                
                # Note: fitness_wrapper calls sim.evaluate internally, so we re-evaluate below for GUI history?
                # Actually, duplicate evaluation is wasteful but ensures consistency with existing flow that separates "fast" eval from "GUI" eval
                # For this fix, correctness > efficiency.
                
                # Update BO with DE candidate result too
                bo.update(cand_a, fit_a_fast, viol_a_fast)

                # Simulate with history for GUI (with logging labels)
                fit_a, hist_a, sat_a, trace_a = self.sim.evaluate(
                    cand_a,
                    label_text="DE",
                    return_history=True,
                    realtime=_CONFIG["display_realtime"],
                    return_full_trace=True,
                )
                fit_b, hist_b, sat_b, trace_b = self.sim.evaluate(
                    cand_b,
                    label_text="BO",
                    return_history=True,
                    realtime=_CONFIG["display_realtime"],
                    return_full_trace=True,
                )
                
                # Save traces
                save_trace(exp_dir / "traces" / f"iter_{iteration}_DE.csv", trace_a)
                save_trace(exp_dir / "traces" / f"iter_{iteration}_BO.csv", trace_b)
                
                # Re-calculate EVERYTHING for consistent logging/GUI variables
                # (metrics_a/b are calculated later in original code, but we calculate viol here)
                metrics_a = calculate_metrics(hist_a, _CONFIG["target_yaw_deg"])
                metrics_b = calculate_metrics(hist_b, _CONFIG["target_yaw_deg"])
                
                viol_a = calculate_violation(
                    sat_a, pid_output_limit, metrics_a,
                    max_rise, max_settle, max_overshoot, max_sat_frac
                )
                viol_b = calculate_violation(
                    sat_b, pid_output_limit, metrics_b,
                    max_rise, max_settle, max_overshoot, max_sat_frac
                )

                # Get robot-specific performance targets
                robot_config = get_robot_config(_CONFIG, robot_type)
                max_overshoot = robot_config.get("pid_max_overshoot_pct", 5)
                max_rise_time = robot_config.get("pid_max_rise_time", 2)
                max_settling_time = robot_config.get(
                    "pid_max_settling_time", 5
                )

                metrics_a, metrics_b = calculate_metrics(
                    hist_a, _CONFIG["target_yaw_deg"]
                ), calculate_metrics(hist_b, _CONFIG["target_yaw_deg"])
                target_ok_a = meets_pid_targets(
                    metrics_a, max_overshoot, max_rise_time, max_settling_time
                )
                target_ok_b = meets_pid_targets(
                    metrics_b, max_overshoot, max_rise_time, max_settling_time
                )

                # Debug: Print termination criteria status
                print(
                    f"\n[Iter {iteration}] Termination Status for {robot_type}:"
                )
                print(
                    f"  DE: Overshoot={metrics_a['overshoot']:.2f}% (limit {max_overshoot}), "
                    f"Rise={metrics_a['rise_time']:.3f}s (limit {max_rise_time}), "
                    f"Settle={metrics_a['settling_time']:.3f}s (limit {max_settling_time}), "
                    f"Viol={viol_a:.3f} → target_ok={target_ok_a}, feasible={viol_a<=0}"
                )
                print(
                    f"  BO: Overshoot={metrics_b['overshoot']:.2f}% (limit {max_overshoot}), "
                    f"Rise={metrics_b['rise_time']:.3f}s (limit {max_rise_time}), "
                    f"Settle={metrics_b['settling_time']:.3f}s (limit {max_settling_time}), "
                    f"Viol={viol_b:.3f} → target_ok={target_ok_b}, feasible={viol_b<=0}"
                )

                # Update best
                for lbl, c, f, v, m in [
                    ("DE", cand_a, fit_a, viol_a, metrics_a),
                    ("BO", cand_b, fit_b, viol_b, metrics_b),
                ]:
                    if better_candidate(f, v, best_record):
                        best_record = {
                            "iteration": iteration,
                            "label": lbl,
                            "params": c.tolist(),
                            "fit": float(f),
                            "violation": float(v),
                            "metrics": m,
                        }

                # Auto-terminate
                if (target_ok_a and viol_a <= 0) or (
                    target_ok_b and viol_b <= 0
                ):
                    print(
                        f"\n🎉 [Auto-terminate] Target met at iteration {iteration}!"
                    )
                    log_iteration(
                        log_path,
                        iteration,
                        "auto_terminate",
                        cand_a,
                        fit_a,
                        viol_a,
                        sat_a,
                        metrics_a,
                        target_ok_a,
                        cand_b,
                        fit_b,
                        viol_b,
                        sat_b,
                        metrics_b,
                        target_ok_b,
                        de.mutation_factor,
                        *de.best_scores(),
                        np.mean(np.std(de.population, axis=0)),
                        min(fit_a, fit_b),
                        bo.bounds[:, 1] - bo.bounds[:, 0],
                        pref_model.weights,
                        "auto_term",
                        time.time() - iter_start,
                    )
                    break

                # GUI feedback
                choice = gui.show_comparison(
                    hist_a, hist_b, cand_a, cand_b, metrics_a, metrics_b
                )
                choice_labels = {
                    1: "prefer_de",
                    2: "prefer_bo",
                    3: "tie_refine",
                    4: "reject_expand",
                }

                if choice == 1:  # Prefer DE
                    pref_model.update_towards(cand_a, cand_b)
                    de.inject_candidate(
                        pref_model.anchor_params(),
                        eval_func=fitness_wrapper,
                        protect_best=True,
                    )
                    bo.nudge_with_preference(cand_a, fit_a, fit_b, viol_a)
                elif choice == 2:  # Prefer BO
                    pref_model.update_towards(cand_b, cand_a)
                    de.inject_candidate(
                        cand_b, eval_func=fitness_wrapper, protect_best=True
                    )
                    de.inject_candidate(
                        pref_model.anchor_params(),
                        eval_func=fitness_wrapper,
                        protect_best=True,
                    )
                    bo.nudge_with_preference(cand_b, fit_b, fit_a, viol_b)
                elif choice == 3:  # Refine
                    avg = (np.array(cand_a) + np.array(cand_b)) / 2.0
                    de.refine_search_space(avg)
                    bo.refine_bounds(avg)
                elif choice == 4:  # Expand
                    de.expand_search_space()
                    bo.expand_bounds()
                else:
                    break

                log_iteration(
                    log_path,
                    iteration,
                    choice_labels.get(choice, "exit"),
                    cand_a,
                    fit_a,
                    viol_a,
                    sat_a,
                    metrics_a,
                    target_ok_a,
                    cand_b,
                    fit_b,
                    viol_b,
                    sat_b,
                    metrics_b,
                    target_ok_b,
                    de.mutation_factor,
                    *de.best_scores(),
                    np.mean(np.std(de.population, axis=0)),
                    min(fit_a, fit_b),
                    bo.bounds[:, 1] - bo.bounds[:, 0],
                    pref_model.weights,
                    choice_labels.get(choice, ""),
                    time.time() - iter_start,
                )

        finally:
            if best_record:
                best_path.write_text(json.dumps(best_record, indent=2))
            print(f"✓ Best solution saved to {best_path}")

        return best_path

    def run_bo_hil_experiment(
        self, run_index, total_runs, batch_id, robot_type="husky"
    ):
        """
        Run a single BO HIL experiment.

        Args:
            run_index: Index of this run (1-based)
            total_runs: Total number of runs
            batch_id: Batch identifier string
            robot_type: Robot type ('husky' or 'ackermann')

        Returns:
            Path to best results JSON file
        """
        # Setup paths
        run_suffix = f"_run{run_index:02d}" if total_runs > 1 else ""
        exp_dir = (
            Path(f"logs/{robot_type}_logs/BO_HIL_{robot_type}")
            / f"bo_hil_{batch_id}{run_suffix}"
        )
        log_path = exp_dir / "iteration_log.csv"
        pkl_path = exp_dir / "iteration_log.pkl"
        config_path = exp_dir / "config.yaml"
        best_path = exp_dir / "best_results.json"

        # Initialize algorithm
        bo = ConstrainedBayesianOptimizer(
            bounds=PID_BOUNDS, pof_min=_CONFIG["bo_pof_min"]
        )
        gui = SingleHILGUI(title="BO HIL")

        # Initialize logging
        config_data = {
            "optimizer": "BO",
            "batch_id": batch_id,
            "run_index": run_index,
            "total_runs": total_runs,
            "pid_bounds": PID_BOUNDS,
            "algorithm": "BO",
        }
        init_logger(config_data, log_path, pkl_path, config_path)

        # Get robot-specific output limit and constraints
        robot_config = get_robot_config(_CONFIG, robot_type)
        pid_output_limit = robot_config["pid_output_limit"]
        max_overshoot = robot_config.get("pid_max_overshoot_pct", 5)
        max_rise = robot_config.get("pid_max_rise_time", 2)
        max_settle = robot_config.get("pid_max_settling_time", 5)
        max_sat_frac = robot_config.get("max_sat_fraction", None)

        def fitness_wrapper(p):
            fit, hist, sat = self.sim.evaluate(p, return_history=True)
            metrics = calculate_metrics(hist, _CONFIG["target_yaw_deg"])
            viol = calculate_violation(
                sat, pid_output_limit, metrics,
                max_rise, max_settle, max_overshoot, max_sat_frac
            )
            return fit, viol

        # Warm-start BO with random samples
        for _ in range(5):
            cand = np.array(
                [np.random.uniform(b[0], b[1]) for b in PID_BOUNDS]
            )
            fit, viol = fitness_wrapper(cand)
            bo.update(cand, fit, viol)

        best_record, iteration = None, 0
        try:
            while iteration < _CONFIG["max_iterations"]:
                iteration += 1
                iter_start = time.time()

                # Generate candidate
                cand = bo.propose_location()

                # Simulate with history
                # Simulate with history
                fit, hist, sat, trace = self.sim.evaluate(
                    cand,
                    label_text=f"BO Iter {iteration}",
                    return_history=True,
                    realtime=_CONFIG["display_realtime"],
                    return_full_trace=True,
                )
                
                # Save trace
                save_trace(exp_dir / "traces" / f"iter_{iteration}_BO.csv", trace)
                
                metrics = calculate_metrics(hist, _CONFIG["target_yaw_deg"])
                viol = calculate_violation(
                    sat, pid_output_limit, metrics,
                    max_rise, max_settle, max_overshoot, max_sat_frac
                )


                # Get robot-specific performance targets
                robot_config = get_robot_config(_CONFIG, robot_type)
                max_overshoot = robot_config.get("pid_max_overshoot_pct", 5)
                max_rise_time = robot_config.get("pid_max_rise_time", 2)
                max_settling_time = robot_config.get(
                    "pid_max_settling_time", 5
                )

                metrics = calculate_metrics(hist, _CONFIG["target_yaw_deg"])
                target_ok = meets_pid_targets(
                    metrics, max_overshoot, max_rise_time, max_settling_time
                )

                # Update BO
                bo.update(cand, fit, viol)

                # Update best
                if best_record is None or (
                    viol <= 0 and fit < best_record["fit"]
                ):
                    best_record = {
                        "iteration": iteration,
                        "params": cand.tolist(),
                        "fit": float(fit),
                        "violation": float(viol),
                        "metrics": metrics,
                    }

                # Debug: Print termination criteria status
                print(
                    f"\n[Iter {iteration}] Termination Status for {robot_type}:"
                )
                print(
                    f"  BO: Overshoot={metrics['overshoot']:.2f}% (limit {max_overshoot}), "
                    f"Rise={metrics['rise_time']:.3f}s (limit {max_rise_time}), "
                    f"Settle={metrics['settling_time']:.3f}s (limit {max_settling_time}), "
                    f"Viol={viol:.3f} → target_ok={target_ok}, feasible={viol<=0}"
                )

                # Auto-terminate
                if target_ok and viol <= 0:
                    print(
                        f"\n🎉 [Auto-terminate] Target met at iteration {iteration}!"
                    )
                    # Log final iteration before termination
                    bo_spans = bo.bounds[:, 1] - bo.bounds[:, 0]
                    log_iteration(
                        log_path,
                        iteration,
                        "auto_terminate",
                        cand,
                        fit,
                        viol,
                        sat,
                        metrics,
                        target_ok,
                        cand,
                        fit,
                        viol,
                        sat,
                        metrics,
                        target_ok,
                        0.0,
                        float("inf"),
                        float("inf"),
                        0.0,
                        float(fit),
                        bo_spans,
                        np.zeros(3),
                        "auto_term",
                        time.time() - iter_start,
                    )
                    append_histories_pickle(
                        pkl_path,
                        [
                            {
                                "iteration": iteration,
                                "label": "BO",
                                "params": cand.tolist(),
                                "fit": float(fit),
                                "violation": float(viol),
                                "metrics": metrics,
                                "history": hist,
                            }
                        ],
                    )
                    break

                # GUI feedback
                choice = gui.show_candidate(
                    hist, cand, fit, metrics, label=f"BO Candidate {iteration}"
                )
                choice_labels = {1: "accept_refine", 2: "reject_expand"}

                if choice == 1:  # ACCEPT (Refine)
                    bo.refine_bounds(cand)
                    print(
                        f"[Accept] Refining around Kp={cand[0]:.2f}, Ki={cand[1]:.2f}, Kd={cand[2]:.2f}"
                    )
                elif choice == 2:  # REJECT (Expand)
                    bo.expand_bounds()
                    print(f"[Reject] Expanding search space")
                else:
                    break

                # Log iteration
                bo_spans = bo.bounds[:, 1] - bo.bounds[:, 0]
                log_iteration(
                    log_path,
                    iteration,
                    choice_labels.get(choice, "exit"),
                    cand,
                    fit,
                    viol,
                    sat,
                    metrics,
                    target_ok,
                    cand,
                    fit,
                    viol,
                    sat,
                    metrics,
                    target_ok,
                    0.0,
                    float("inf"),
                    float("inf"),
                    0.0,
                    float(fit),
                    bo_spans,
                    np.zeros(3),
                    choice_labels.get(choice, ""),
                    time.time() - iter_start,
                )

                # Save iteration history
                append_histories_pickle(
                    pkl_path,
                    [
                        {
                            "iteration": iteration,
                            "label": "BO",
                            "params": cand.tolist(),
                            "fit": float(fit),
                            "violation": float(viol),
                            "metrics": metrics,
                            "history": hist,
                        }
                    ],
                )

        finally:
            if best_record:
                best_path.write_text(json.dumps(best_record, indent=2))
            print(f"✓ Best solution saved to {best_path}")

        return best_path

    def run_de_hil_experiment(
        self, run_index, total_runs, batch_id, robot_type="husky"
    ):
        """
        Run a single DE HIL experiment.

        Args:
            run_index: Index of this run (1-based)
            total_runs: Total number of runs
            batch_id: Batch identifier string
            robot_type: Robot type ('husky' or 'ackermann')

        Returns:
            Path to best results JSON file
        """
        # Setup paths
        run_suffix = f"_run{run_index:02d}" if total_runs > 1 else ""
        exp_dir = (
            Path(f"logs/{robot_type}_logs/DE_HIL_{robot_type}")
            / f"de_hil_{batch_id}{run_suffix}"
        )
        log_path = exp_dir / "iteration_log.csv"
        pkl_path = exp_dir / "iteration_log.pkl"
        config_path = exp_dir / "config.yaml"
        best_path = exp_dir / "best_results.json"

        # Initialize algorithm
        de = DifferentialEvolutionOptimizer(
            bounds=PID_BOUNDS,
            pop_size=6,
            mutation_factor=_CONFIG["base_mutation"],
        )
        gui = SingleHILGUI(title="DE HIL")

        # Initialize logging
        config_data = {
            "optimizer": "DE",
            "batch_id": batch_id,
            "run_index": run_index,
            "total_runs": total_runs,
            "pid_bounds": PID_BOUNDS,
            "algorithm": "DE",
        }
        init_logger(config_data, log_path, pkl_path, config_path)



        # Get robot-specific output limit
        # Get robot-specific constraints
        robot_config = get_robot_config(_CONFIG, robot_type)
        pid_output_limit = robot_config["pid_output_limit"]
        max_overshoot = robot_config.get("pid_max_overshoot_pct", 5)
        max_rise = robot_config.get("pid_max_rise_time", 2)
        max_settle = robot_config.get("pid_max_settling_time", 5)
        max_sat_frac = robot_config.get("max_sat_fraction", None)

        def fitness_wrapper(p):
            fit, hist, sat = self.sim.evaluate(p, return_history=True)
            metrics = calculate_metrics(hist, _CONFIG["target_yaw_deg"])
            viol = calculate_violation(
                sat, pid_output_limit, metrics,
                max_rise, max_settle, max_overshoot, max_sat_frac
            )
            return fit, viol

        best_record, iteration = None, 0
        try:
            while iteration < _CONFIG["max_iterations"]:
                iteration += 1
                iter_start = time.time()

                # Generate candidate
                cand, fit_fast, viol_fast = de.evolve(fitness_wrapper)

                # Simulate with history
                fit, hist, sat, trace = self.sim.evaluate(
                    cand,
                    label_text=f"DE Iter {iteration}",
                    return_history=True,
                    realtime=_CONFIG["display_realtime"],
                    return_full_trace=True,
                )
                
                # Save trace
                save_trace(exp_dir / "traces" / f"iter_{iteration}_DE.csv", trace)
                
                metrics = calculate_metrics(hist, _CONFIG["target_yaw_deg"])
                viol = calculate_violation(
                    sat, pid_output_limit, metrics,
                    max_rise, max_settle, max_overshoot, max_sat_frac
                )

                # Get robot-specific performance targets
                robot_config = get_robot_config(_CONFIG, robot_type)
                max_overshoot = robot_config.get("pid_max_overshoot_pct", 5)
                max_rise_time = robot_config.get("pid_max_rise_time", 2)
                max_settling_time = robot_config.get(
                    "pid_max_settling_time", 5
                )

                metrics = calculate_metrics(hist, _CONFIG["target_yaw_deg"])
                target_ok = meets_pid_targets(
                    metrics, max_overshoot, max_rise_time, max_settling_time
                )

                # Update best
                if best_record is None or (
                    viol <= 0 and fit < best_record["fit"]
                ):
                    best_record = {
                        "iteration": iteration,
                        "params": cand.tolist(),
                        "fit": float(fit),
                        "violation": float(viol),
                        "metrics": metrics,
                    }

                # Debug: Print termination criteria status
                print(
                    f"\n[Iter {iteration}] Termination Status for {robot_type}:"
                )
                print(
                    f"  DE: Overshoot={metrics['overshoot']:.2f}% (limit {max_overshoot}), "
                    f"Rise={metrics['rise_time']:.3f}s (limit {max_rise_time}), "
                    f"Settle={metrics['settling_time']:.3f}s (limit {max_settling_time}), "
                    f"Viol={viol:.3f} → target_ok={target_ok}, feasible={viol<=0}"
                )

                # Auto-terminate
                if target_ok and viol <= 0:
                    print(
                        f"\n🎉 [Auto-terminate] Target met at iteration {iteration}!"
                    )
                    # Log final iteration before termination
                    de_best_fit, de_best_viol = de.best_scores()
                    log_iteration(
                        log_path,
                        iteration,
                        "auto_terminate",
                        cand,
                        fit,
                        viol,
                        sat,
                        metrics,
                        target_ok,
                        cand,
                        fit,
                        viol,
                        sat,
                        metrics,
                        target_ok,
                        de.mutation_factor,
                        de_best_fit,
                        de_best_viol,
                        np.mean(np.std(de.population, axis=0)),
                        float(fit),
                        np.zeros(3),
                        np.zeros(3),
                        "auto_term",
                        time.time() - iter_start,
                    )
                    append_histories_pickle(
                        pkl_path,
                        [
                            {
                                "iteration": iteration,
                                "label": "DE",
                                "params": cand.tolist(),
                                "fit": float(fit),
                                "violation": float(viol),
                                "metrics": metrics,
                                "history": hist,
                            }
                        ],
                    )
                    break

                # GUI feedback
                choice = gui.show_candidate(
                    hist, cand, fit, metrics, label=f"DE Candidate {iteration}"
                )
                choice_labels = {1: "accept_refine", 2: "reject_expand"}

                if choice == 1:  # ACCEPT (Refine)
                    de.refine_search_space(cand)
                    print(
                        f"[Accept] Refining around Kp={cand[0]:.2f}, Ki={cand[1]:.2f}, Kd={cand[2]:.2f}"
                    )
                elif choice == 2:  # REJECT (Expand)
                    de.expand_search_space()
                    print(f"[Reject] Expanding search space")
                else:
                    break

                # Log iteration
                de_best_fit, de_best_viol = de.best_scores()
                log_iteration(
                    log_path,
                    iteration,
                    choice_labels.get(choice, "exit"),
                    cand,
                    fit,
                    viol,
                    sat,
                    metrics,
                    target_ok,
                    cand,
                    fit,
                    viol,
                    sat,
                    metrics,
                    target_ok,
                    de.mutation_factor,
                    de_best_fit,
                    de_best_viol,
                    np.mean(np.std(de.population, axis=0)),
                    float(fit),
                    np.zeros(3),
                    np.zeros(3),
                    choice_labels.get(choice, ""),
                    time.time() - iter_start,
                )

                # Save iteration history
                append_histories_pickle(
                    pkl_path,
                    [
                        {
                            "iteration": iteration,
                            "label": "DE",
                            "params": cand.tolist(),
                            "fit": float(fit),
                            "violation": float(viol),
                            "metrics": metrics,
                            "history": hist,
                        }
                    ],
                )

        finally:
            if best_record:
                best_path.write_text(json.dumps(best_record, indent=2))
            print(f"✓ Best solution saved to {best_path}")

        return best_path

    def run_bo_vanilla_experiment(
        self, run_index, total_runs, batch_id, robot_type="husky"
    ):
        """
        Run a single BO vanilla (autorun) experiment.

        Args:
            run_index: Index of this run (1-based)
            total_runs: Total number of runs
            batch_id: Batch identifier string
            robot_type: Robot type ('husky' or 'ackermann')

        Returns:
            Path to best results JSON file
        """
        # Setup paths
        run_suffix = f"_run{run_index:02d}" if total_runs > 1 else ""
        exp_dir = (
            Path(f"logs/{robot_type}_logs/BO_Vanilla_{robot_type}")
            / f"BO_{batch_id}{run_suffix}"
        )
        log_path = exp_dir / "iteration_log.csv"
        pkl_path = exp_dir / "iteration_log.pkl"
        config_path = exp_dir / "config.yaml"
        best_path = exp_dir / "best_results.json"

        # BO-specific config
        INIT_BO_SEEDS = 2
        BO_MAX_RETRIES = 5

        # Initialize logging
        config_data = {
            "optimizer": "BO",
            "batch_id": batch_id,
            "run_index": run_index,
            "total_runs": total_runs,
            "pid_bounds": PID_BOUNDS,
            "bo_pof_min": _CONFIG["bo_pof_min"],
            "init_seeds": INIT_BO_SEEDS,
            "max_retries": BO_MAX_RETRIES,
            "max_iterations": _CONFIG["max_iterations"],
            "target_yaw_deg": _CONFIG["target_yaw_deg"],
        }
        init_logger(config_data, log_path, pkl_path, config_path)

        # Initialize BO
        bo = ConstrainedBayesianOptimizer(
            bounds=PID_BOUNDS, pof_min=_CONFIG["bo_pof_min"]
        )

        # Safe seed
        safe_seed = np.array([b[0] for b in PID_BOUNDS], dtype=float)

        best_overall_fit = float("inf")
        best_record = None

        # Get robot-specific output limit
        robot_config = get_robot_config(_CONFIG, robot_type)
        pid_output_limit = robot_config["pid_output_limit"]
        max_overshoot = robot_config.get("pid_max_overshoot_pct", 5)
        max_rise = robot_config.get("pid_max_rise_time", 2)
        max_settle = robot_config.get("pid_max_settling_time", 5)
        max_sat_frac = robot_config.get("max_sat_fraction", None)

        def eval_obj_and_violation(params):
            """Evaluate fitness and constraint violation."""
            fit, hist, sat = self.sim.evaluate(
                params, return_history=True, realtime=False
            )
            metrics = calculate_metrics(hist, _CONFIG["target_yaw_deg"])
            viol = calculate_violation(
                sat, pid_output_limit, metrics,
                max_rise, max_settle, max_overshoot, max_sat_frac
            )
            return float(fit), float(viol)

        # Warm start BO
        # Warm start BO
        # safe_fit, safe_viol = eval_obj_and_violation(safe_seed)
        # bo.update(safe_seed, safe_fit, safe_viol)
        # print(f"[Init BO] safe_seed: fit={safe_fit:.4f}, viol={safe_viol:.3f}")

        # Random initialization
        lows, highs = np.array([b[0] for b in PID_BOUNDS]), np.array(
            [b[1] for b in PID_BOUNDS]
        )
        for idx in range(INIT_BO_SEEDS):
            cand = np.random.uniform(lows, highs)
            f, v = eval_obj_and_violation(cand)
            bo.update(cand, f, v)
            print(f"[Init BO] Seed {idx}: fit={f:.4f}, viol={v:.3f}")

        iteration = 0
        try:
            while iteration < _CONFIG["max_iterations"]:
                iter_start = time.time()
                iteration += 1
                print(f"\n--- Iteration {iteration} ---")

                # Propose with retries
                # Propose (Single attempt, no retries)
                cand = bo.propose_location()
                fit_fast, viol_fast = eval_obj_and_violation(cand)
                bo.update(cand, fit_fast, viol_fast)

                # Evaluate with history
                fit, hist, sat, trace = self.sim.evaluate(
                    cand,
                    label_text=f"BO Iter {iteration}",
                    return_history=True,
                    realtime=_CONFIG["display_realtime"],
                    return_full_trace=True,
                )
                
                # Save trace
                save_trace(exp_dir / "traces" / f"iter_{iteration}_BO.csv", trace)
                metrics = calculate_metrics(hist, _CONFIG["target_yaw_deg"])
                violation = calculate_violation(
                    sat, pid_output_limit, metrics,
                    max_rise, max_settle, max_overshoot, max_sat_frac
                )

                # Get robot-specific performance targets
                # (targets already retrieved)

                target_ok = meets_pid_targets(
                    metrics, max_overshoot, max_rise, max_settle
                )
                safe_ok = violation <= 0.0
                best_overall_fit = min(best_overall_fit, float(fit))

                # Update best
                if best_record is None or (
                    safe_ok and fit < best_record["fit"]
                ):
                    best_record = {
                        "iteration": iteration,
                        "params": cand.tolist(),
                        "fit": float(fit),
                        "violation": float(violation),
                        "metrics": metrics,
                    }

                # Log
                bo_spans = bo.bounds[:, 1] - bo.bounds[:, 0]
                log_iteration(
                    log_path,
                    iteration,
                    "auto",
                    cand,
                    fit,
                    violation,
                    sat,
                    metrics,
                    target_ok,
                    cand,
                    fit,
                    violation,
                    sat,
                    metrics,
                    target_ok,
                    0.0,
                    float("inf"),
                    float("inf"),
                    0.0,
                    best_overall_fit,
                    bo_spans,
                    np.zeros(3),
                    "BO_autorun",
                    time.time() - iter_start,
                )

                append_histories_pickle(
                    pkl_path,
                    [
                        {
                            "iteration": iteration,
                            "label": "BO",
                            "params": cand.tolist(),
                            "fit": float(fit),
                            "violation": float(violation),
                            "metrics": metrics,
                            "history": hist,
                        }
                    ],
                )

                # Check termination
                if target_ok and safe_ok:
                    print("[Terminate] Targets met!")
                    break

        finally:
            if best_record:
                best_path.write_text(json.dumps(best_record, indent=2))
                print(f"✓ Best saved to {best_path}")

        return best_path

    def run_de_vanilla_experiment(
        self, run_index, total_runs, batch_id, robot_type="husky"
    ):
        """
        Run a single DE vanilla (autorun) experiment.

        Args:
            run_index: Index of this run (1-based)
            total_runs: Total number of runs
            batch_id: Batch identifier string
            robot_type: Robot type ('husky' or 'ackermann')

        Returns:
            Path to best results JSON file
        """
        # Setup paths
        run_suffix = f"_run{run_index:02d}" if total_runs > 1 else ""
        exp_dir = (
            Path(f"logs/{robot_type}_logs/DE_Vanilla_{robot_type}")
            / f"DE_{batch_id}{run_suffix}"
        )
        log_path = exp_dir / "iteration_log.csv"
        pkl_path = exp_dir / "iteration_log.pkl"
        config_path = exp_dir / "config.yaml"
        best_path = exp_dir / "best_results.json"

        # Initialize logging
        config_data = {
            "optimizer": "DE",
            "batch_id": batch_id,
            "run_index": run_index,
            "total_runs": total_runs,
            "pid_bounds": PID_BOUNDS,
            "base_mutation": _CONFIG["base_mutation"],
            "max_iterations": _CONFIG["max_iterations"],
            "target_yaw_deg": _CONFIG["target_yaw_deg"],
        }
        init_logger(config_data, log_path, pkl_path, config_path)

        # Initialize DE
        de = DifferentialEvolutionOptimizer(
            bounds=PID_BOUNDS,
            pop_size=6,
            mutation_factor=_CONFIG["base_mutation"],
        )

        # Safe seed
        safe_seed = np.array([b[0] for b in PID_BOUNDS], dtype=float)
        de.population[0] = safe_seed

        best_overall_fit = float("inf")
        best_record = None

        # Get robot-specific output limit and constraints BEFORE defining eval function
        robot_config = get_robot_config(_CONFIG, robot_type)
        pid_output_limit = robot_config["pid_output_limit"]
        max_overshoot = robot_config.get("pid_max_overshoot_pct", 5)
        max_rise_time = robot_config.get("pid_max_rise_time", 2)
        max_settling_time = robot_config.get("pid_max_settling_time", 5)
        max_sat_frac = robot_config.get("max_sat_fraction", None)

        def eval_obj_and_violation(params):
            """Evaluate fitness and constraint violation."""
            fit, hist, sat = self.sim.evaluate(
                params, return_history=True, realtime=False
            )
            # Calculate comprehensive violation including time constraints
            metrics = calculate_metrics(hist, _CONFIG["target_yaw_deg"])
            violation = calculate_violation(
                sat, pid_output_limit, metrics,
                max_rise_time, max_settling_time, max_overshoot, max_sat_frac
            )
            return float(fit), float(violation)

        print(f"Initializing DE... mutation={de.mutation_factor:.2f}")

        iteration = 0
        try:
            while iteration < _CONFIG["max_iterations"]:
                iter_start = time.time()
                iteration += 1
                print(f"\n--- Iteration {iteration} ---")

                # Evolve
                cand, _, _ = de.evolve(eval_obj_and_violation)

                # Evaluate with history
                fit, hist, sat, trace = self.sim.evaluate(
                    cand,
                    label_text=f"DE Iter {iteration}",
                    return_history=True,
                    realtime=_CONFIG["display_realtime"],
                    return_full_trace=True,
                )
                
                # Save trace
                save_trace(exp_dir / "traces" / f"iter_{iteration}_DE.csv", trace)

                metrics = calculate_metrics(hist, _CONFIG["target_yaw_deg"])
                
                # Calculate comprehensive violation including time constraints
                violation = calculate_violation(
                    sat, pid_output_limit, metrics,
                    max_rise_time, max_settling_time, max_overshoot, max_sat_frac
                )

                target_ok = meets_pid_targets(
                    metrics, max_overshoot, max_rise_time, max_settling_time
                )
                safe_ok = violation <= 0.0
                best_overall_fit = min(best_overall_fit, float(fit))

                # Update best
                if best_record is None or (
                    safe_ok and fit < best_record["fit"]
                ):
                    best_record = {
                        "iteration": iteration,
                        "params": cand.tolist(),
                        "fit": float(fit),
                        "violation": float(violation),
                        "metrics": metrics,
                    }

                # Log
                de_best_fit, de_best_viol = de.best_scores()
                log_iteration(
                    log_path,
                    iteration,
                    "auto",
                    cand,
                    fit,
                    violation,
                    sat,
                    metrics,
                    target_ok,
                    cand,
                    fit,
                    violation,
                    sat,
                    metrics,
                    target_ok,
                    de.mutation_factor,
                    de_best_fit,
                    de_best_viol,
                    np.mean(np.std(de.population, axis=0)),
                    best_overall_fit,
                    np.zeros(3),
                    np.zeros(3),
                    "DE_autorun",
                    time.time() - iter_start,
                )

                append_histories_pickle(
                    pkl_path,
                    [
                        {
                            "iteration": iteration,
                            "label": "DE",
                            "params": cand.tolist(),
                            "fit": float(fit),
                            "violation": float(violation),
                            "metrics": metrics,
                            "history": hist,
                        }
                    ],
                )

                # Check termination
                if target_ok and safe_ok:
                    print("[Terminate] Targets met!")
                    break

        finally:
            if best_record:
                best_path.write_text(json.dumps(best_record, indent=2))
                print(f"✓ Best saved to {best_path}")

        return best_path
