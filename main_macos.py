import csv
import json
import time
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import multiprocessing as mp

from differential_evolution import DifferentialEvolutionOptimizer
from bayesian_optimization import BayesianOptimizer
from visualizer_macos import FeedbackGUI


# --- CONFIGURATION ---
PID_BOUNDS = [(0.1, 100), (0.01, 50.0), (0.01, 50.0)]  # enforce nonzero Kp/Ki/Kd
SIMULATION_STEPS = 2500
DT = 1.0 / 240.0
BASE_MUTATION = 0.5
PREFERENCE_LR = 0.3
MAX_ITERATIONS = 100  # iteration cap (can be adjusted)
PID_MAX_OVERSHOOT_PCT = 1  # set None to disable
PID_MAX_RISE_TIME = 1      # seconds; set None to disable
PID_MAX_SETTLING_TIME = 1  # seconds; set None to disable
DISPLAY_REALTIME = False  # set False to show candidates fast (no realtime sleep)
PID_OUTPUT_LIMIT = 255.0  # max PWM command
PID_SAT_PENALTY = 0.01  # cost weight for exceeding the output limit
PID_STRICT_OUTPUT_LIMIT = True  # penalize any candidate that exceeds the limit
PID_SAT_HARD_PENALTY = 10000.0  # cost added when the limit is exceeded
START_TIME = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
EXP_DIR = Path("logs/mixed") / f"mixed_{START_TIME}"
LOG_PATH = EXP_DIR / "iteration_log.csv"
PKL_PATH = EXP_DIR / "iteration_log.pkl"
CONFIG_PATH = EXP_DIR / "config.yaml"
BEST_PATH = EXP_DIR / "best_results.json"


def calculate_metrics(history, target_val):
    time_arr = np.array(history['time'])
    actual_arr = np.array(history['actual'])

    max_val = np.max(actual_arr)
    overshoot = 0.0
    if max_val > target_val:
        overshoot = ((max_val - target_val) / target_val) * 100.0

    try:
        t_10_idx = np.where(actual_arr >= 0.1 * target_val)[0][0]
        t_90_idx = np.where(actual_arr >= 0.9 * target_val)[0][0]
        rise_time = time_arr[t_90_idx] - time_arr[t_10_idx]
    except IndexError:
        rise_time = -1.0

    tolerance = 0.05 * target_val
    upper_bound = target_val + tolerance
    lower_bound = target_val - tolerance
    out_of_bounds = np.where((actual_arr > upper_bound) | (actual_arr < lower_bound))[0]

    if len(out_of_bounds) == 0:
        settling_time = 0.0
    elif out_of_bounds[-1] == len(actual_arr) - 1:
        settling_time = -1.0
    else:
        settling_time = time_arr[out_of_bounds[-1] + 1]

    return {"overshoot": overshoot, "rise_time": rise_time, "settling_time": settling_time}


def meets_pid_targets(metrics):
    """
    Returns True if the metrics satisfy configured PID termination thresholds.
    """
    if PID_MAX_OVERSHOOT_PCT is not None and metrics["overshoot"] > PID_MAX_OVERSHOOT_PCT:
        return False
    if PID_MAX_RISE_TIME is not None:
        if metrics["rise_time"] <= 0 or metrics["rise_time"] > PID_MAX_RISE_TIME:
            return False
    if PID_MAX_SETTLING_TIME is not None:
        if metrics["settling_time"] <= 0 or metrics["settling_time"] > PID_MAX_SETTLING_TIME:
            return False
    return True


def init_logger():
    """Create the CSV log file with headers if it does not exist."""
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    config_data = {
        "start_time_utc": START_TIME,
        "pid_bounds": PID_BOUNDS,
        "simulation_steps": SIMULATION_STEPS,
        "dt": DT,
        "base_mutation": BASE_MUTATION,
        "preference_lr": PREFERENCE_LR,
        "max_iterations": MAX_ITERATIONS,
        "pid_max_overshoot_pct": PID_MAX_OVERSHOOT_PCT,
        "pid_max_rise_time": PID_MAX_RISE_TIME,
        "pid_max_settling_time": PID_MAX_SETTLING_TIME,
        "display_realtime": DISPLAY_REALTIME,
        "pid_output_limit": PID_OUTPUT_LIMIT,
        "pid_sat_penalty": PID_SAT_PENALTY,
        "pid_strict_output_limit": PID_STRICT_OUTPUT_LIMIT,
        "pid_sat_hard_penalty": PID_SAT_HARD_PENALTY,
    }
    CONFIG_PATH.write_text(json.dumps(config_data, indent=2))
    if not LOG_PATH.exists():
        with LOG_PATH.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "iteration",
                "choice",
                "cand_a_kp",
                "cand_a_ki",
                "cand_a_kd",
                "fit_a",
                "overshoot_a",
                "rise_time_a",
                "settling_time_a",
                "target_ok_a",
                "cand_b_kp",
                "cand_b_ki",
                "cand_b_kd",
                "fit_b",
                "overshoot_b",
                "rise_time_b",
                "settling_time_b",
                "target_ok_b",
                "de_mutation",
                "de_best_fit",
                "de_pop_std",
                "best_overall_fit",
                "bo_span_kp",
                "bo_span_ki",
                "bo_span_kd",
                "pref_weights",
                "gap_note",
                "iter_seconds",
            ])
    # Initialize pickle history file
    if not PKL_PATH.exists():
        with PKL_PATH.open("wb") as f:
            pickle.dump([], f)


def log_iteration(iteration, choice_label, cand_a, fit_a, metrics_a, cand_b, fit_b, metrics_b, target_ok_a, target_ok_b, de_mutation, de_best_fit, de_pop_std, best_overall_fit, bo_spans, pref_weights, gap_note="", iter_seconds=0.0):
    """Append one iteration of results to CSV."""
    ts = datetime.utcnow().isoformat()
    row = [
        ts,
        iteration,
        choice_label,
        *[float(x) for x in cand_a],
        float(fit_a),
        float(metrics_a["overshoot"]),
        float(metrics_a["rise_time"]),
        float(metrics_a["settling_time"]),
        int(bool(target_ok_a)),
        *[float(x) for x in cand_b],
        float(fit_b),
        float(metrics_b["overshoot"]),
        float(metrics_b["rise_time"]),
        float(metrics_b["settling_time"]),
        int(bool(target_ok_b)),
        float(de_mutation),
        float(de_best_fit),
        float(de_pop_std),
        float(best_overall_fit),
        float(bo_spans[0]),
        float(bo_spans[1]),
        float(bo_spans[2]),
        "|".join(f"{w:.4f}" for w in pref_weights),
        gap_note,
        float(iter_seconds),
    ]
    with LOG_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def append_histories_pickle(records):
    """
    Append candidate histories to the pickle file.
    Each record should be a dict containing iteration, label, params, fit, metrics, and history.
    """
    try:
        with PKL_PATH.open("rb") as f:
            existing = pickle.load(f)
    except FileNotFoundError:
        existing = []
    existing.extend(records)
    with PKL_PATH.open("wb") as f:
        pickle.dump(existing, f)


class PreferenceModel:
    """
    Tracks human-driven weights and an anchor PID derived from them.
    This gives effect to the flowchart's 'Calc Gap / Update Weights / Normalize' steps.
    """
    def __init__(self, bounds, lr=PREFERENCE_LR):
        self.bounds = np.array(bounds, dtype=float)
        self.lr = lr
        self.weights = self._init_weights()

    def _init_weights(self):
        # Start with random normalized weights (0-1) so the anchor is inside bounds.
        return np.random.rand(len(self.bounds))

    def _normalize(self):
        self.weights = np.clip(self.weights, 0.0, 1.0)

    def anchor_params(self):
        """Map weights back into PID space."""
        min_b = self.bounds[:, 0]
        max_b = self.bounds[:, 1]
        return min_b + self.weights * (max_b - min_b)

    def update_towards(self, preferred, other):
        """
        Shift weights toward the preferred candidate using the normalized gap (other - preferred).
        Returns the computed gap for logging.
        """
        preferred = np.array(preferred, dtype=float)
        other = np.array(other, dtype=float)
        min_b = self.bounds[:, 0]
        span = self.bounds[:, 1] - min_b

        pref_norm = (preferred - min_b) / (span + 1e-9)
        other_norm = (other - min_b) / (span + 1e-9)
        gap = other_norm - pref_norm  # aligns with flowchart "Calc Gap"

        # Move weights toward the preferred point, then normalize.
        self.weights = self.weights + self.lr * (pref_norm - self.weights)
        self._normalize()
        return gap


# ------------------ PyBullet worker (separate process) ------------------

def bullet_worker(req_q: mp.Queue, resp_q: mp.Queue):
    import pybullet as p
    import pybullet_data

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.resetDebugVisualizerCamera(
        cameraDistance=2.0, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 0]
    )

    def evaluate_pid(pid_params, label_text="", return_history=False, realtime=False):
        Kp, Ki, Kd = pid_params

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        start_pos = [0, 0, 0.02]  # near-ground to avoid "flying" spawn
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        robotId = p.loadURDF("husky/husky.urdf", start_pos, start_orn)

        # Suppress warnings & stability tweaks
        for i in range(p.getNumJoints(robotId)):
            info = p.getJointInfo(robotId, i)
            link_name = info[12].decode("utf-8")
            if any(n in link_name for n in ["imu", "plate", "rail", "bumper"]):
                p.changeDynamics(robotId, i, mass=0)

        left_wheels = [2, 4]
        right_wheels = [3, 5]
        for joint in left_wheels + right_wheels:
            p.changeDynamics(robotId, joint, lateralFriction=2.0)

        # Let the robot settle on the ground before applying control.
        settle_steps = 120
        for joint in left_wheels + right_wheels:
            p.setJointMotorControl2(robotId, joint, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        for _ in range(settle_steps):
            p.stepSimulation()

        if label_text:
            p.addUserDebugText(label_text, [0, 0, 1.0], textColorRGB=[0, 0, 0], textSize=2.0)

        target_yaw_deg = 90.0
        target_yaw_rad = np.deg2rad(target_yaw_deg)

        integral_error = 0.0
        prev_error = 0.0
        total_cost = 0.0

        history = {"time": [], "target": [], "actual": []}
        max_abs_raw_output = 0.0

        for i in range(SIMULATION_STEPS):
            p.stepSimulation()

            _, current_orn = p.getBasePositionAndOrientation(robotId)
            current_yaw = p.getEulerFromQuaternion(current_orn)[2]

            error = target_yaw_rad - current_yaw
            integral_error += error * DT
            derivative_error = (error - prev_error) / DT

            raw_output = (Kp * error) + (Ki * integral_error) + (Kd * derivative_error)
            abs_raw_output = abs(raw_output)
            if abs_raw_output > max_abs_raw_output:
                max_abs_raw_output = abs_raw_output
            turn_speed = float(np.clip(raw_output, -PID_OUTPUT_LIMIT, PID_OUTPUT_LIMIT))

            # Anti-windup: don't integrate further when saturated in the error direction.
            if raw_output != turn_speed and np.sign(raw_output) == np.sign(error):
                integral_error -= error * DT

            for j in left_wheels:
                p.setJointMotorControl2(robotId, j, p.VELOCITY_CONTROL, targetVelocity=-turn_speed, force=50)
            for j in right_wheels:
                p.setJointMotorControl2(robotId, j, p.VELOCITY_CONTROL, targetVelocity=turn_speed, force=50)

            saturation_excess = max(0.0, abs_raw_output - PID_OUTPUT_LIMIT)
            total_cost += (error ** 2) + (0.001 * (turn_speed ** 2)) + (PID_SAT_PENALTY * (saturation_excess ** 2))
            prev_error = error

            if return_history:
                history["time"].append(i * DT)
                history["target"].append(target_yaw_deg)
                history["actual"].append(float(np.rad2deg(current_yaw)))

            # realtime only for the “demo” runs
            if realtime:
                time.sleep(DT)

        if PID_STRICT_OUTPUT_LIMIT and max_abs_raw_output > PID_OUTPUT_LIMIT:
            excess = max_abs_raw_output - PID_OUTPUT_LIMIT
            total_cost += PID_SAT_HARD_PENALTY * (1.0 + (excess / PID_OUTPUT_LIMIT))
        fitness = total_cost / SIMULATION_STEPS
        if return_history:
            return fitness, history
        return fitness, None

    # Simple RPC loop
    running = True
    while running:
        msg = req_q.get()

        if msg["type"] == "shutdown":
            running = False
            break

        if msg["type"] == "eval":
            job_id = msg["id"]
            params = msg["params"]
            label = msg.get("label_text", "")
            return_history = msg.get("return_history", False)
            realtime = msg.get("realtime", False)

            fitness, history = evaluate_pid(
                params, label_text=label, return_history=return_history, realtime=realtime
            )

            resp_q.put({
                "id": job_id,
                "fitness": float(fitness),
                "history": history,  # dict of lists or None
            })

    p.disconnect()


# ------------------ Main app (Tk + optimizers) ------------------

def main():
    mp.set_start_method("spawn", force=True)  # IMPORTANT on macOS

    req_q = mp.Queue()
    resp_q = mp.Queue()
    worker = mp.Process(target=bullet_worker, args=(req_q, resp_q), daemon=False)
    worker.start()

    # --- RPC helper ---
    next_id = 0
    def eval_in_bullet(params, label_text="", return_history=False, realtime=False):
        nonlocal next_id
        next_id += 1
        job_id = next_id
        req_q.put({
            "type": "eval",
            "id": job_id,
            "params": [float(x) for x in params],
            "label_text": label_text,
            "return_history": return_history,
            "realtime": realtime
        })
        # Wait for matching response
        while True:
            resp = resp_q.get()
            if resp["id"] == job_id:
                return resp["fitness"], resp["history"]

    pref_model = PreferenceModel(PID_BOUNDS, lr=PREFERENCE_LR)
    anchor_seed = pref_model.anchor_params()

    de = DifferentialEvolutionOptimizer(bounds=PID_BOUNDS, pop_size=6, mutation_factor=BASE_MUTATION)
    de.population[0] = anchor_seed  # initialize with the anchored weights

    bo = BayesianOptimizer(bounds=PID_BOUNDS)
    gui = FeedbackGUI()

    print(f"Initializing Optimization System... (mutation={de.mutation_factor:.2f}, anchor seed={anchor_seed})")
    init_logger()

    best_overall_fit = np.inf
    best_record = None
    prev_de_histories = []
    prev_bo_histories = []

    # Warm-start BO with the same candidates as the DE population.
    for idx, cand in enumerate(de.population):
        fit, _ = eval_in_bullet(cand, return_history=False, realtime=False)
        bo.update(cand, fit)
        best_overall_fit = min(best_overall_fit, fit)
        print(f"[Init BO] Seed {idx}: params={cand}, fitness={fit:.4f}")

    iteration = 0
    try:
        while True:
            iter_start = time.time()
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")

            # Wrapper for optimization
            def fitness_wrapper(params):
                # FAST: no history, no realtime sleep
                fit, _ = eval_in_bullet(params, return_history=False, realtime=False)
                return fit

            # 1) Generate Candidates
            cand_a, fit_a = de.evolve(fitness_wrapper)
            cand_b = bo.propose_location()
            fit_b = fitness_wrapper(cand_b)

            bo.update(cand_b, fit_b)
            bo.update(cand_a, fit_a)

            # 2) Simulate & Visualize (realtime ON so you see it)
            print("Simulating A (DE)...")
            _, hist_a = eval_in_bullet(
                cand_a,
                label_text="DE CANDIDATE",
                return_history=True,
                realtime=DISPLAY_REALTIME,
            )

            print("Simulating B (BO)...")
            _, hist_b = eval_in_bullet(
                cand_b,
                label_text="BO CANDIDATE",
                return_history=True,
                realtime=DISPLAY_REALTIME,
            )

            # 3) Metrics
            metrics_a = calculate_metrics(hist_a, 90.0)
            metrics_b = calculate_metrics(hist_b, 90.0)
            target_ok_a = meets_pid_targets(metrics_a)
            target_ok_b = meets_pid_targets(metrics_b)

            best_overall_fit = min(best_overall_fit, fit_a, fit_b)

            # Update best record
            for label, cand, fit, metrics in [
                ("DE", cand_a, fit_a, metrics_a),
                ("BO", cand_b, fit_b, metrics_b),
            ]:
                if best_record is None or fit < best_record["fit"]:
                    best_record = {
                        "iteration": iteration,
                        "label": label,
                        "params": [float(x) for x in cand],
                        "fit": float(fit),
                        "metrics": metrics,
                    }

            # Optimizer stats
            finite_fit = de.fitness_scores[np.isfinite(de.fitness_scores)]
            de_best_fit = float(np.min(finite_fit)) if finite_fit.size > 0 else float("inf")
            de_pop_std = float(np.mean(np.std(de.population, axis=0))) if de.population.size else 0.0
            bo_spans = bo.bounds[:, 1] - bo.bounds[:, 0]

            # 3.5) Auto-termination based on PID targets (before GUI)
            terminate_reason = None
            terminate_label = None
            if target_ok_a:
                terminate_reason = "DE candidate met PID targets"
                terminate_label = "auto_terminate_de"
            elif target_ok_b:
                terminate_reason = "BO candidate met PID targets"
                terminate_label = "auto_terminate_bo"

            if terminate_reason:
                print(f"[Terminate] {terminate_reason}")
                log_iteration(
                    iteration=iteration,
                    choice_label=terminate_label,
                    cand_a=cand_a,
                    fit_a=fit_a,
                    metrics_a=metrics_a,
                    cand_b=cand_b,
                    fit_b=fit_b,
                    metrics_b=metrics_b,
                    target_ok_a=target_ok_a,
                    target_ok_b=target_ok_b,
                    de_mutation=de.mutation_factor,
                    de_best_fit=de_best_fit,
                    de_pop_std=de_pop_std,
                    best_overall_fit=best_overall_fit,
                    bo_spans=bo_spans,
                    pref_weights=pref_model.weights,
                    gap_note=terminate_reason,
                    iter_seconds=time.time() - iter_start,
                )
                # Save histories before exit
                append_histories_pickle([
                    {"iteration": iteration, "label": "DE", "params": cand_a, "fit": fit_a, "metrics": metrics_a, "history": hist_a},
                    {"iteration": iteration, "label": "BO", "params": cand_b, "fit": fit_b, "metrics": metrics_b, "history": hist_b},
                ])
                break

            # 4) GUI Feedback (blocks until click)
            choice = gui.show_comparison(hist_a, hist_b, cand_a, cand_b, fit_a, fit_b, metrics_a, metrics_b, prev_de_histories, prev_bo_histories)

            gap_note = ""
            choice_label = {1: "prefer_de", 2: "prefer_bo", 3: "tie_refine", 4: "reject_expand"}.get(choice, "exit")
            if choice == 1:
                print("User Preferred: DE")
                gap = pref_model.update_towards(cand_a, cand_b)
                anchor = pref_model.anchor_params()
                de.mutation_factor = BASE_MUTATION  # reset per flowchart
                de.population[np.random.randint(0, de.pop_size)] = anchor
                bo.nudge_with_preference(cand_a, fit_a, fit_b)
                gap_note = "B-A:" + ",".join(f"{g:.4f}" for g in gap)
                print(f"[Flow] Gap (B-A)={gap}, anchor={anchor}, mutation reset to {de.mutation_factor:.2f}")
            elif choice == 2:
                print("User Preferred: BO")
                gap = pref_model.update_towards(cand_b, cand_a)
                anchor = pref_model.anchor_params()
                de.mutation_factor = BASE_MUTATION  # reset per flowchart
                de.population[np.random.randint(0, de.pop_size)] = cand_b
                de.population[np.random.randint(0, de.pop_size)] = anchor
                bo.nudge_with_preference(cand_b, fit_b, fit_a)
                gap_note = "A-B:" + ",".join(f"{g:.4f}" for g in gap)
                print(f"[Flow] Gap (A-B)={gap}, anchor={anchor}, mutation reset to {de.mutation_factor:.2f}")
            elif choice == 3:
                print("TIE: Refine Mode")
                avg_c = (cand_a + cand_b) / 2
                de.refine_search_space(avg_c)
                bo.refine_bounds(avg_c)
                gap_note = "tie"
            elif choice == 4:
                print("REJECT: Expand Mode")
                de.expand_search_space()
                bo.expand_bounds()
                gap_note = "reject"
            else:
                break

            log_iteration(
                iteration=iteration,
                choice_label=choice_label,
                cand_a=cand_a,
                fit_a=fit_a,
                metrics_a=metrics_a,
                cand_b=cand_b,
                fit_b=fit_b,
                metrics_b=metrics_b,
                target_ok_a=target_ok_a,
                target_ok_b=target_ok_b,
                de_mutation=de.mutation_factor,
                de_best_fit=de_best_fit,
                de_pop_std=de_pop_std,
                best_overall_fit=best_overall_fit,
                bo_spans=bo_spans,
                pref_weights=pref_model.weights,
                gap_note=gap_note,
                iter_seconds=time.time() - iter_start,
            )
            # Save current histories to pickle store
            append_histories_pickle([
                {"iteration": iteration, "label": "DE", "params": cand_a, "fit": fit_a, "metrics": metrics_a, "history": hist_a},
                {"iteration": iteration, "label": "BO", "params": cand_b, "fit": fit_b, "metrics": metrics_b, "history": hist_b},
            ])

            # Track histories for future fading plots
            prev_de_histories.append(hist_a)
            prev_bo_histories.append(hist_b)

            if iteration >= MAX_ITERATIONS:
                break

    finally:
        req_q.put({"type": "shutdown"})
        worker.join(timeout=3)
        if best_record:
            with BEST_PATH.open("w") as f:
                json.dump(best_record, f, indent=2)


if __name__ == "__main__":
    main()
