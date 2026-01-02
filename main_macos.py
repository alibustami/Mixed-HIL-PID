import csv
import json
import time
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import multiprocessing as mp


# ------------------ CONFIGURATION ------------------

# Enforce positive Kp/Ki/Kd via bounds; also used by optimizers.
PID_BOUNDS = [(0.1, 5.0), (0.01, 5.0), (0.01, 5.0)]

SIMULATION_STEPS = 2500
DT = 1.0 / 240.0

# Differential Evolution params
BASE_MUTATION = 0.5

# Preference anchor model
PREFERENCE_LR = 0.3

# Iteration cap
MAX_ITERATIONS = 100

# PID target thresholds (set None to disable)
PID_MAX_OVERSHOOT_PCT = 5.0
PID_MAX_RISE_TIME = 1.0       # seconds
PID_MAX_SETTLING_TIME = 2.0   # seconds

# Visualization pacing
DISPLAY_REALTIME = False  # set True to see real-time motion (sleeps DT)

# Actuator (PWM) limit
PID_OUTPUT_LIMIT = 255.0

# "Soft" penalty for exceeding saturation (still useful for shaping near the boundary)
PID_SAT_PENALTY = 0.01

# Optional hard penalty inside objective (in addition to explicit constraints).
# With explicit constraints enabled, leaving this False usually makes the GP behave better.
PID_STRICT_OUTPUT_LIMIT = False
PID_SAT_HARD_PENALTY = 10000.0

# Constrained BO config: minimum probability-of-feasibility required for proposals
BO_POF_MIN = 0.95
# If BO proposes an infeasible candidate anyway, retry a few times (each infeasible eval updates BO's constraint GP).
BO_MAX_RETRIES = 5

START_TIME = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
EXP_DIR = Path("logs/mixed") / f"mixed_{START_TIME}"
LOG_PATH = EXP_DIR / "iteration_log.csv"
PKL_PATH = EXP_DIR / "iteration_log.pkl"
CONFIG_PATH = EXP_DIR / "config.yaml"
BEST_PATH = EXP_DIR / "best_results.json"


# ------------------ Metrics & termination ------------------

def calculate_metrics(history, target_val):
    time_arr = np.array(history["time"])
    actual_arr = np.array(history["actual"])

    max_val = float(np.max(actual_arr))
    overshoot = 0.0
    if max_val > target_val:
        overshoot = ((max_val - target_val) / target_val) * 100.0

    try:
        t_10_idx = np.where(actual_arr >= 0.1 * target_val)[0][0]
        t_90_idx = np.where(actual_arr >= 0.9 * target_val)[0][0]
        rise_time = float(time_arr[t_90_idx] - time_arr[t_10_idx])
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
        settling_time = float(time_arr[out_of_bounds[-1] + 1])

    return {"overshoot": float(overshoot), "rise_time": float(rise_time), "settling_time": float(settling_time)}


def meets_pid_targets(metrics):
    """Return True if the metrics satisfy configured PID termination thresholds."""
    if PID_MAX_OVERSHOOT_PCT is not None and metrics["overshoot"] > PID_MAX_OVERSHOOT_PCT:
        return False
    if PID_MAX_RISE_TIME is not None:
        if metrics["rise_time"] <= 0 or metrics["rise_time"] > PID_MAX_RISE_TIME:
            return False
    if PID_MAX_SETTLING_TIME is not None:
        if metrics["settling_time"] <= 0 or metrics["settling_time"] > PID_MAX_SETTLING_TIME:
            return False
    return True


# ------------------ Logging ------------------

def init_logger():
    """Create the experiment dir + CSV log headers."""
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
        "bo_pof_min": BO_POF_MIN,
        "bo_max_retries": BO_MAX_RETRIES,
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

    if not PKL_PATH.exists():
        with PKL_PATH.open("wb") as f:
            pickle.dump([], f)


def log_iteration(
    iteration,
    choice_label,
    cand_a,
    fit_a,
    viol_a,
    sat_a,
    metrics_a,
    cand_b,
    fit_b,
    viol_b,
    sat_b,
    metrics_b,
    target_ok_a,
    target_ok_b,
    de_mutation,
    de_best_fit,
    de_best_viol,
    de_pop_std,
    best_overall_fit,
    bo_spans,
    pref_weights,
    gap_note="",
    iter_seconds=0.0,
):
    """Append one iteration of results to CSV."""
    ts = datetime.utcnow().isoformat()
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
    with LOG_PATH.open("a", newline="") as f:
        csv.writer(f).writerow(row)


def append_histories_pickle(records):
    """Append candidate histories to a pickle file for later analysis/plotting."""
    try:
        with PKL_PATH.open("rb") as f:
            existing = pickle.load(f)
    except FileNotFoundError:
        existing = []
    existing.extend(records)
    with PKL_PATH.open("wb") as f:
        pickle.dump(existing, f)


# ------------------ Preference model ------------------

class PreferenceModel:
    """Tracks human-driven weights and an anchor PID derived from them."""
    def __init__(self, bounds, lr=PREFERENCE_LR):
        self.bounds = np.array(bounds, dtype=float)
        self.lr = float(lr)
        self.weights = self._init_weights()

    def _init_weights(self):
        return np.random.rand(len(self.bounds))

    def _normalize(self):
        self.weights = np.clip(self.weights, 0.0, 1.0)

    def anchor_params(self):
        min_b = self.bounds[:, 0]
        max_b = self.bounds[:, 1]
        return min_b + self.weights * (max_b - min_b)

    def update_towards(self, preferred, other):
        preferred = np.array(preferred, dtype=float)
        other = np.array(other, dtype=float)
        min_b = self.bounds[:, 0]
        span = self.bounds[:, 1] - min_b

        pref_norm = (preferred - min_b) / (span + 1e-9)
        other_norm = (other - min_b) / (span + 1e-9)
        gap = other_norm - pref_norm

        self.weights = self.weights + self.lr * (pref_norm - self.weights)
        self._normalize()
        return gap


# ------------------ PyBullet worker (separate process) ------------------

def bullet_worker(req_q: mp.Queue, resp_q: mp.Queue):
    """
    Runs PyBullet in its own process (required on macOS due to GUI constraints).
    Communicates via a simple request/response queue protocol.
    """
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
        """
        Evaluates a PID candidate and returns:
          - objective fitness (scalar)
          - history dict (if return_history)
          - sat_info dict: max_abs_raw_output, sat_fraction
        """
        Kp, Ki, Kd = [float(x) for x in pid_params]

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        start_pos = [0, 0, 0.02]
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

        # Let the robot settle
        settle_steps = 120
        for joint in left_wheels + right_wheels:
            p.setJointMotorControl2(robotId, joint, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        for _ in range(settle_steps):
            p.stepSimulation()

        if label_text:
            p.addUserDebugText(label_text, [0, 0, 1.0], textColorRGB=[0, 0, 0], textSize=2.0)

        target_yaw_deg = 90.0
        target_yaw_rad = float(np.deg2rad(target_yaw_deg))

        integral_error = 0.0
        total_cost = 0.0

        history = {"time": [], "target": [], "actual": []} if return_history else None

        max_abs_raw_output = 0.0
        sat_steps = 0

        # Derivative-on-measurement to reduce setpoint derivative kick
        prev_yaw = None

        for i in range(SIMULATION_STEPS):
            p.stepSimulation()

            _, current_orn = p.getBasePositionAndOrientation(robotId)
            current_yaw = float(p.getEulerFromQuaternion(current_orn)[2])  # radians

            error = target_yaw_rad - current_yaw

            if prev_yaw is None:
                prev_yaw = current_yaw

            # PID (I term on error; D term on measurement)
            integral_error += error * DT
            dyaw = (current_yaw - prev_yaw) / DT
            raw_output = (Kp * error) + (Ki * integral_error) - (Kd * dyaw)

            abs_raw = abs(raw_output)
            max_abs_raw_output = max(max_abs_raw_output, abs_raw)

            # Saturated command actually applied (actuator model)
            turn_speed = float(np.clip(raw_output, -PID_OUTPUT_LIMIT, PID_OUTPUT_LIMIT))

            if abs_raw > PID_OUTPUT_LIMIT:
                sat_steps += 1

            # Anti-windup: don't integrate further when saturated in the error direction
            if raw_output != turn_speed and np.sign(raw_output) == np.sign(error):
                integral_error -= error * DT

            # Apply turning
            for j in left_wheels:
                p.setJointMotorControl2(robotId, j, p.VELOCITY_CONTROL, targetVelocity=-turn_speed, force=50)
            for j in right_wheels:
                p.setJointMotorControl2(robotId, j, p.VELOCITY_CONTROL, targetVelocity=turn_speed, force=50)

            # Objective: tracking + control effort (plus optional soft saturation shaping)
            saturation_excess = max(0.0, abs_raw - PID_OUTPUT_LIMIT)
            total_cost += (error ** 2) + (0.001 * (turn_speed ** 2)) + (PID_SAT_PENALTY * (saturation_excess ** 2))

            # Optional hard penalty inside objective
            if PID_STRICT_OUTPUT_LIMIT and saturation_excess > 0:
                total_cost += PID_SAT_HARD_PENALTY * (saturation_excess / PID_OUTPUT_LIMIT)

            prev_yaw = current_yaw

            if return_history:
                history["time"].append(i * DT)
                history["target"].append(target_yaw_deg)
                history["actual"].append(float(np.rad2deg(current_yaw)))

            if realtime:
                time.sleep(DT)

        fitness = float(total_cost / SIMULATION_STEPS)
        sat_info = {
            "max_abs_raw_output": float(max_abs_raw_output),
            "sat_fraction": float(sat_steps / SIMULATION_STEPS),
        }
        return fitness, history, sat_info

    # RPC loop
    running = True
    while running:
        msg = req_q.get()

        if msg.get("type") == "shutdown":
            running = False
            break

        if msg.get("type") == "eval":
            job_id = msg["id"]
            params = msg["params"]
            label = msg.get("label_text", "")
            return_history = bool(msg.get("return_history", False))
            realtime = bool(msg.get("realtime", False))

            fitness, history, sat = evaluate_pid(
                params, label_text=label, return_history=return_history, realtime=realtime
            )

            resp_q.put({
                "id": job_id,
                "fitness": float(fitness),
                "history": history,
                "sat": sat,
            })

    p.disconnect()


# ------------------ Main app (Tk + optimizers) ------------------

def main():
    # Import optimizers lazily so the PyBullet worker process doesn't import heavy ML libs
    from differential_evolution import DifferentialEvolutionOptimizer
    from bayesian_optimization import ConstrainedBayesianOptimizer

    mp.set_start_method("spawn", force=True)  # IMPORTANT on macOS

    req_q = mp.Queue()
    resp_q = mp.Queue()
    worker = mp.Process(target=bullet_worker, args=(req_q, resp_q), daemon=False)
    worker.start()

    # --- RPC helper ---
    next_id = 0

    def eval_in_bullet(params, label_text="", return_history=False, realtime=False):
        """
        Evaluate params in the PyBullet process.
        Returns (fitness, history_or_None, sat_info_dict).
        """
        nonlocal next_id
        next_id += 1
        job_id = next_id
        req_q.put({
            "type": "eval",
            "id": job_id,
            "params": [float(x) for x in params],
            "label_text": label_text,
            "return_history": bool(return_history),
            "realtime": bool(realtime),
        })

        # Wait for matching response
        while True:
            resp = resp_q.get()
            if resp.get("id") == job_id:
                return float(resp["fitness"]), resp["history"], resp["sat"]

    # --- Helper: objective + constraint violation ---
    def eval_obj_and_violation(params):
        fit, _, sat = eval_in_bullet(params, return_history=False, realtime=False)
        violation = float(sat["max_abs_raw_output"] - PID_OUTPUT_LIMIT)  # <=0 is feasible
        return fit, violation

    # --- Setup models ---
    pref_model = PreferenceModel(PID_BOUNDS, lr=PREFERENCE_LR)
    anchor_seed = pref_model.anchor_params()

    # Conservative seed very likely feasible (bootstrap safe region)
    safe_seed = np.array([PID_BOUNDS[0][0], PID_BOUNDS[1][0], PID_BOUNDS[2][0]], dtype=float)

    de = DifferentialEvolutionOptimizer(bounds=PID_BOUNDS, pop_size=6, mutation_factor=BASE_MUTATION)
    # Seed DE with both anchor and conservative safe seed
    de.population[0] = anchor_seed
    if de.pop_size > 1:
        de.population[1] = safe_seed

    bo = ConstrainedBayesianOptimizer(bounds=PID_BOUNDS, pof_min=BO_POF_MIN)

    # Import GUI lazily so the PyBullet worker process doesn't import Tkinter
    from visualizer_macos import FeedbackGUI
    gui = FeedbackGUI()

    print(f"Initializing Optimization System... (DE mutation={de.mutation_factor:.2f})")
    print(f"Anchor seed: {anchor_seed}")
    init_logger()

    best_overall_fit = np.inf
    best_record = None

    prev_de_histories = []
    prev_bo_histories = []

    # --- Warm-start BO with safe seed + DE initial population (and initialize DE score arrays too) ---
    fit_safe, viol_safe = eval_obj_and_violation(safe_seed)
    bo.update(safe_seed, fit_safe, viol_safe)
    best_overall_fit = min(best_overall_fit, fit_safe)
    print(f"[Init] Safe seed params={safe_seed}, fitness={fit_safe:.4f}, viol={viol_safe:.3f}")

    for idx, cand in enumerate(de.population):
        fit, viol = eval_obj_and_violation(cand)
        # Populate DE score arrays so the first evolve() does not re-evaluate the entire population
        de.fitness_scores[idx] = fit
        de.violations[idx] = viol

        bo.update(cand, fit, viol)
        best_overall_fit = min(best_overall_fit, fit)
        print(f"[Init BO] Seed {idx}: params={cand}, fitness={fit:.4f}, viol={viol:.3f}")

    iteration = 0
    try:
        while True:
            iter_start = time.time()
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")

            # 1) Differential Evolution: one generation (constraint-aware)
            cand_a, fit_a, viol_a = de.evolve(eval_obj_and_violation)

            # 2) Constrained BO: propose, retry if infeasible (each infeasible eval updates BO)
            cand_b = None
            fit_b = None
            viol_b = None
            for attempt in range(1, BO_MAX_RETRIES + 1):
                proposal = bo.propose_location()
                f, v = eval_obj_and_violation(proposal)
                bo.update(proposal, f, v)
                if v <= 0.0:
                    cand_b, fit_b, viol_b = proposal, f, v
                    break
                print(f"[BO] Retry {attempt}/{BO_MAX_RETRIES}: infeasible viol={v:.3f} (updating constraint GP)")

            # Fallback if BO couldn't find feasible quickly: use best known feasible
            if cand_b is None:
                best = bo.best_feasible()
                if best is not None:
                    cand_b, fit_b, viol_b = best
                    print(f"[BO] Fallback to best feasible seen so far. fit={fit_b:.4f}, viol={viol_b:.3f}")
                else:
                    cand_b, fit_b, viol_b = safe_seed, fit_safe, viol_safe
                    print("[BO] Fallback to safe_seed (no feasible data found)")

            # Keep BO up-to-date with DE winner too
            bo.update(cand_a, fit_a, viol_a)

            best_overall_fit = min(best_overall_fit, fit_a, fit_b)

            # 3) Simulate & Visualize (history ON)
            print("Simulating A (DE)...")
            _, hist_a, sat_a = eval_in_bullet(
                cand_a,
                label_text="DE CANDIDATE",
                return_history=True,
                realtime=DISPLAY_REALTIME,
            )

            print("Simulating B (BO)...")
            _, hist_b, sat_b = eval_in_bullet(
                cand_b,
                label_text="BO CANDIDATE",
                return_history=True,
                realtime=DISPLAY_REALTIME,
            )

            # Violation from visualization run (for logging/termination)
            viol_a_vis = float(sat_a["max_abs_raw_output"] - PID_OUTPUT_LIMIT)
            viol_b_vis = float(sat_b["max_abs_raw_output"] - PID_OUTPUT_LIMIT)

            # 4) Metrics
            metrics_a = calculate_metrics(hist_a, 90.0)
            metrics_b = calculate_metrics(hist_b, 90.0)
            target_ok_a = meets_pid_targets(metrics_a)
            target_ok_b = meets_pid_targets(metrics_b)

            # Update best record (prefer feasible first, then lower fitness)
            def is_better_record(fit, viol, current):
                if current is None:
                    return True
                cur_fit = current["fit"]
                cur_viol = current.get("violation", 0.0)
                a_feas = viol <= 0.0
                b_feas = cur_viol <= 0.0
                if a_feas and not b_feas:
                    return True
                if b_feas and not a_feas:
                    return False
                if a_feas and b_feas:
                    return fit < cur_fit
                return viol < cur_viol

            for label, cand, fit, viol, metrics in [
                ("DE", cand_a, float(fit_a), float(viol_a_vis), metrics_a),
                ("BO", cand_b, float(fit_b), float(viol_b_vis), metrics_b),
            ]:
                if is_better_record(fit, viol, best_record):
                    best_record = {
                        "iteration": int(iteration),
                        "label": str(label),
                        "params": [float(x) for x in cand],
                        "fit": float(fit),
                        "violation": float(viol),
                        "metrics": metrics,
                    }

            # Optimizer stats
            de_best_fit, de_best_viol = de.best_scores()
            de_pop_std = float(np.mean(np.std(de.population, axis=0))) if de.population.size else 0.0
            bo_spans = bo.bounds[:, 1] - bo.bounds[:, 0]

            # 5) Auto-termination: require BOTH PID metrics and feasibility
            terminate_reason = None
            terminate_label = None
            if target_ok_a and (viol_a_vis <= 0.0):
                terminate_reason = "DE candidate met PID targets AND respected actuator limit"
                terminate_label = "auto_terminate_de"
            elif target_ok_b and (viol_b_vis <= 0.0):
                terminate_reason = "BO candidate met PID targets AND respected actuator limit"
                terminate_label = "auto_terminate_bo"

            if terminate_reason:
                print(f"[Terminate] {terminate_reason}")
                log_iteration(
                    iteration=iteration,
                    choice_label=terminate_label,
                    cand_a=cand_a,
                    fit_a=fit_a,
                    viol_a=viol_a_vis,
                    sat_a=sat_a,
                    metrics_a=metrics_a,
                    cand_b=cand_b,
                    fit_b=fit_b,
                    viol_b=viol_b_vis,
                    sat_b=sat_b,
                    metrics_b=metrics_b,
                    target_ok_a=target_ok_a,
                    target_ok_b=target_ok_b,
                    de_mutation=de.mutation_factor,
                    de_best_fit=de_best_fit,
                    de_best_viol=de_best_viol,
                    de_pop_std=de_pop_std,
                    best_overall_fit=best_overall_fit,
                    bo_spans=bo_spans,
                    pref_weights=pref_model.weights,
                    gap_note=terminate_reason,
                    iter_seconds=time.time() - iter_start,
                )

                append_histories_pickle([
                    {
                        "iteration": int(iteration),
                        "label": "DE",
                        "params": [float(x) for x in cand_a],
                        "fit": float(fit_a),
                        "violation": float(viol_a_vis),
                        "metrics": metrics_a,
                        "sat": sat_a,
                        "history": hist_a,
                    },
                    {
                        "iteration": int(iteration),
                        "label": "BO",
                        "params": [float(x) for x in cand_b],
                        "fit": float(fit_b),
                        "violation": float(viol_b_vis),
                        "metrics": metrics_b,
                        "sat": sat_b,
                        "history": hist_b,
                    },
                ])
                break

            # 6) GUI Feedback (blocks until click)
            choice = gui.show_comparison(
                hist_a, hist_b,
                cand_a, cand_b,
                fit_a, fit_b,
                metrics_a, metrics_b,
                prev_de_histories, prev_bo_histories
            )

            gap_note = ""
            choice_label = {1: "prefer_de", 2: "prefer_bo", 3: "tie_refine", 4: "reject_expand"}.get(choice, "exit")

            if choice == 1:
                print("User Preferred: DE")
                gap = pref_model.update_towards(cand_a, cand_b)
                anchor = pref_model.anchor_params()
                de.mutation_factor = BASE_MUTATION

                # Inject anchor without overwriting DE's best individual (keeps feasibility alive)
                de.inject_candidate(anchor, eval_func=eval_obj_and_violation, protect_best=True)

                bo.nudge_with_preference(
                    preferred=cand_a,
                    preferred_cost=float(fit_a),
                    other_cost=float(fit_b),
                    preferred_violation=float(viol_a),
                )

                gap_note = "B-A:" + ",".join(f"{g:.4f}" for g in gap)

            elif choice == 2:
                print("User Preferred: BO")
                gap = pref_model.update_towards(cand_b, cand_a)
                anchor = pref_model.anchor_params()
                de.mutation_factor = BASE_MUTATION

                # Inject BO point + anchor (with DE score consistency preserved)
                de.inject_candidate(cand_b, fitness=fit_b, violation=viol_b, protect_best=True)
                de.inject_candidate(anchor, eval_func=eval_obj_and_violation, protect_best=True)

                bo.nudge_with_preference(
                    preferred=cand_b,
                    preferred_cost=float(fit_b),
                    other_cost=float(fit_a),
                    preferred_violation=float(viol_b),
                )

                gap_note = "A-B:" + ",".join(f"{g:.4f}" for g in gap)

            elif choice == 3:
                print("TIE: Refine Mode")
                # Use the better of the two as the refine center (both are feasible by design)
                center = cand_a if float(fit_a) <= float(fit_b) else cand_b
                de.refine_search_space(center)
                bo.refine_bounds(center)
                gap_note = "tie"

            elif choice == 4:
                print("REJECT: Expand Mode")
                de.expand_search_space()
                bo.expand_bounds()
                gap_note = "reject"

            else:
                break

            # 7) Log iteration
            log_iteration(
                iteration=iteration,
                choice_label=choice_label,
                cand_a=cand_a,
                fit_a=fit_a,
                viol_a=viol_a_vis,
                sat_a=sat_a,
                metrics_a=metrics_a,
                cand_b=cand_b,
                fit_b=fit_b,
                viol_b=viol_b_vis,
                sat_b=sat_b,
                metrics_b=metrics_b,
                target_ok_a=target_ok_a,
                target_ok_b=target_ok_b,
                de_mutation=de.mutation_factor,
                de_best_fit=de_best_fit,
                de_best_viol=de_best_viol,
                de_pop_std=de_pop_std,
                best_overall_fit=best_overall_fit,
                bo_spans=bo_spans,
                pref_weights=pref_model.weights,
                gap_note=gap_note,
                iter_seconds=time.time() - iter_start,
            )

            append_histories_pickle([
                {
                    "iteration": int(iteration),
                    "label": "DE",
                    "params": [float(x) for x in cand_a],
                    "fit": float(fit_a),
                    "violation": float(viol_a_vis),
                    "metrics": metrics_a,
                    "sat": sat_a,
                    "history": hist_a,
                },
                {
                    "iteration": int(iteration),
                    "label": "BO",
                    "params": [float(x) for x in cand_b],
                    "fit": float(fit_b),
                    "violation": float(viol_b_vis),
                    "metrics": metrics_b,
                    "sat": sat_b,
                    "history": hist_b,
                },
            ])

            prev_de_histories.append(hist_a)
            prev_bo_histories.append(hist_b)

            if iteration >= MAX_ITERATIONS:
                break

    finally:
        try:
            req_q.put({"type": "shutdown"})
        except Exception:
            pass
        worker.join(timeout=3)

        if best_record:
            with BEST_PATH.open("w") as f:
                json.dump(best_record, f, indent=2)


if __name__ == "__main__":
    main()
