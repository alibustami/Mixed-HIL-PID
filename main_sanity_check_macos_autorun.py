import csv
import json
import time
import pickle
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
import numpy as np

from differential_evolution import DifferentialEvolutionOptimizer
from bayesian_optimization import ConstrainedBayesianOptimizer


# --- CONFIGURATION ---
OPTIMIZER_MODE = "BO"  # choose "DE" or "BO"

PID_BOUNDS = [(0.1, 10.0), (0.01, 10.0), (0.01, 10.0)]
SIMULATION_STEPS = 2500
DT = 1.0 / 240.0

# DE params
BASE_MUTATION = 0.5

# Iterations
MAX_ITERATIONS = 100

# Visualization pacing
DISPLAY_REALTIME = False

# Task
TARGET_YAW_DEG = 90.0
PID_MAX_OVERSHOOT_PCT = 5.0
PID_MAX_RISE_TIME = 1.0
PID_MAX_SETTLING_TIME = 2.0

# Actuator limit (PWM)
PID_OUTPUT_LIMIT = 255.0

# Soft shaping near saturation (objective still sees saturated command)
PID_SAT_PENALTY = 0.01

# Optional hard penalty *inside objective* (still logs explicit violation separately)
PID_STRICT_OUTPUT_LIMIT = True
PID_SAT_HARD_PENALTY = 10000.0

# BO warmstart + feasibility gating
INIT_BO_SEEDS = 6
BO_POF_MIN = 0.95
BO_MAX_RETRIES = 5

# Sequential experiments
EXPERIMENT_RUNS = 30
STATE_PATH = Path(f"logs/{OPTIMIZER_MODE}_seq") / "sanity_check_autorun_state.json"

# Per-run globals (set by init_experiment_paths)
BATCH_ID = None
START_TIME = None
EXP_DIR = None
LOG_PATH = None
PKL_PATH = None
CONFIG_PATH = None
BEST_PATH = None


# ------------------ Autorun state / paths ------------------

def init_experiment_paths(run_index, run_total, batch_id):
    global BATCH_ID, START_TIME, EXP_DIR, LOG_PATH, PKL_PATH, CONFIG_PATH, BEST_PATH
    BATCH_ID = batch_id
    START_TIME = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_suffix = f"_run{run_index:02d}" if run_total > 1 else ""
    EXP_DIR = Path(f"logs/{OPTIMIZER_MODE}_seq") / f"{OPTIMIZER_MODE}_{BATCH_ID}{run_suffix}"
    LOG_PATH = EXP_DIR / "iteration_log.csv"
    PKL_PATH = EXP_DIR / "iteration_log.pkl"
    CONFIG_PATH = EXP_DIR / "config.yaml"
    BEST_PATH = EXP_DIR / "best_results.json"


def load_autorun_state():
    try:
        return json.loads(STATE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_autorun_state(batch_id, last_completed_run):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "optimizer_mode": OPTIMIZER_MODE,
        "experiment_total": EXPERIMENT_RUNS,
        "batch_id": batch_id,
        "last_completed_run": int(last_completed_run),
        "updated_utc": datetime.utcnow().isoformat(),
    }
    STATE_PATH.write_text(json.dumps(state, indent=2))


def resolve_resume_state():
    state = load_autorun_state()
    if state and state.get("optimizer_mode") == OPTIMIZER_MODE:
        batch_id = state.get("batch_id")
        try:
            last_completed = int(state.get("last_completed_run", 0))
        except (TypeError, ValueError):
            last_completed = 0
        last_completed = max(last_completed, 0)
        if isinstance(batch_id, str) and batch_id:
            next_run = last_completed + 1
            if next_run <= EXPERIMENT_RUNS:
                return batch_id, next_run
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S"), 1


# ------------------ Metrics / termination ------------------

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

def init_logger(run_index, run_total):
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    config_data = {
        "start_time_utc": START_TIME,
        "batch_id": BATCH_ID,
        "experiment_index": int(run_index),
        "experiment_total": int(run_total),
        "optimizer_mode": OPTIMIZER_MODE,
        "pid_bounds": PID_BOUNDS,
        "simulation_steps": SIMULATION_STEPS,
        "dt": DT,
        "base_mutation": BASE_MUTATION,
        "max_iterations": MAX_ITERATIONS,
        "display_realtime": DISPLAY_REALTIME,
        "target_yaw_deg": TARGET_YAW_DEG,
        "pid_max_overshoot_pct": PID_MAX_OVERSHOOT_PCT,
        "pid_max_rise_time": PID_MAX_RISE_TIME,
        "pid_max_settling_time": PID_MAX_SETTLING_TIME,
        "pid_output_limit": PID_OUTPUT_LIMIT,
        "pid_sat_penalty": PID_SAT_PENALTY,
        "pid_strict_output_limit": PID_STRICT_OUTPUT_LIMIT,
        "pid_sat_hard_penalty": PID_SAT_HARD_PENALTY,
        "bo_pof_min": BO_POF_MIN,
        "bo_max_retries": BO_MAX_RETRIES,
        "init_bo_seeds": INIT_BO_SEEDS,
    }
    CONFIG_PATH.write_text(json.dumps(config_data, indent=2))

    if not LOG_PATH.exists():
        with LOG_PATH.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "iteration",
                "choice",
                "cand_kp",
                "cand_ki",
                "cand_kd",
                "fit",
                "violation",
                "max_abs_raw_output",
                "sat_fraction",
                "overshoot",
                "rise_time",
                "settling_time",
                "target_ok",
                "safe_ok",
                "de_mutation",
                "de_best_fit",
                "de_best_viol",
                "de_pop_std",
                "best_overall_fit",
                "bo_span_kp",
                "bo_span_ki",
                "bo_span_kd",
                "iter_seconds",
            ])

    if not PKL_PATH.exists():
        with PKL_PATH.open("wb") as f:
            pickle.dump([], f)


def log_iteration(
    iteration,
    choice_label,
    cand,
    fit,
    violation,
    sat_info,
    metrics,
    target_ok,
    safe_ok,
    de_mutation,
    de_best_fit,
    de_best_viol,
    de_pop_std,
    best_overall_fit,
    bo_spans,
    iter_seconds=0.0,
):
    ts = datetime.utcnow().isoformat()
    row = [
        ts,
        int(iteration),
        str(choice_label),
        float(cand[0]),
        float(cand[1]),
        float(cand[2]),
        float(fit),
        float(violation),
        float(sat_info["max_abs_raw_output"]),
        float(sat_info["sat_fraction"]),
        float(metrics["overshoot"]),
        float(metrics["rise_time"]),
        float(metrics["settling_time"]),
        int(bool(target_ok)),
        int(bool(safe_ok)),
        float(de_mutation),
        float(de_best_fit),
        float(de_best_viol),
        float(de_pop_std),
        float(best_overall_fit),
        float(bo_spans[0]),
        float(bo_spans[1]),
        float(bo_spans[2]),
        float(iter_seconds),
    ]
    with LOG_PATH.open("a", newline="") as f:
        csv.writer(f).writerow(row)


def append_histories_pickle(records):
    try:
        with PKL_PATH.open("rb") as f:
            existing = pickle.load(f)
    except FileNotFoundError:
        existing = []
    existing.extend(records)
    with PKL_PATH.open("wb") as f:
        pickle.dump(existing, f)


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
        """
        Returns:
          fitness (float),
          history (dict or None),
          sat_info (dict): {max_abs_raw_output, sat_fraction}
        """
        Kp, Ki, Kd = [float(x) for x in pid_params]

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        start_pos = [0, 0, 0.02]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        robotId = p.loadURDF("husky/husky.urdf", start_pos, start_orn)

        # Stability tweaks
        for i in range(p.getNumJoints(robotId)):
            info = p.getJointInfo(robotId, i)
            link_name = info[12].decode("utf-8")
            if any(n in link_name for n in ["imu", "plate", "rail", "bumper"]):
                p.changeDynamics(robotId, i, mass=0)

        left_wheels = [2, 4]
        right_wheels = [3, 5]
        for joint in left_wheels + right_wheels:
            p.changeDynamics(robotId, joint, lateralFriction=2.0)

        # Settle
        settle_steps = 120
        for joint in left_wheels + right_wheels:
            p.setJointMotorControl2(robotId, joint, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        for _ in range(settle_steps):
            p.stepSimulation()

        if label_text:
            p.addUserDebugText(label_text, [0, 0, 1.0], textColorRGB=[0, 0, 0], textSize=2.0)

        target_yaw_rad = float(np.deg2rad(TARGET_YAW_DEG))

        integral_error = 0.0
        total_cost = 0.0

        history = {"time": [], "target": [], "actual": []} if return_history else None

        max_abs_raw_output = 0.0
        sat_steps = 0

        # Derivative-on-measurement to reduce derivative kick
        prev_yaw = None

        for i in range(SIMULATION_STEPS):
            p.stepSimulation()

            _, current_orn = p.getBasePositionAndOrientation(robotId)
            current_yaw = float(p.getEulerFromQuaternion(current_orn)[2])

            if prev_yaw is None:
                prev_yaw = current_yaw

            error = target_yaw_rad - current_yaw
            integral_error += error * DT

            dyaw = (current_yaw - prev_yaw) / DT
            raw_output = (Kp * error) + (Ki * integral_error) - (Kd * dyaw)

            abs_raw = abs(raw_output)
            max_abs_raw_output = max(max_abs_raw_output, abs_raw)

            turn_speed = float(np.clip(raw_output, -PID_OUTPUT_LIMIT, PID_OUTPUT_LIMIT))

            if abs_raw > PID_OUTPUT_LIMIT:
                sat_steps += 1

            # Anti-windup: don't integrate further when saturated in error direction
            if raw_output != turn_speed and np.sign(raw_output) == np.sign(error):
                integral_error -= error * DT

            for j in left_wheels:
                p.setJointMotorControl2(robotId, j, p.VELOCITY_CONTROL, targetVelocity=-turn_speed, force=50)
            for j in right_wheels:
                p.setJointMotorControl2(robotId, j, p.VELOCITY_CONTROL, targetVelocity=turn_speed, force=50)

            saturation_excess = max(0.0, abs_raw - PID_OUTPUT_LIMIT)
            total_cost += (error ** 2) + (0.001 * (turn_speed ** 2)) + (PID_SAT_PENALTY * (saturation_excess ** 2))

            prev_yaw = current_yaw

            if return_history:
                history["time"].append(i * DT)
                history["target"].append(TARGET_YAW_DEG)
                history["actual"].append(float(np.rad2deg(current_yaw)))

            if realtime:
                time.sleep(DT)

        if PID_STRICT_OUTPUT_LIMIT and max_abs_raw_output > PID_OUTPUT_LIMIT:
            excess = max_abs_raw_output - PID_OUTPUT_LIMIT
            total_cost += PID_SAT_HARD_PENALTY * (1.0 + (excess / PID_OUTPUT_LIMIT))

        fitness = float(total_cost / SIMULATION_STEPS)
        sat_info = {
            "max_abs_raw_output": float(max_abs_raw_output),
            "sat_fraction": float(sat_steps / SIMULATION_STEPS),
        }

        return fitness, history, sat_info

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


# ------------------ Experiment runner ------------------

def run_experiment(eval_in_bullet, experiment_index, experiment_total, batch_id):
    init_experiment_paths(experiment_index, experiment_total, batch_id)
    init_logger(experiment_index, experiment_total)

    bo = ConstrainedBayesianOptimizer(bounds=PID_BOUNDS, pof_min=BO_POF_MIN) if OPTIMIZER_MODE == "BO" else None
    de = DifferentialEvolutionOptimizer(bounds=PID_BOUNDS, pop_size=6, mutation_factor=BASE_MUTATION) if OPTIMIZER_MODE == "DE" else None

    # A conservative “likely-feasible” seed
    safe_seed = np.array([b[0] for b in PID_BOUNDS], dtype=float)

    best_overall_fit = float("inf")
    best_record = None

    def eval_obj_and_violation(params):
        fit, _, sat = eval_in_bullet(params, return_history=False, realtime=False)
        viol = float(sat["max_abs_raw_output"] - PID_OUTPUT_LIMIT)  # <=0 feasible
        return float(fit), float(viol)

    # Warm start (BO)
    safe_fit, safe_viol = eval_obj_and_violation(safe_seed)
    if bo is not None:
        bo.update(safe_seed, safe_fit, safe_viol)
        print(f"[Init BO] safe_seed params={safe_seed}, fit={safe_fit:.4f}, viol={safe_viol:.3f}")

        lows = np.array([b[0] for b in PID_BOUNDS], dtype=float)
        highs = np.array([b[1] for b in PID_BOUNDS], dtype=float)
        init_points = np.random.uniform(low=lows, high=highs, size=(INIT_BO_SEEDS, len(PID_BOUNDS)))
        for idx, cand in enumerate(init_points):
            f, v = eval_obj_and_violation(cand)
            bo.update(cand, f, v)
            print(f"[Init BO] Seed {idx}: params={cand}, fit={f:.4f}, viol={v:.3f}")

    # Seed DE with safe point
    if de is not None:
        de.population[0] = safe_seed
        # Optionally pre-score to avoid re-evaluating all at first evolve()
        de.fitness_scores[0] = safe_fit
        de.violations[0] = safe_viol
        print(f"Initializing DE... mutation={de.mutation_factor:.2f}")

    iteration = 0
    try:
        while True:
            iter_start = time.time()
            iteration += 1
            print(f"\n--- Iteration {iteration} ({OPTIMIZER_MODE}) ---")

            # Select candidate (fast evals; no history)
            if OPTIMIZER_MODE == "DE":
                cand, fit_fast, viol_fast = de.evolve(eval_obj_and_violation)

            else:
                cand = None
                fit_fast = None
                viol_fast = None

                for attempt in range(1, BO_MAX_RETRIES + 1):
                    proposal = bo.propose_location()
                    f, v = eval_obj_and_violation(proposal)
                    bo.update(proposal, f, v)

                    if v <= 0.0:
                        cand, fit_fast, viol_fast = proposal, f, v
                        break
                    print(f"[BO] Retry {attempt}/{BO_MAX_RETRIES}: infeasible viol={v:.3f}")

                if cand is None:
                    best = bo.best_feasible()
                    if best is not None:
                        cand, fit_fast, viol_fast = best
                        print(f"[BO] Fallback to best feasible: fit={fit_fast:.4f}, viol={viol_fast:.3f}")
                    else:
                        cand, fit_fast, viol_fast = safe_seed, safe_fit, safe_viol
                        print("[BO] Fallback to safe_seed (no feasible points yet)")

            # Evaluate chosen candidate with history (for metrics/logging)
            fit_eval, hist, sat = eval_in_bullet(
                cand,
                label_text=f"{OPTIMIZER_MODE} CANDIDATE",
                return_history=True,
                realtime=DISPLAY_REALTIME,
            )

            violation_eval = float(sat["max_abs_raw_output"] - PID_OUTPUT_LIMIT)
            safe_ok = (violation_eval <= 0.0)

            metrics = calculate_metrics(hist, TARGET_YAW_DEG)
            target_ok = meets_pid_targets(metrics)

            best_overall_fit = min(best_overall_fit, float(fit_eval))

            # Best-record: prefer feasible first
            def record_better(fit, viol, cur):
                if cur is None:
                    return True
                cur_fit = float(cur["fit"])
                cur_viol = float(cur.get("violation", 0.0))
                a_feas = (viol <= 0.0)
                b_feas = (cur_viol <= 0.0)
                if a_feas and not b_feas:
                    return True
                if b_feas and not a_feas:
                    return False
                if a_feas and b_feas:
                    return fit < cur_fit
                return viol < cur_viol

            if record_better(float(fit_eval), float(violation_eval), best_record):
                best_record = {
                    "iteration": int(iteration),
                    "label": OPTIMIZER_MODE,
                    "params": [float(x) for x in cand],
                    "fit": float(fit_eval),
                    "violation": float(violation_eval),
                    "sat": sat,
                    "metrics": metrics,
                }

            # Stats
            if de is not None:
                de_best_fit, de_best_viol = de.best_scores()
                de_pop_std = float(np.mean(np.std(de.population, axis=0))) if de.population.size else 0.0
                de_mut = float(de.mutation_factor)
            else:
                de_best_fit, de_best_viol, de_pop_std, de_mut = float("inf"), float("inf"), 0.0, 0.0

            bo_spans = (bo.bounds[:, 1] - bo.bounds[:, 0]) if bo is not None else np.zeros(len(PID_BOUNDS), dtype=float)

            choice_label = "auto"
            if target_ok and safe_ok:
                choice_label = "auto_terminate_target_and_limit_met"
                print("[Terminate] PID targets satisfied AND actuator limit respected.")

            log_iteration(
                iteration=iteration,
                choice_label=choice_label,
                cand=cand,
                fit=fit_eval,
                violation=violation_eval,
                sat_info=sat,
                metrics=metrics,
                target_ok=target_ok,
                safe_ok=safe_ok,
                de_mutation=de_mut,
                de_best_fit=de_best_fit,
                de_best_viol=de_best_viol,
                de_pop_std=de_pop_std,
                best_overall_fit=best_overall_fit,
                bo_spans=bo_spans,
                iter_seconds=time.time() - iter_start,
            )

            append_histories_pickle([
                {
                    "iteration": int(iteration),
                    "label": OPTIMIZER_MODE,
                    "params": [float(x) for x in cand],
                    "fit": float(fit_eval),
                    "violation": float(violation_eval),
                    "sat": sat,
                    "metrics": metrics,
                    "history": hist,
                }
            ])

            if (target_ok and safe_ok) or iteration >= MAX_ITERATIONS:
                break

    finally:
        if best_record:
            with BEST_PATH.open("w") as f:
                json.dump(best_record, f, indent=2)


# ------------------ Main ------------------

def main():
    mp.set_start_method("spawn", force=True)

    req_q = mp.Queue()
    resp_q = mp.Queue()
    worker = mp.Process(target=bullet_worker, args=(req_q, resp_q), daemon=False)
    worker.start()

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
            "return_history": bool(return_history),
            "realtime": bool(realtime),
        })
        while True:
            resp = resp_q.get()
            if resp["id"] == job_id:
                return float(resp["fitness"]), resp["history"], resp["sat"]

    try:
        if EXPERIMENT_RUNS < 1:
            raise ValueError("EXPERIMENT_RUNS must be >= 1")

        batch_id, start_run = resolve_resume_state()
        save_autorun_state(batch_id, start_run - 1)

        if start_run > 1:
            print(f"Resuming batch {batch_id} from run {start_run}/{EXPERIMENT_RUNS}.")

        for experiment_index in range(start_run, EXPERIMENT_RUNS + 1):
            print(f"\n=== Experiment {experiment_index}/{EXPERIMENT_RUNS} ({OPTIMIZER_MODE}) ===")
            run_experiment(eval_in_bullet, experiment_index, EXPERIMENT_RUNS, batch_id)
            save_autorun_state(batch_id, experiment_index)

    finally:
        try:
            req_q.put({"type": "shutdown"})
        except Exception:
            pass
        worker.join(timeout=3)


if __name__ == "__main__":
    main()
