import csv
import json
import time
import pickle
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
import numpy as np

from differential_evolution import DifferentialEvolutionOptimizer
from bayesian_optimization import BayesianOptimizer


# --- CONFIGURATION ---
OPTIMIZER_MODE = "DE"  # choose "DE" or "BO"
PID_BOUNDS = [(0.1, 100), (0.01, 50.0), (0.01, 50.0)]  # enforce nonzero Kp/Ki/Kd
SIMULATION_STEPS = 2500
DT = 1.0 / 240.0
BASE_MUTATION = 0.5
MAX_ITERATIONS = 100
DISPLAY_REALTIME = False
TARGET_YAW_DEG = 90.0
PID_MAX_OVERSHOOT_PCT = 1
PID_MAX_RISE_TIME = 1
PID_MAX_SETTLING_TIME = 1
INIT_BO_SEEDS = 6

START_TIME = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
EXP_DIR = Path("logs") / f"{OPTIMIZER_MODE}_{START_TIME}"
LOG_PATH = EXP_DIR / "iteration_log.csv"
PKL_PATH = EXP_DIR / "iteration_log.pkl"
CONFIG_PATH = EXP_DIR / "config.yaml"
BEST_PATH = EXP_DIR / "best_results.json"


def calculate_metrics(history, target_val):
    time_arr = np.array(history["time"])
    actual_arr = np.array(history["actual"])

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
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    config_data = {
        "start_time_utc": START_TIME,
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
                "overshoot",
                "rise_time",
                "settling_time",
                "target_ok",
                "de_mutation",
                "de_best_fit",
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


def log_iteration(iteration, choice_label, cand, fit, metrics, target_ok, de_mutation, de_best_fit, de_pop_std, best_overall_fit, bo_spans, iter_seconds=0.0):
    ts = datetime.utcnow().isoformat()
    row = [
        ts,
        iteration,
        choice_label,
        float(cand[0]),
        float(cand[1]),
        float(cand[2]),
        float(fit),
        float(metrics["overshoot"]),
        float(metrics["rise_time"]),
        float(metrics["settling_time"]),
        int(bool(target_ok)),
        float(de_mutation),
        float(de_best_fit),
        float(de_pop_std),
        float(best_overall_fit),
        float(bo_spans[0]),
        float(bo_spans[1]),
        float(bo_spans[2]),
        float(iter_seconds),
    ]
    with LOG_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


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
        Kp, Ki, Kd = pid_params

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        start_pos = [0, 0, 0.02]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        robotId = p.loadURDF("husky/husky.urdf", start_pos, start_orn)

        for i in range(p.getNumJoints(robotId)):
            info = p.getJointInfo(robotId, i)
            link_name = info[12].decode("utf-8")
            if any(n in link_name for n in ["imu", "plate", "rail", "bumper"]):
                p.changeDynamics(robotId, i, mass=0)

        left_wheels = [2, 4]
        right_wheels = [3, 5]
        for joint in left_wheels + right_wheels:
            p.changeDynamics(robotId, joint, lateralFriction=2.0)

        settle_steps = 120
        for joint in left_wheels + right_wheels:
            p.setJointMotorControl2(robotId, joint, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        for _ in range(settle_steps):
            p.stepSimulation()

        if label_text:
            p.addUserDebugText(label_text, [0, 0, 1.0], textColorRGB=[0, 0, 0], textSize=2.0)

        target_yaw_rad = np.deg2rad(TARGET_YAW_DEG)
        integral_error = 0.0
        prev_error = 0.0
        total_cost = 0.0
        history = {"time": [], "target": [], "actual": []}

        for i in range(SIMULATION_STEPS):
            p.stepSimulation()
            _, current_orn = p.getBasePositionAndOrientation(robotId)
            current_yaw = p.getEulerFromQuaternion(current_orn)[2]

            error = target_yaw_rad - current_yaw
            integral_error += error * DT
            derivative_error = (error - prev_error) / DT

            turn_speed = (Kp * error) + (Ki * integral_error) + (Kd * derivative_error)
            turn_speed = float(np.clip(turn_speed, -40, 40))

            for j in left_wheels:
                p.setJointMotorControl2(robotId, j, p.VELOCITY_CONTROL, targetVelocity=-turn_speed, force=50)
            for j in right_wheels:
                p.setJointMotorControl2(robotId, j, p.VELOCITY_CONTROL, targetVelocity=turn_speed, force=50)

            total_cost += (error ** 2) + (0.001 * (turn_speed ** 2))
            prev_error = error

            if return_history:
                history["time"].append(i * DT)
                history["target"].append(TARGET_YAW_DEG)
                history["actual"].append(float(np.rad2deg(current_yaw)))

            if realtime:
                time.sleep(DT)

        fitness = total_cost / SIMULATION_STEPS
        if return_history:
            return fitness, history
        return fitness, None

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
                "history": history,
            })
    p.disconnect()


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
            "return_history": return_history,
            "realtime": realtime,
        })
        while True:
            resp = resp_q.get()
            if resp["id"] == job_id:
                return resp["fitness"], resp["history"]

    init_logger()

    bo = BayesianOptimizer(bounds=PID_BOUNDS) if OPTIMIZER_MODE == "BO" else None
    de = DifferentialEvolutionOptimizer(bounds=PID_BOUNDS, pop_size=6, mutation_factor=BASE_MUTATION) if OPTIMIZER_MODE == "DE" else None

    best_overall_fit = np.inf
    best_record = None
    prev_histories = []

    # Warm start
    if bo and not de:
        lows = np.array([b[0] for b in PID_BOUNDS])
        highs = np.array([b[1] for b in PID_BOUNDS])
        init_points = np.random.uniform(low=lows, high=highs, size=(INIT_BO_SEEDS, len(PID_BOUNDS)))
        for idx, cand in enumerate(init_points):
            fit, _ = eval_in_bullet(cand, return_history=False, realtime=False)
            bo.update(cand, fit)
            best_overall_fit = min(best_overall_fit, fit)
            print(f"[Init BO] Seed {idx}: params={cand}, fitness={fit:.4f}")

    if de:
        print(f"Initializing DE... mutation={de.mutation_factor:.2f}")
    if bo:
        print(f"Initializing BO with {INIT_BO_SEEDS if not de else 0} seeds")

    iteration = 0
    try:
        while True:
            iter_start = time.time()
            iteration += 1
            print(f"\n--- Iteration {iteration} ({OPTIMIZER_MODE}) ---")

            def fitness_wrapper(params):
                fit, _ = eval_in_bullet(params, return_history=False, realtime=False)
                return fit

            if OPTIMIZER_MODE == "DE":
                cand, fit_fast = de.evolve(fitness_wrapper)
            else:
                cand = bo.propose_location()
                fit_fast = fitness_wrapper(cand)
                bo.update(cand, fit_fast)

            fit_eval, hist = eval_in_bullet(
                cand,
                label_text=f"{OPTIMIZER_MODE} CANDIDATE",
                return_history=True,
                realtime=DISPLAY_REALTIME,
            )
            metrics = calculate_metrics(hist, TARGET_YAW_DEG)
            target_ok = meets_pid_targets(metrics)

            best_overall_fit = min(best_overall_fit, fit_eval)
            if best_record is None or fit_eval < best_record["fit"]:
                best_record = {
                    "iteration": iteration,
                    "label": OPTIMIZER_MODE,
                    "params": [float(x) for x in cand],
                    "fit": float(fit_eval),
                    "metrics": metrics,
                }

            finite_fit = de.fitness_scores[np.isfinite(de.fitness_scores)] if de else np.array([])
            de_best_fit = float(np.min(finite_fit)) if finite_fit.size > 0 else float("inf")
            de_pop_std = float(np.mean(np.std(de.population, axis=0))) if de else 0.0
            bo_spans = bo.bounds[:, 1] - bo.bounds[:, 0] if bo else np.zeros(len(PID_BOUNDS))

            choice_label = "auto"
            if target_ok:
                choice_label = "auto_terminate_target_met"
                print("[Terminate] PID targets satisfied.")

            log_iteration(
                iteration=iteration,
                choice_label=choice_label,
                cand=cand,
                fit=fit_eval,
                metrics=metrics,
                target_ok=target_ok,
                de_mutation=de.mutation_factor if de else 0.0,
                de_best_fit=de_best_fit,
                de_pop_std=de_pop_std,
                best_overall_fit=best_overall_fit,
                bo_spans=bo_spans,
                iter_seconds=time.time() - iter_start,
            )

            append_histories_pickle([
                {"iteration": iteration, "label": OPTIMIZER_MODE, "params": cand, "fit": fit_eval, "metrics": metrics, "history": hist},
            ])
            prev_histories.append(hist)

            if target_ok or iteration >= MAX_ITERATIONS:
                break

    finally:
        req_q.put({"type": "shutdown"})
        worker.join(timeout=3)
        if best_record:
            with BEST_PATH.open("w") as f:
                json.dump(best_record, f, indent=2)


if __name__ == "__main__":
    main()
