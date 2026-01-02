import csv
import json
import time
import pickle
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
import numpy as np

from differential_evolution import DifferentialEvolutionOptimizer
from visualizer_single import SingleFeedbackGUI


# --- CONFIGURATION ---
PID_BOUNDS = [(0.1, 10.0), (0.01, 10.0), (0.01, 10.0)]
SIMULATION_STEPS = 2500
DT = 1.0 / 240.0
MAX_ITERATIONS = 100
DISPLAY_REALTIME = False
BASE_MUTATION = 0.5

TARGET_YAW_DEG = 90.0
PID_MAX_OVERSHOOT_PCT = 5.0
PID_MAX_RISE_TIME = 1.0
PID_MAX_SETTLING_TIME = 2.0

# Actuator constraint model (PWM-like limit)
PID_OUTPUT_LIMIT = 255.0
PID_SAT_PENALTY = 0.01
PID_STRICT_OUTPUT_LIMIT = True
PID_SAT_HARD_PENALTY = 10000.0

START_TIME = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
EXP_DIR = Path("logs/DE_HIL") / f"DE_HIL_{START_TIME}"
LOG_PATH = EXP_DIR / "iteration_log.csv"
PKL_PATH = EXP_DIR / "iteration_log.pkl"
CONFIG_PATH = EXP_DIR / "config.yaml"
BEST_PATH = EXP_DIR / "best_results.json"

GLOBAL_BOUNDS = np.array(PID_BOUNDS, dtype=float)


# ------------------ Small math helpers ------------------

def wrap_pi(x: float) -> float:
    """Wrap angle (radians) into [-pi, pi)."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


# ------------------ Metrics / termination ------------------

def calculate_metrics(history, target_val):
    time_arr = np.array(history["time"], dtype=float)
    actual_arr = np.array(history["actual"], dtype=float)

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

    return {
        "overshoot": float(overshoot),
        "rise_time": float(rise_time),
        "settling_time": float(settling_time),
    }


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

def init_logger():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    config_data = {
        "start_time_utc": START_TIME,
        "type": "DE_HIL",
        "pid_bounds": PID_BOUNDS,
        "simulation_steps": SIMULATION_STEPS,
        "dt": DT,
        "max_iterations": MAX_ITERATIONS,
        "display_realtime": DISPLAY_REALTIME,
        "base_mutation": BASE_MUTATION,
        "target_yaw_deg": TARGET_YAW_DEG,
        "pid_max_overshoot_pct": PID_MAX_OVERSHOOT_PCT,
        "pid_max_rise_time": PID_MAX_RISE_TIME,
        "pid_max_settling_time": PID_MAX_SETTLING_TIME,
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
        cameraDistance=2.0,
        cameraYaw=0,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0],
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

        # Wrap-safe measurement handling:
        prev_yaw_wrapped = None
        yaw_unwrapped = 0.0

        for i in range(SIMULATION_STEPS):
            p.stepSimulation()

            _, orn = p.getBasePositionAndOrientation(robotId)
            yaw_wrapped = float(p.getEulerFromQuaternion(orn)[2])  # in [-pi, pi]

            if prev_yaw_wrapped is None:
                prev_yaw_wrapped = yaw_wrapped
                yaw_unwrapped = yaw_wrapped

            # unwrap using smallest delta
            dyaw_wrapped = wrap_pi(yaw_wrapped - prev_yaw_wrapped)
            yaw_unwrapped += dyaw_wrapped

            # shortest-path heading error (wrap-safe)
            error = wrap_pi(target_yaw_rad - yaw_wrapped)

            # integral of wrapped error
            integral_error += error * DT

            # derivative-on-measurement (wrap-safe)
            dyaw = dyaw_wrapped / DT
            raw_output = (Kp * error) + (Ki * integral_error) - (Kd * dyaw)

            abs_raw = abs(raw_output)
            if abs_raw > max_abs_raw_output:
                max_abs_raw_output = abs_raw

            # Saturated actuator model
            u = float(np.clip(raw_output, -PID_OUTPUT_LIMIT, PID_OUTPUT_LIMIT))
            if abs_raw > PID_OUTPUT_LIMIT:
                sat_steps += 1

            # Anti-windup: don't integrate further when saturated in error direction
            if raw_output != u and np.sign(raw_output) == np.sign(error):
                integral_error -= error * DT

            for j in left_wheels:
                p.setJointMotorControl2(robotId, j, p.VELOCITY_CONTROL, targetVelocity=-u, force=50)
            for j in right_wheels:
                p.setJointMotorControl2(robotId, j, p.VELOCITY_CONTROL, targetVelocity=u, force=50)

            saturation_excess = max(0.0, abs_raw - PID_OUTPUT_LIMIT)
            total_cost += (error ** 2) + (0.001 * (u ** 2)) + (PID_SAT_PENALTY * (saturation_excess ** 2))

            prev_yaw_wrapped = yaw_wrapped

            if return_history:
                history["time"].append(i * DT)
                history["target"].append(float(TARGET_YAW_DEG))
                history["actual"].append(float(np.rad2deg(yaw_unwrapped)))

            if realtime:
                time.sleep(DT)

        # Optional hard penalty inside objective (separate from explicit violation we log/optimize)
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
                params,
                label_text=label,
                return_history=return_history,
                realtime=realtime,
            )

            resp_q.put({
                "id": job_id,
                "fitness": float(fitness),
                "history": history,
                "sat": sat,
            })

    p.disconnect()


# ------------------ Main ------------------

def main():
    mp.set_start_method("spawn", force=True)

    req_q, resp_q = mp.Queue(), mp.Queue()
    worker = mp.Process(target=bullet_worker, args=(req_q, resp_q), daemon=False)
    worker.start()

    next_id = 0

    def run_sim(params, label_text="", return_history=False, realtime=False):
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

    def clamp_de_bounds_to_global(de_obj):
        """
        If your DE expands/refines bounds, clamp them back into the declared PID_BOUNDS.
        This prevents the 'DE escapes bounds after reject' effect when PID_BOUNDS are meant
        to be hard limits for the experiment.
        """
        if not hasattr(de_obj, "bounds"):
            return

        b = np.array(de_obj.bounds, dtype=float)
        b[:, 0] = np.maximum(b[:, 0], GLOBAL_BOUNDS[:, 0])
        b[:, 1] = np.minimum(b[:, 1], GLOBAL_BOUNDS[:, 1])
        b[:, 1] = np.maximum(b[:, 1], b[:, 0] + 1e-9)
        de_obj.bounds = b

        if hasattr(de_obj, "population") and de_obj.population is not None:
            de_obj.population = np.clip(de_obj.population, de_obj.bounds[:, 0], de_obj.bounds[:, 1])

    init_logger()

    de = DifferentialEvolutionOptimizer(bounds=PID_BOUNDS, pop_size=8, mutation_factor=BASE_MUTATION)
    gui = SingleFeedbackGUI()
    prev_histories = []

    best_overall_fit = float("inf")
    best_record = None

    def eval_obj_and_violation(p):
        fit, _, sat = run_sim(p, return_history=False, realtime=False)
        viol = float(sat["max_abs_raw_output"] - PID_OUTPUT_LIMIT)  # <=0 feasible
        return float(fit), float(viol)

    # Optional: keep a conservative seed in the initial population (no scoring here)
    safe_seed = np.array([b[0] for b in PID_BOUNDS], dtype=float)
    if hasattr(de, "population") and de.population is not None and de.population.shape[0] > 0:
        de.population[0] = np.clip(safe_seed, de.bounds[:, 0], de.bounds[:, 1])

    iteration = 0
    try:
        while iteration < MAX_ITERATIONS:
            iteration += 1
            iter_start = time.time()
            print(f"\n--- DE HIL Iteration {iteration} ---")

            # 1) One DE generation (constraint-aware if your DE uses violations)
            evolve_out = de.evolve(eval_obj_and_violation)
            if isinstance(evolve_out, (tuple, list)) and len(evolve_out) >= 3:
                cand, fit_fast, viol_fast = evolve_out[0], evolve_out[1], evolve_out[2]
            else:
                # Backward-compat fallback
                cand, fit_fast = evolve_out
                viol_fast = float("inf")

            # 2) Evaluate chosen candidate with history
            fit_eval, hist, sat = run_sim(
                cand,
                label_text=f"DE Gen {iteration}",
                return_history=True,
                realtime=DISPLAY_REALTIME,
            )

            violation_eval = float(sat["max_abs_raw_output"] - PID_OUTPUT_LIMIT)
            safe_ok = (violation_eval <= 0.0)

            metrics = calculate_metrics(hist, TARGET_YAW_DEG)
            target_ok = meets_pid_targets(metrics)

            best_overall_fit = min(best_overall_fit, float(fit_eval))

            # Update best record (prefer feasible first)
            def better(fit, viol, cur):
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

            if better(float(fit_eval), float(violation_eval), best_record):
                best_record = {
                    "iteration": int(iteration),
                    "label": "DE",
                    "params": [float(x) for x in cand],
                    "fit": float(fit_eval),
                    "violation": float(violation_eval),
                    "sat": sat,
                    "metrics": metrics,
                }

            # DE stats
            if hasattr(de, "best_scores"):
                de_best_fit, de_best_viol = de.best_scores()
            else:
                de_best_fit, de_best_viol = float("inf"), float("inf")

            de_pop_std = float(np.mean(np.std(de.population, axis=0))) if getattr(de, "population", None) is not None else 0.0

            # 2.5) Auto-terminate (targets + safe)
            if target_ok and safe_ok:
                print(">>> Auto-terminate: Targets met AND actuator limit respected.")
                log_iteration(
                    iteration=iteration,
                    choice_label="auto_terminate_target_and_limit_met",
                    cand=cand,
                    fit=fit_eval,
                    violation=violation_eval,
                    sat_info=sat,
                    metrics=metrics,
                    target_ok=True,
                    safe_ok=True,
                    de_mutation=float(getattr(de, "mutation_factor", 0.0)),
                    de_best_fit=float(de_best_fit),
                    de_best_viol=float(de_best_viol),
                    de_pop_std=de_pop_std,
                    best_overall_fit=best_overall_fit,
                    iter_seconds=time.time() - iter_start,
                )
                append_histories_pickle([{
                    "iteration": int(iteration),
                    "label": "DE",
                    "params": [float(x) for x in cand],
                    "fit": float(fit_eval),
                    "violation": float(violation_eval),
                    "sat": sat,
                    "metrics": metrics,
                    "history": hist,
                    "choice": "auto_terminate_target_and_limit_met",
                }])
                break

            # 3) Human feedback (binary single-candidate)
            choice = gui.show_candidate(
                hist,
                cand,
                float(fit_eval),
                metrics,
                prev_histories,
                label_text=f"DE Gen {iteration} Best",
            )

            if choice == 1:  # ACCEPT -> REFINE
                print("User ACCEPTED. Refining search space around best candidate.")
                de.refine_search_space(cand)
                clamp_de_bounds_to_global(de)
                # keep a conservative seed in-pop (optional)
                if getattr(de, "population", None) is not None and de.population.shape[0] >= 2:
                    de.population[1] = np.clip(safe_seed, de.bounds[:, 0], de.bounds[:, 1])
                log_label = "accept_refine"

            elif choice == 2:  # REJECT -> EXPAND
                print("User REJECTED. Expanding search space.")
                de.expand_search_space()
                clamp_de_bounds_to_global(de)
                if getattr(de, "population", None) is not None and de.population.shape[0] >= 2:
                    de.population[1] = np.clip(safe_seed, de.bounds[:, 0], de.bounds[:, 1])
                log_label = "reject_expand"

            else:
                print("User EXIT.")
                break

            log_iteration(
                iteration=iteration,
                choice_label=log_label,
                cand=cand,
                fit=fit_eval,
                violation=violation_eval,
                sat_info=sat,
                metrics=metrics,
                target_ok=target_ok,
                safe_ok=safe_ok,
                de_mutation=float(getattr(de, "mutation_factor", 0.0)),
                de_best_fit=float(de_best_fit),
                de_best_viol=float(de_best_viol),
                de_pop_std=de_pop_std,
                best_overall_fit=best_overall_fit,
                iter_seconds=time.time() - iter_start,
            )

            prev_histories.append(hist)
            append_histories_pickle([{
                "iteration": int(iteration),
                "label": "DE",
                "params": [float(x) for x in cand],
                "fit": float(fit_eval),
                "violation": float(violation_eval),
                "sat": sat,
                "metrics": metrics,
                "history": hist,
                "choice": log_label,
            }])

        # Save best record at end
        if best_record is not None:
            with BEST_PATH.open("w") as f:
                json.dump(best_record, f, indent=2)

    finally:
        try:
            req_q.put({"type": "shutdown"})
        except Exception:
            pass
        worker.join(timeout=3)


if __name__ == "__main__":
    main()
