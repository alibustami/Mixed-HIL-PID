import csv
import json
import time
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import multiprocessing as mp

from differential_evolution import DifferentialEvolutionOptimizer
from bayesian_optimization import ConstrainedBayesianOptimizer
from visualizer_macos import FeedbackGUI


# --- CONFIGURATION ---
# Treat these as HARD global domain bounds for the paper
PID_BOUNDS = [(0.1, 10.0), (0.01, 10.0), (0.01, 10.0)]

SIMULATION_STEPS = 2500
DT = 1.0 / 240.0

BASE_MUTATION = 0.5
PREFERENCE_LR = 0.3
MAX_ITERATIONS = 100

TARGET_YAW_DEG = 90.0

PID_MAX_OVERSHOOT_PCT = 5
PID_MAX_RISE_TIME = 1
PID_MAX_SETTLING_TIME = 2

DISPLAY_REALTIME = False

# Actuator constraint (physical)
PID_OUTPUT_LIMIT = 255.0
PID_SAT_PENALTY = 0.01
PID_STRICT_OUTPUT_LIMIT = True
PID_SAT_HARD_PENALTY = 10000.0

BO_POF_MIN = 0.95

START_TIME = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
EXP_DIR = Path("logs/mixed") / f"mixed_{START_TIME}"
LOG_PATH = EXP_DIR / "iteration_log.csv"
PKL_PATH = EXP_DIR / "iteration_log.pkl"
CONFIG_PATH = EXP_DIR / "config.yaml"
BEST_PATH = EXP_DIR / "best_results.json"


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
        "pid_bounds_hard_global": PID_BOUNDS,
        "simulation_steps": SIMULATION_STEPS,
        "dt": DT,
        "base_mutation": BASE_MUTATION,
        "preference_lr": PREFERENCE_LR,
        "max_iterations": MAX_ITERATIONS,
        "target_yaw_deg": TARGET_YAW_DEG,
        "pid_max_overshoot_pct": PID_MAX_OVERSHOOT_PCT,
        "pid_max_rise_time": PID_MAX_RISE_TIME,
        "pid_max_settling_time": PID_MAX_SETTLING_TIME,
        "display_realtime": DISPLAY_REALTIME,
        "pid_output_limit": PID_OUTPUT_LIMIT,
        "pid_sat_penalty": PID_SAT_PENALTY,
        "pid_strict_output_limit": PID_STRICT_OUTPUT_LIMIT,
        "pid_sat_hard_penalty": PID_SAT_HARD_PENALTY,
        "bo_pof_min": BO_POF_MIN,
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
    cand_a, fit_a, viol_a, sat_a, metrics_a, target_ok_a,
    cand_b, fit_b, viol_b, sat_b, metrics_b, target_ok_b,
    de_mutation, de_best_fit, de_best_viol, de_pop_std,
    best_overall_fit, bo_spans, pref_weights,
    gap_note="",
    iter_seconds=0.0
):
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
    try:
        with PKL_PATH.open("rb") as f:
            existing = pickle.load(f)
    except FileNotFoundError:
        existing = []
    existing.extend(records)
    with PKL_PATH.open("wb") as f:
        pickle.dump(existing, f)


class PreferenceModel:
    def __init__(self, bounds, lr=PREFERENCE_LR):
        self.bounds = np.array(bounds, dtype=float)
        self.lr = float(lr)
        self.weights = np.random.rand(len(self.bounds))

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
        Kp, Ki, Kd = [float(x) for x in pid_params]

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

        target_yaw_deg = float(TARGET_YAW_DEG)
        target_yaw_rad = float(np.deg2rad(target_yaw_deg))

        integral_error = 0.0
        total_cost = 0.0

        history = {"time": [], "target": [], "actual": []} if return_history else None

        max_abs_raw_output = 0.0
        sat_steps = 0

        # Derivative-on-measurement (reduces derivative kick) :contentReference[oaicite:3]{index=3}
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

            abs_raw_output = abs(raw_output)
            if abs_raw_output > max_abs_raw_output:
                max_abs_raw_output = abs_raw_output

            turn_speed = float(np.clip(raw_output, -PID_OUTPUT_LIMIT, PID_OUTPUT_LIMIT))
            if abs_raw_output > PID_OUTPUT_LIMIT:
                sat_steps += 1

            # Anti-windup: don't integrate further when saturated in the error direction.
            if raw_output != turn_speed and np.sign(raw_output) == np.sign(error):
                integral_error -= error * DT

            for j in left_wheels:
                p.setJointMotorControl2(robotId, j, p.VELOCITY_CONTROL, targetVelocity=-turn_speed, force=50)
            for j in right_wheels:
                p.setJointMotorControl2(robotId, j, p.VELOCITY_CONTROL, targetVelocity=turn_speed, force=50)

            saturation_excess = max(0.0, abs_raw_output - PID_OUTPUT_LIMIT)
            total_cost += (error ** 2) + (0.001 * (turn_speed ** 2)) + (PID_SAT_PENALTY * (saturation_excess ** 2))

            prev_yaw = current_yaw

            if return_history:
                history["time"].append(i * DT)
                history["target"].append(target_yaw_deg)
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

        if msg["type"] == "shutdown":
            running = False
            break

        if msg["type"] == "eval":
            job_id = msg["id"]
            params = msg["params"]
            label = msg.get("label_text", "")
            return_history = msg.get("return_history", False)
            realtime = msg.get("realtime", False)

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
    mp.set_start_method("spawn", force=True)  # IMPORTANT on macOS

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

    def violation_from_sat(sat):
        return float(sat["max_abs_raw_output"] - PID_OUTPUT_LIMIT)  # <= 0 is feasible

    pref_model = PreferenceModel(PID_BOUNDS, lr=PREFERENCE_LR)
    anchor_seed = pref_model.anchor_params()

    de = DifferentialEvolutionOptimizer(bounds=PID_BOUNDS, pop_size=6, mutation_factor=BASE_MUTATION)
    # place anchor into population safely (scores remain inf until evaluated)
    de.population[0] = np.clip(anchor_seed, de.bounds[:, 0], de.bounds[:, 1])

    bo = ConstrainedBayesianOptimizer(bounds=PID_BOUNDS, pof_min=BO_POF_MIN)
    gui = FeedbackGUI()

    print(f"Initializing Optimization System... (mutation={de.mutation_factor:.2f}, anchor seed={anchor_seed})")
    init_logger()

    best_overall_fit = np.inf
    best_record = None

    # Warm-start BO with DE population
    for idx, cand in enumerate(de.population):
        fit, _, sat = eval_in_bullet(cand, return_history=False, realtime=False)
        viol = violation_from_sat(sat)
        bo.update(cand, fit, viol)
        best_overall_fit = min(best_overall_fit, fit)
        print(f"[Init BO] Seed {idx}: params={cand}, fit={fit:.4f}, viol={viol:.3f}")

    iteration = 0
    try:
        while True:
            iter_start = time.time()
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")

            # DE needs (fit, viol)
            def fitness_wrapper(params):
                fit, _, sat = eval_in_bullet(params, return_history=False, realtime=False)
                return fit, violation_from_sat(sat)

            # 1) Generate Candidates
            cand_a, fit_a_fast, viol_a_fast = de.evolve(fitness_wrapper)

            cand_b = bo.propose_location()
            fit_b_fast, _, sat_b_fast = eval_in_bullet(cand_b, return_history=False, realtime=False)
            viol_b_fast = violation_from_sat(sat_b_fast)

            # Update BO with both points (sharing information)
            bo.update(cand_b, fit_b_fast, viol_b_fast)
            bo.update(cand_a, fit_a_fast, viol_a_fast)

            # 2) Simulate & Visualize (history ON)
            print("Simulating A (DE)...")
            fit_a, hist_a, sat_a = eval_in_bullet(
                cand_a, label_text="DE CANDIDATE", return_history=True, realtime=DISPLAY_REALTIME
            )
            viol_a = violation_from_sat(sat_a)

            print("Simulating B (BO)...")
            fit_b, hist_b, sat_b = eval_in_bullet(
                cand_b, label_text="BO CANDIDATE", return_history=True, realtime=DISPLAY_REALTIME
            )
            viol_b = violation_from_sat(sat_b)

            # 3) Metrics
            metrics_a = calculate_metrics(hist_a, TARGET_YAW_DEG)
            metrics_b = calculate_metrics(hist_b, TARGET_YAW_DEG)
            target_ok_a = meets_pid_targets(metrics_a)
            target_ok_b = meets_pid_targets(metrics_b)

            safe_ok_a = (viol_a <= 0.0)
            safe_ok_b = (viol_b <= 0.0)

            best_overall_fit = min(best_overall_fit, fit_a, fit_b)

            # Update best record (prefer feasible first)
            def better(fit, viol, cur):
                if cur is None:
                    return True
                cur_fit = float(cur["fit"])
                cur_viol = float(cur["violation"])
                a_feas = (viol <= 0.0)
                b_feas = (cur_viol <= 0.0)
                if a_feas and not b_feas:
                    return True
                if b_feas and not a_feas:
                    return False
                if a_feas and b_feas:
                    return fit < cur_fit
                return viol < cur_viol

            for label, cand, fit, viol, metrics in [
                ("DE", cand_a, fit_a, viol_a, metrics_a),
                ("BO", cand_b, fit_b, viol_b, metrics_b),
            ]:
                if better(float(fit), float(viol), best_record):
                    best_record = {
                        "iteration": int(iteration),
                        "label": label,
                        "params": [float(x) for x in cand],
                        "fit": float(fit),
                        "violation": float(viol),
                        "metrics": metrics,
                    }

            # Optimizer stats
            de_best_fit, de_best_viol = de.best_scores()
            de_pop_std = float(np.mean(np.std(de.population, axis=0))) if de.population.size else 0.0
            bo_spans = bo.bounds[:, 1] - bo.bounds[:, 0]

            # 3.5) Auto-termination (performance + physical safety)
            terminate_reason = None
            terminate_label = None

            if target_ok_a and safe_ok_a:
                terminate_reason = "DE candidate met PID targets AND respected actuator limit"
                terminate_label = "auto_terminate_de"
            elif target_ok_b and safe_ok_b:
                terminate_reason = "BO candidate met PID targets AND respected actuator limit"
                terminate_label = "auto_terminate_bo"

            if terminate_reason:
                print(f"[Terminate] {terminate_reason}")

                log_iteration(
                    iteration=iteration,
                    choice_label=terminate_label,
                    cand_a=cand_a, fit_a=fit_a, viol_a=viol_a, sat_a=sat_a,
                    metrics_a=metrics_a, target_ok_a=target_ok_a,
                    cand_b=cand_b, fit_b=fit_b, viol_b=viol_b, sat_b=sat_b,
                    metrics_b=metrics_b, target_ok_b=target_ok_b,
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
                    {"iteration": iteration, "label": "DE", "params": cand_a, "fit": fit_a, "violation": viol_a, "metrics": metrics_a, "history": hist_a, "sat": sat_a},
                    {"iteration": iteration, "label": "BO", "params": cand_b, "fit": fit_b, "violation": viol_b, "metrics": metrics_b, "history": hist_b, "sat": sat_b},
                ])
                break

            # 4) GUI Feedback (blocks)
            choice = gui.show_comparison(
                hist_a, hist_b, cand_a, cand_b, fit_a, fit_b, metrics_a, metrics_b, [], []
            )

            gap_note = ""
            choice_label = {1: "prefer_de", 2: "prefer_bo", 3: "tie_refine", 4: "reject_expand"}.get(choice, "exit")

            if choice == 1:
                print("User Preferred: DE")
                gap = pref_model.update_towards(cand_a, cand_b)
                anchor = pref_model.anchor_params()

                de.mutation_factor = BASE_MUTATION
                de.inject_candidate(anchor, eval_func=fitness_wrapper, protect_best=True)

                bo.nudge_with_preference(cand_a, fit_a, fit_b, viol_a)
                gap_note = "B-A:" + ",".join(f"{g:.4f}" for g in gap)

            elif choice == 2:
                print("User Preferred: BO")
                gap = pref_model.update_towards(cand_b, cand_a)
                anchor = pref_model.anchor_params()

                de.mutation_factor = BASE_MUTATION
                de.inject_candidate(cand_b, eval_func=fitness_wrapper, protect_best=True)
                de.inject_candidate(anchor, eval_func=fitness_wrapper, protect_best=True)

                bo.nudge_with_preference(cand_b, fit_b, fit_a, viol_b)
                gap_note = "A-B:" + ",".join(f"{g:.4f}" for g in gap)

            elif choice == 3:
                print("TIE: Refine Mode")
                avg_c = (np.array(cand_a) + np.array(cand_b)) / 2.0
                de.refine_search_space(avg_c)
                bo.refine_bounds(avg_c)
                gap_note = "tie"

            elif choice == 4:
                print("REJECT: Expand Mode (within HARD global bounds)")
                de.expand_search_space()
                bo.expand_bounds()
                gap_note = "reject"

            else:
                break

            log_iteration(
                iteration=iteration,
                choice_label=choice_label,
                cand_a=cand_a, fit_a=fit_a, viol_a=viol_a, sat_a=sat_a,
                metrics_a=metrics_a, target_ok_a=target_ok_a,
                cand_b=cand_b, fit_b=fit_b, viol_b=viol_b, sat_b=sat_b,
                metrics_b=metrics_b, target_ok_b=target_ok_b,
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
                {"iteration": iteration, "label": "DE", "params": cand_a, "fit": fit_a, "violation": viol_a, "metrics": metrics_a, "history": hist_a, "sat": sat_a, "choice": choice_label},
                {"iteration": iteration, "label": "BO", "params": cand_b, "fit": fit_b, "violation": viol_b, "metrics": metrics_b, "history": hist_b, "sat": sat_b, "choice": choice_label},
            ])

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
