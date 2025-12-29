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
PID_BOUNDS = [(0.1, 100), (0.01, 50.0), (0.01, 50.0)]
SIMULATION_STEPS = 2500
DT = 1.0 / 240.0
MAX_ITERATIONS = 100
DISPLAY_REALTIME = True 
BASE_MUTATION = 0.5
PID_MAX_OVERSHOOT_PCT = 1
PID_MAX_RISE_TIME = 1
PID_MAX_SETTLING_TIME = 1

START_TIME = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
EXP_DIR = Path("logs") / f"DE_HIL_{START_TIME}"
LOG_PATH = EXP_DIR / "iteration_log.csv"
PKL_PATH = EXP_DIR / "iteration_log.pkl"
CONFIG_PATH = EXP_DIR / "config.yaml"

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
        "type": "DE_HIL",
        "pid_bounds": PID_BOUNDS,
        "max_iterations": MAX_ITERATIONS,
    }
    CONFIG_PATH.write_text(json.dumps(config_data, indent=2))
    if not LOG_PATH.exists():
        with LOG_PATH.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "iteration", "choice", "cand_kp", "cand_ki", "cand_kd",
                "fit", "overshoot", "rise_time", "settling_time", "target_ok",
                "de_mutation", "iter_seconds"
            ])
    if not PKL_PATH.exists():
        with PKL_PATH.open("wb") as f:
            pickle.dump([], f)

def log_iteration(iteration, choice_label, cand, fit, metrics, target_ok, de_mutation, iter_seconds=0.0):
    ts = datetime.utcnow().isoformat()
    row = [
        ts, iteration, choice_label,
        float(cand[0]), float(cand[1]), float(cand[2]),
        float(fit),
        float(metrics["overshoot"]), float(metrics["rise_time"]), float(metrics["settling_time"]),
        int(bool(target_ok)),
        float(de_mutation),
        float(iter_seconds)
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

def bullet_worker(req_q: mp.Queue, resp_q: mp.Queue):
    import pybullet as p
    import pybullet_data
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])

    def evaluate_pid(pid_params, label_text="", return_history=False, realtime=False):
        Kp, Ki, Kd = pid_params
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        start_pos = [0, 0, 0.02]
        robotId = p.loadURDF("husky/husky.urdf", start_pos, p.getQuaternionFromEuler([0, 0, 0]))

        for i in range(p.getNumJoints(robotId)):
            info = p.getJointInfo(robotId, i)
            if any(n in info[12].decode("utf-8") for n in ["imu", "plate", "rail", "bumper"]):
                p.changeDynamics(robotId, i, mass=0)

        left = [2, 4]
        right = [3, 5]
        for j in left + right:
            p.changeDynamics(robotId, j, lateralFriction=2.0)
            p.setJointMotorControl2(robotId, j, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

        for _ in range(100): p.stepSimulation() 

        if label_text:
            p.addUserDebugText(label_text, [0, 0, 1.0], textColorRGB=[0, 0, 0], textSize=2.0)

        target_rad = np.deg2rad(90.0)
        integral, prev_err, cost = 0.0, 0.0, 0.0
        hist = {"time": [], "actual": []}

        for i in range(SIMULATION_STEPS):
            p.stepSimulation()
            curr = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(robotId)[1])[2]
            err = target_rad - curr
            integral += err * DT
            deriv = (err - prev_err) / DT
            out = float(np.clip((Kp * err) + (Ki * integral) + (Kd * deriv), -40, 40))
            
            for j in left: p.setJointMotorControl2(robotId, j, p.VELOCITY_CONTROL, targetVelocity=-out, force=50)
            for j in right: p.setJointMotorControl2(robotId, j, p.VELOCITY_CONTROL, targetVelocity=out, force=50)
            
            cost += (err ** 2) + (0.001 * (out ** 2))
            prev_err = err
            if return_history:
                hist["time"].append(i * DT)
                hist["actual"].append(float(np.rad2deg(curr)))
            if realtime: time.sleep(DT)

        return cost / SIMULATION_STEPS, hist if return_history else None

    while True:
        msg = req_q.get()
        if msg["type"] == "shutdown": break
        if msg["type"] == "eval":
            f, h = evaluate_pid(msg["params"], msg.get("label", ""), msg.get("hist", False), msg.get("rt", False))
            resp_q.put({"id": msg["id"], "fit": float(f), "hist": h})
    p.disconnect()

def main():
    mp.set_start_method("spawn", force=True)
    req_q, resp_q = mp.Queue(), mp.Queue()
    worker = mp.Process(target=bullet_worker, args=(req_q, resp_q), daemon=False)
    worker.start()
    
    next_id = 0
    def run_sim(params, label="", hist=False, rt=False):
        nonlocal next_id
        next_id += 1
        req_q.put({"type": "eval", "id": next_id, "params": list(params), "label": label, "hist": hist, "rt": rt})
        while True:
            r = resp_q.get()
            if r["id"] == next_id: return r["fit"], r["hist"]

    init_logger()
    de = DifferentialEvolutionOptimizer(bounds=PID_BOUNDS, pop_size=8, mutation_factor=BASE_MUTATION)
    gui = SingleFeedbackGUI()
    prev_histories = []
    
    iteration = 0
    try:
        while iteration < MAX_ITERATIONS:
            iteration += 1
            iter_start = time.time()
            print(f"\n--- DE HIL Iteration {iteration} ---")
            
            # 1. Evolve population (Fitness wrapper calls PyBullet for entire generation)
            def fit_wrapper(p):
                f, _ = run_sim(p, rt=False) # Fast eval for population
                return f
            
            cand, fit = de.evolve(fit_wrapper)
            
            # 2. Visualize Best Candidate of this Generation
            f_vis, hist = run_sim(cand, label=f"DE Gen {iteration}", hist=True, rt=DISPLAY_REALTIME)
            metrics = calculate_metrics(hist, 90.0)
            target_ok = meets_pid_targets(metrics)
            
            if target_ok:
                print(">>> Auto-terminate: Targets met.")
                log_iteration(iteration, "auto_success", cand, fit, metrics, True, de.mutation_factor, time.time()-iter_start)
                break

            # 3. HIL Feedback
            choice = gui.show_candidate(hist, cand, fit, metrics, prev_histories, label_text=f"DE Gen {iteration} Best")
            
            if choice == 1: # ACCEPT -> REFINE
                print(f"User ACCEPTED. Refining search space around best candidate.")
                de.refine_search_space(cand)
                log_label = "accept_refine"
            elif choice == 2: # REJECT -> EXPAND
                print("User REJECTED. Expanding search space.")
                de.expand_search_space()
                log_label = "reject_expand"
            else:
                break
            
            log_iteration(iteration, log_label, cand, fit, metrics, False, de.mutation_factor, time.time()-iter_start)
            prev_histories.append(hist)
            append_histories_pickle([{"iteration": iteration, "params": cand, "fit": fit, "metrics": metrics, "history": hist}])
            
    finally:
        req_q.put({"type": "shutdown"})
        worker.join()

if __name__ == "__main__":
    main()