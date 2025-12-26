import time
import numpy as np
import pybullet as p
import pybullet_data
from differential_evolution import DifferentialEvolutionOptimizer
from bayesian_optimization import BayesianOptimizer
from visualizer import FeedbackGUI 

# --- CONFIGURATION ---
PID_BOUNDS = [(0.0, 10), (0.0, 5.0), (0.0, 10.0)] 
SIMULATION_STEPS = 2500 
DT = 1./240.

def calculate_metrics(history, target_val):
    """
    Calculates control metrics from time-series data.
    """
    time_arr = np.array(history['time'])
    actual_arr = np.array(history['actual'])
    
    # 1. Overshoot (%)
    max_val = np.max(actual_arr)
    overshoot = 0.0
    if max_val > target_val:
        overshoot = ((max_val - target_val) / target_val) * 100.0
    
    # 2. Rise Time (10% to 90%)
    try:
        t_10_idx = np.where(actual_arr >= 0.1 * target_val)[0][0]
        t_90_idx = np.where(actual_arr >= 0.9 * target_val)[0][0]
        rise_time = time_arr[t_90_idx] - time_arr[t_10_idx]
    except IndexError:
        rise_time = -1.0 # Failed to rise
        
    # 3. Settling Time (Within 5% band)
    # Time after which error never exceeds 5%
    tolerance = 0.05 * target_val
    upper_bound = target_val + tolerance
    lower_bound = target_val - tolerance
    
    # Find indices where signal is OUTSIDE bounds
    out_of_bounds = np.where((actual_arr > upper_bound) | (actual_arr < lower_bound))[0]
    
    if len(out_of_bounds) == 0:
        settling_time = 0.0 # Settled instantly (unlikely)
    elif out_of_bounds[-1] == len(actual_arr) - 1:
        settling_time = -1.0 # Never settled
    else:
        settling_time = time_arr[out_of_bounds[-1] + 1]

    return {
        "overshoot": overshoot,
        "rise_time": rise_time,
        "settling_time": settling_time
    }

class RobotEvaluator:
    def __init__(self, gui=True):
        self.gui = gui
        self.physicsClient = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0,0,0])

    def evaluate_pid(self, pid_params, label_text="", return_history=False):
        Kp, Ki, Kd = pid_params
        
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        
        start_pos = [0, 0, 0.1]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        robotId = p.loadURDF("husky/husky.urdf", start_pos, start_orn)
        
        # --- FIX: Suppress Warnings & Stability ---
        for i in range(p.getNumJoints(robotId)):
            info = p.getJointInfo(robotId, i)
            link_name = info[12].decode('utf-8')
            if any(n in link_name for n in ["imu", "plate", "rail", "bumper"]):
                p.changeDynamics(robotId, i, mass=0)

        left_wheels = [2, 4]
        right_wheels = [3, 5]
        for joint in left_wheels + right_wheels:
            p.changeDynamics(robotId, joint, lateralFriction=2.0)

        # --- FEATURE: Add Label Text ---
        if self.gui and label_text:
            # Draw text 1 meter above the robot
            p.addUserDebugText(label_text, [0, 0, 1.0], textColorRGB=[0, 0, 0], textSize=2.0)

        target_yaw_deg = 90.0
        target_yaw_rad = np.deg2rad(target_yaw_deg)
        
        integral_error = 0
        prev_error = 0
        total_cost = 0
        history = {'time': [], 'target': [], 'actual': []}

        for i in range(SIMULATION_STEPS):
            p.stepSimulation()
            
            _, current_orn = p.getBasePositionAndOrientation(robotId)
            current_euler = p.getEulerFromQuaternion(current_orn)
            current_yaw = current_euler[2] 
            
            error = target_yaw_rad - current_yaw
            integral_error += error * DT
            derivative_error = (error - prev_error) / DT
            
            turn_speed = (Kp * error) + (Ki * integral_error) + (Kd * derivative_error)
            turn_speed = np.clip(turn_speed, -40, 40)
            
            for j in left_wheels:
                p.setJointMotorControl2(robotId, j, p.VELOCITY_CONTROL, targetVelocity=-turn_speed, force=50)
            for j in right_wheels:
                p.setJointMotorControl2(robotId, j, p.VELOCITY_CONTROL, targetVelocity=turn_speed, force=50)

            total_cost += (error ** 2) + (0.001 * (turn_speed**2))
            prev_error = error
            
            if return_history:
                history['time'].append(i * DT)
                history['target'].append(target_yaw_deg)
                history['actual'].append(np.rad2deg(current_yaw))
            
            if self.gui:
                time.sleep(DT)

        fitness = total_cost / SIMULATION_STEPS
        
        if return_history:
            return fitness, history
        return fitness

    def close(self):
        p.disconnect()

def main():
    de = DifferentialEvolutionOptimizer(bounds=PID_BOUNDS, pop_size=6)
    bo = BayesianOptimizer(bounds=PID_BOUNDS)
    evaluator = RobotEvaluator(gui=True) 
    gui = FeedbackGUI() 

    print("Initializing Optimization System...")

    iteration = 0
    while True:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")

        # Wrapper for optimization (doesn't need history/labels)
        def fitness_wrapper(params):
            return evaluator.evaluate_pid(params, return_history=False)

        # 1. Generate Candidates
        cand_a, fit_a = de.evolve(fitness_wrapper)
        cand_b = bo.propose_location()
        fit_b = fitness_wrapper(cand_b) 
        
        bo.update(cand_b, fit_b)
        bo.update(cand_a, fit_a)

        # 2. Simulate & Visualize
        print("Simulating A (DE)...")
        _, hist_a = evaluator.evaluate_pid(cand_a, label_text="DE CANDIDATE", return_history=True)
        
        print("Simulating B (BO)...")
        _, hist_b = evaluator.evaluate_pid(cand_b, label_text="BO CANDIDATE", return_history=True)

        # 3. Calculate Metrics
        metrics_a = calculate_metrics(hist_a, 90.0)
        metrics_b = calculate_metrics(hist_b, 90.0)

        # 4. GUI Feedback
        choice = gui.show_comparison(hist_a, hist_b, cand_a, cand_b, fit_a, fit_b, metrics_a, metrics_b)

        if choice == 1: 
            print("User Preferred: DE")
        elif choice == 2: 
            print("User Preferred: BO")
            de.population[np.random.randint(0, de.pop_size)] = cand_b
        elif choice == 3: 
            print("TIE: Refine Mode")
            avg_c = (cand_a + cand_b) / 2
            de.refine_search_space(avg_c)
            bo.refine_bounds(avg_c)
        elif choice == 4:
            print("REJECT: Expand Mode")
            de.expand_search_space()
            bo.expand_bounds()
        else:
            break

        if iteration >= 15: break

    evaluator.close()

if __name__ == "__main__":
    main()