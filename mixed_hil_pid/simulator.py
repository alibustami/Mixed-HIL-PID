"""
PyBullet simulation module for PID controller evaluation.

This module provides a multiprocessing-based PyBullet simulation worker
for evaluating PID controllers on a Husky robot in a safe, isolated process.
"""

import time
import numpy as np
import multiprocessing as mp


def bullet_worker(req_q: mp.Queue, resp_q: mp.Queue, config, robot_config):
    """
    PyBullet simulation worker process.
    
    This function runs in a separate process and handles all PyBullet operations
    to avoid conflicts with Tkinter's event loop.
    
    Args:
        req_q: Queue for receiving evaluation requests
        resp_q: Queue for sending back results
        config: Dictionary with simulation configuration
        robot_config: Dictionary with robot-specific configuration
    """
    import pybullet as p
    import pybullet_data

    # Connect to PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.resetDebugVisualizerCamera(
        cameraDistance=2.0, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 0]
    )

    def evaluate_pid(pid_params, label_text="", return_history=False, realtime=False):
        """
        Evaluate a PID controller in PyBullet simulation.
        
        Args:
            pid_params: [Kp, Ki, Kd] parameters
            label_text: Optional text to display in simulation
            return_history: Whether to return time history
            realtime: Whether to run in real-time
            
        Returns:
            Tuple of (fitness, history, saturation_info)
        """
        Kp, Ki, Kd = [float(x) for x in pid_params]

        # Reset simulation
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")

        # Load robot using config
        start_pos = robot_config['start_pos']
        start_orn = p.getQuaternionFromEuler(robot_config['start_orn'])
        robotId = p.loadURDF(robot_config['urdf_path'], start_pos, start_orn)

        # Remove mass from decorative links
        if robot_config.get('decorative_links'):
            for i in range(p.getNumJoints(robotId)):
                info = p.getJointInfo(robotId, i)
                link_name = info[12].decode("utf-8")
                if any(n in link_name for n in robot_config['decorative_links']):
                    p.changeDynamics(robotId, i, mass=0)

        # Configure wheel friction
        left_wheels = robot_config['left_wheels']
        right_wheels = robot_config['right_wheels']
        wheel_friction = robot_config.get('wheel_friction', 2.0)
        control_force = robot_config.get('control_force', 50)
        
        for joint in left_wheels + right_wheels:
            p.changeDynamics(robotId, joint, lateralFriction=wheel_friction)

        # Settle robot
        settle_steps = 120
        for joint in left_wheels + right_wheels:
            p.setJointMotorControl2(robotId, joint, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        for _ in range(settle_steps):
            p.stepSimulation()

        # Add debug text
        if label_text:
            p.addUserDebugText(label_text, [0, 0, 1.0], textColorRGB=[0, 0, 0], textSize=2.0)

        # Simulation parameters from config
        target_yaw_deg = float(config["target_yaw_deg"])
        target_yaw_rad = float(np.deg2rad(target_yaw_deg))
        simulation_steps = config["simulation_steps"]
        dt = config["dt"]
        pid_output_limit = config["pid_output_limit"]
        pid_sat_penalty = config["pid_sat_penalty"]
        pid_strict_output_limit = config["pid_strict_output_limit"]
        pid_sat_hard_penalty = config["pid_sat_hard_penalty"]

        # PID control loop
        integral_error = 0.0
        total_cost = 0.0
        history = {"time": [], "target": [], "actual": []} if return_history else None
        max_abs_raw_output = 0.0
        sat_steps = 0
        prev_yaw = None
        
        # Get control mode
        control_mode = robot_config.get('control_mode', 'differential')

        for i in range(simulation_steps):
            p.stepSimulation()

            # Get current yaw
            _, current_orn = p.getBasePositionAndOrientation(robotId)
            current_yaw = float(p.getEulerFromQuaternion(current_orn)[2])

            if prev_yaw is None:
                prev_yaw = current_yaw

            # PID calculation (derivative-on-measurement to reduce kick)
            error = target_yaw_rad - current_yaw
            integral_error += error * dt
            dyaw = (current_yaw - prev_yaw) / dt
            raw_output = (Kp * error) + (Ki * integral_error) - (Kd * dyaw)

            # Track saturation
            abs_raw_output = abs(raw_output)
            if abs_raw_output > max_abs_raw_output:
                max_abs_raw_output = abs_raw_output

            # Apply limits
            if control_mode == 'ackermann':
                # For Ackermann: output is steering angle in radians
                # Use configured limit (user set this to be appropriate for their robot)
                steering_angle = float(np.clip(raw_output, -pid_output_limit, pid_output_limit))
                
                # Check saturation
                if abs_raw_output > pid_output_limit:
                    sat_steps += 1
                
                # Anti-windup: stop integrating when saturated
                if abs(raw_output) > pid_output_limit and np.sign(raw_output) == np.sign(error):
                    integral_error -= error * dt
                
                # Apply steering to front wheels
                steering_joints = robot_config.get('steering_joints', [])
                for j in steering_joints:
                    p.setJointMotorControl2(robotId, j, p.POSITION_CONTROL, 
                                          targetPosition=steering_angle, 
                                          force=control_force)
                
                # Apply constant forward speed to rear wheels
                drive_joints = robot_config.get('drive_joints', [])
                constant_speed = robot_config.get('constant_speed', 5.0)
                for j in drive_joints:
                    p.setJointMotorControl2(robotId, j, p.VELOCITY_CONTROL, 
                                          targetVelocity=constant_speed, 
                                          force=control_force)
                
                # Calculate cost for Ackermann (penalize steering angle)
                saturation_excess = max(0.0, abs_raw_output - pid_output_limit)
                total_cost += (error ** 2) + (0.001 * (steering_angle ** 2)) + (pid_sat_penalty * (saturation_excess ** 2))
                
            else:  # differential drive mode
                turn_speed = float(np.clip(raw_output, -pid_output_limit, pid_output_limit))
                if abs_raw_output > pid_output_limit:
                    sat_steps += 1

                # Anti-windup: stop integrating when saturated
                if raw_output != turn_speed and np.sign(raw_output) == np.sign(error):
                    integral_error -= error * dt

                # Apply control to wheels
                for j in left_wheels:
                    p.setJointMotorControl2(robotId, j, p.VELOCITY_CONTROL, targetVelocity=-turn_speed, force=control_force)
                for j in right_wheels:
                    p.setJointMotorControl2(robotId, j, p.VELOCITY_CONTROL, targetVelocity=turn_speed, force=control_force)

                # Calculate cost for differential drive
                saturation_excess = max(0.0, abs_raw_output - pid_output_limit)
                total_cost += (error ** 2) + (0.001 * (turn_speed ** 2)) + (pid_sat_penalty * (saturation_excess ** 2))

            prev_yaw = current_yaw

            # Record history
            if return_history:
                history["time"].append(i * dt)
                history["target"].append(target_yaw_deg)
                history["actual"].append(float(np.rad2deg(current_yaw)))

            # Real-time delay
            if realtime:
                time.sleep(dt)

        # Apply hard penalty for exceeding limits
        if pid_strict_output_limit:
            if max_abs_raw_output > pid_output_limit:
                excess = max_abs_raw_output - pid_output_limit
                total_cost += pid_sat_hard_penalty * (1.0 + (excess / pid_output_limit))

        # Calculate fitness and saturation info
        fitness = float(total_cost / simulation_steps)
        sat_info = {
            "max_abs_raw_output": float(max_abs_raw_output),
            "sat_fraction": float(sat_steps / simulation_steps),
        }

        return fitness, history, sat_info

    # Main worker loop
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



class SimulationManager:
    """
    Manager class for PyBullet simulation worker process.
    
    Handles process lifecycle and provides a clean API for evaluating
    PID controllers in the simulation.
    """
    
    def __init__(self, config, robot_config):
        """
        Initialize simulation manager.
        
        Args:
            config: Dictionary with simulation configuration
            robot_config: Dictionary with robot-specific configuration
        """
        self.config = config
        self.robot_config = robot_config
        # Use spawn context for cross-platform compatibility
        self.ctx = mp.get_context('spawn')
        self.req_q = self.ctx.Queue()
        self.resp_q = self.ctx.Queue()
        self.worker = None
        self.next_id = 0
        
    def start(self):
        """Start the simulation worker process."""
        self.worker = self.ctx.Process(
            target=bullet_worker, 
            args=(self.req_q, self.resp_q, self.config, self.robot_config),
            daemon=False
        )
        self.worker.start()
        
    def evaluate(self, params, label_text="", return_history=False, realtime=False):
        """
        Evaluate PID parameters in simulation.
        
        Args:
            params: [Kp, Ki, Kd] parameters
            label_text: Optional text to display
            return_history: Whether to return time history
            realtime: Whether to run in real-time
            
        Returns:
            Tuple of (fitness, history, saturation_info)
        """
        self.next_id += 1
        job_id = self.next_id
        
        self.req_q.put({
            "type": "eval",
            "id": job_id,
            "params": [float(x) for x in params],
            "label_text": label_text,
            "return_history": bool(return_history),
            "realtime": bool(realtime),
        })
        
        while True:
            resp = self.resp_q.get()
            if resp["id"] == job_id:
                return float(resp["fitness"]), resp["history"], resp["sat"]
                
    def shutdown(self):
        """Shutdown the simulation worker process."""
        self.req_q.put({"type": "shutdown"})
        if self.worker:
            self.worker.join(timeout=3)
