"""
PyBullet simulation module for PID controller evaluation.

This module provides a multiprocessing-based PyBullet simulation worker
for evaluating PID controllers on a Husky robot in a safe, isolated process.
"""

import multiprocessing as mp
import time

import numpy as np


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
        cameraDistance=2.0,
        cameraYaw=0,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0],
    )

    def evaluate_pid(
        pid_params, label_text="", return_history=False, realtime=False, return_full_trace=False
    ):
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
        start_pos = robot_config["start_pos"]
        start_orn = p.getQuaternionFromEuler(robot_config["start_orn"])
        robotId = p.loadURDF(robot_config["urdf_path"], start_pos, start_orn)

        # Get control mode early
        control_mode = robot_config.get("control_mode", "differential")

        # ========== ACKERMANN JOINT DIAGNOSTICS & SIGN CORRECTION ==========
        if control_mode == "ackermann":
            print(f"\n{'='*70}")
            print(f"ACKERMANN URDF DIAGNOSTICS: {robot_config['urdf_path']}")
            print(f"{'='*70}")

            # Helper function for detailed joint dump
            def dump_joint(robotId, j):
                info = p.getJointInfo(robotId, j)
                name = info[1].decode("utf-8")
                link = info[12].decode("utf-8")
                jtype = info[2]
                axis = info[13]  # joint axis (x, y, z)
                lo, hi = info[8], info[9]  # limits
                type_names = {0: "REVOLUTE", 1: "PRISMATIC", 4: "FIXED"}
                return (
                    name,
                    link,
                    type_names.get(jtype, f"TYPE_{jtype}"),
                    axis,
                    (lo, hi),
                )

            # Dump critical joints
            print("\nCRITICAL JOINTS:")
            for j in [2, 3, 4, 5, 6, 7]:
                name, link, jtype, axis, limits = dump_joint(robotId, j)
                print(f"  Joint {j}: {name:30s} -> {link:30s}")
                print(
                    f"           Type: {jtype:10s}  Axis: {axis}  Limits: {limits}"
                )

            # Determine sign corrections from joint axes
            # Rear wheels (driven): 2=left, 3=right
            _, _, _, axis_rear_left, _ = dump_joint(robotId, 2)
            _, _, _, axis_rear_right, _ = dump_joint(robotId, 3)

            # Front wheels (spin): 5=left, 7=right
            _, _, _, axis_front_left, _ = dump_joint(robotId, 5)
            _, _, _, axis_front_right, _ = dump_joint(robotId, 7)

            # Steering hinges: 4=left, 6=right
            _, _, _, axis_steer_left, _ = dump_joint(robotId, 4)
            _, _, _, axis_steer_right, _ = dump_joint(robotId, 6)

            # Check if axes are mirrored (dot product negative means opposite direction)
            steer_dot = np.dot(axis_steer_left, axis_steer_right)
            rear_wheel_dot = np.dot(axis_rear_left, axis_rear_right)
            front_wheel_dot = np.dot(axis_front_left, axis_front_right)

            print(f"\n{'='*70}")
            print("AXIS ALIGNMENT CHECK:")
            print(
                f"  Steering hinges [4,6]:     dot_product = {steer_dot:.3f}"
            )
            print(
                f"  Rear wheels [2,3]:         dot_product = {rear_wheel_dot:.3f}"
            )
            print(
                f"  Front wheels [5,7]:        dot_product = {front_wheel_dot:.3f}"
            )
            print(f"  (dot < 0 means MIRRORED axes → need sign correction)")

            # Determine sign correction factors
            # For steering: if mirrored, right gets opposite sign
            steer_left_sign = 1.0
            steer_right_sign = -1.0 if steer_dot < 0 else 1.0

            # For wheels: if mirrored, right gets opposite sign
            rear_left_sign = 1.0
            rear_right_sign = -1.0 if rear_wheel_dot < 0 else 1.0

            front_left_sign = 1.0
            front_right_sign = -1.0 if front_wheel_dot < 0 else 1.0

            print(f"\nSIGN CORRECTIONS APPLIED:")
            print(
                f"  Steering: left={steer_left_sign:+.0f}, right={steer_right_sign:+.0f}"
            )
            print(
                f"  Rear wheels: left={rear_left_sign:+.0f}, right={rear_right_sign:+.0f}"
            )
            print(
                f"  Front wheels: left={front_left_sign:+.0f}, right={front_right_sign:+.0f}"
            )
            print(f"{'='*70}\n")

            # Store sign corrections for runtime use
            ackermann_signs = {
                "steer_left": steer_left_sign,
                "steer_right": steer_right_sign,
                "rear_left": rear_left_sign,
                "rear_right": rear_right_sign,
                "front_left": front_left_sign,
                "front_right": front_right_sign,
            }

        # Remove mass from decorative links
        if robot_config.get("decorative_links"):
            for i in range(p.getNumJoints(robotId)):
                info = p.getJointInfo(robotId, i)
                link_name = info[12].decode("utf-8")
                if any(
                    n in link_name for n in robot_config["decorative_links"]
                ):
                    p.changeDynamics(robotId, i, mass=0)

        # CRITICAL: Disable ALL motors at initialization to prevent hidden brakes
        for j in range(p.getNumJoints(robotId)):
            p.setJointMotorControl2(
                robotId, j, p.VELOCITY_CONTROL, targetVelocity=0, force=0
            )

        # Configure wheel friction - ONLY on actual wheel joints, NOT steering hinges
        wheel_friction = robot_config.get("wheel_friction", 2.0)
        control_force = robot_config.get("control_force", 50)

        if control_mode == "ackermann":
            # Apply friction to ALL wheel joints [2,3,5,7], not steering [4,6]
            wheel_joints = [2, 3, 5, 7]
            for joint in wheel_joints:
                p.changeDynamics(
                    robotId, joint, lateralFriction=wheel_friction
                )
            print(
                f"Applied friction {wheel_friction} to wheel joints: {wheel_joints}"
            )
        else:
            # Differential drive - define wheel lists for use throughout control loop
            left_wheels = robot_config["left_wheels"]
            right_wheels = robot_config["right_wheels"]
            for joint in left_wheels + right_wheels:
                p.changeDynamics(
                    robotId, joint, lateralFriction=wheel_friction
                )

        # Settle robot (no motors active, just physics settling)
        settle_steps = 120
        for _ in range(settle_steps):
            p.stepSimulation()

        # Add debug text
        if label_text:
            p.addUserDebugText(
                label_text, [0, 0, 1.0], textColorRGB=[0, 0, 0], textSize=2.0
            )

        # Simulation parameters from config
        target_yaw_deg = float(config["target_yaw_deg"])
        target_yaw_rad = float(np.deg2rad(target_yaw_deg))
        simulation_steps = config["simulation_steps"]
        dt = config["dt"]

        # Robot-specific constraint parameters
        pid_output_limit = robot_config["pid_output_limit"]
        pid_sat_penalty = robot_config.get("pid_sat_penalty", 0.01)
        pid_strict_output_limit = robot_config.get(
            "pid_strict_output_limit", True
        )
        pid_sat_hard_penalty = robot_config.get(
            "pid_sat_hard_penalty", 10000.0
        )

        # PID control loop
        integral_error = 0.0
        total_cost = 0.0
        history = (
            {"time": [], "target": [], "actual": []}
            if return_history
            else None
        )
        trace_data = [] if return_full_trace else None
        max_abs_raw_output = 0.0
        sat_steps = 0
        prev_yaw = None
        prev_steering_deg = None

        # Get control mode
        control_mode = robot_config.get("control_mode", "differential")

        # Ackermann-specific parameters
        if control_mode == "ackermann":
            pass  # No special parameters needed for simplified controller

        for i in range(simulation_steps):
            p.stepSimulation()

            # Get current yaw
            _, current_orn = p.getBasePositionAndOrientation(robotId)
            current_yaw = float(p.getEulerFromQuaternion(current_orn)[2])

            if prev_yaw is None:
                prev_yaw = current_yaw

            # Apply limits
            if control_mode == "ackermann":
                # ========== SIMPLIFIED ACKERMANN CONTROLLER (like Husky) ==========
                
                # 1. Calculate error in DEGREES
                current_yaw_deg = float(np.rad2deg(current_yaw))
                error_deg = target_yaw_deg - current_yaw_deg
                
                # 2. Wrap error to [-180, 180] degrees
                def wrap_to_180(angle_deg):
                    return ((angle_deg + 180.0) % 360.0) - 180.0
                
                error_deg = wrap_to_180(error_deg)
                
                # 3. PID controller (all in degrees)
                integral_error += error_deg * dt
                
                # Derivative on measurement (to avoid derivative kick)
                if prev_yaw is not None:
                    prev_yaw_deg = float(np.rad2deg(prev_yaw))
                    dyaw_deg = (current_yaw_deg - prev_yaw_deg) / dt
                else:
                    dyaw_deg = 0.0
                
                # PID calculation
                raw_output_deg = (
                    (Kp * error_deg) + (Ki * integral_error) - (Kd * dyaw_deg)
                )
                
                # 4. Track saturation
                abs_raw_output = abs(raw_output_deg)
                if abs_raw_output > max_abs_raw_output:
                    max_abs_raw_output = abs_raw_output
                
                # 5. Clamp to steering limits (in degrees)
                steering_deg = float(
                    np.clip(raw_output_deg, -pid_output_limit, pid_output_limit)
                )
                
                if abs_raw_output > pid_output_limit:
                    sat_steps += 1
                
                # Anti-windup: stop integrating when saturated
                if (raw_output_deg != steering_deg) and (
                    np.sign(raw_output_deg) == np.sign(error_deg)
                ):
                    integral_error -= error_deg * dt
                
                # 6. Convert to radians for PyBullet
                steering_angle = float(np.deg2rad(steering_deg))
                
                # Track for saturation info
                steering_excess = max(0.0, abs_raw_output - pid_output_limit)

                # 8. Apply steering to BOTH front steering hinges with sign correction
                steering_joints = robot_config.get("steering_joints", [4, 6])

                # Apply to left steering hinge (joint 4)
                p.setJointMotorControl2(
                    robotId,
                    steering_joints[0],
                    p.POSITION_CONTROL,
                    targetPosition=steering_angle
                    * ackermann_signs["steer_left"],
                    force=control_force,
                )

                # Apply to right steering hinge (joint 6) with sign correction
                p.setJointMotorControl2(
                    robotId,
                    steering_joints[1],
                    p.POSITION_CONTROL,
                    targetPosition=steering_angle
                    * ackermann_signs["steer_right"],
                    force=control_force,
                )

                # 9. Apply wheel velocities with sign correction
                # Use 4WD for debugging (all wheels driven)
                use_4wd = robot_config.get("use_4wd", True)
                constant_speed = robot_config.get("constant_speed", 5.0)

                if use_4wd:
                    # Drive all 4 wheels: [2=rear_left, 3=rear_right, 5=front_left, 7=front_right]
                    # Rear left (joint 2)
                    p.setJointMotorControl2(
                        robotId,
                        2,
                        p.VELOCITY_CONTROL,
                        targetVelocity=constant_speed
                        * ackermann_signs["rear_left"],
                        force=control_force,
                    )
                    # Rear right (joint 3)
                    p.setJointMotorControl2(
                        robotId,
                        3,
                        p.VELOCITY_CONTROL,
                        targetVelocity=constant_speed
                        * ackermann_signs["rear_right"],
                        force=control_force,
                    )
                    # Front left (joint 5)
                    p.setJointMotorControl2(
                        robotId,
                        5,
                        p.VELOCITY_CONTROL,
                        targetVelocity=constant_speed
                        * ackermann_signs["front_left"],
                        force=control_force,
                    )
                    # Front right (joint 7)
                    p.setJointMotorControl2(
                        robotId,
                        7,
                        p.VELOCITY_CONTROL,
                        targetVelocity=constant_speed
                        * ackermann_signs["front_right"],
                        force=control_force,
                    )
                else:
                    # RWD only: rear wheels [2, 3]
                    p.setJointMotorControl2(
                        robotId,
                        2,
                        p.VELOCITY_CONTROL,
                        targetVelocity=constant_speed
                        * ackermann_signs["rear_left"],
                        force=control_force,
                    )
                    p.setJointMotorControl2(
                        robotId,
                        3,
                        p.VELOCITY_CONTROL,
                        targetVelocity=constant_speed
                        * ackermann_signs["rear_right"],
                        force=control_force,
                    )
                    # Front wheels free-rolling (already disabled in init)

                # 10. Calculate cost (matching Husky's approach)
                saturation_excess = max(0.0, abs_raw_output - pid_output_limit)
                
                # Steering rate cost (smoothing)
                steering_rate_cost = 0.0
                if prev_steering_deg is not None:
                    steering_rate_cost = abs(steering_deg - prev_steering_deg)
                
                steering_rate_weight = robot_config.get("steering_rate_cost_weight", 0.0)
                
                total_cost += (
                    (error_deg**2)                              # Tracking error
                    + (0.001 * (steering_deg**2))               # Control effort
                    + (pid_sat_penalty * (saturation_excess**2)) # Saturation penalty
                    + (steering_rate_weight * steering_rate_cost) # Smoothing penalty
                )
                
                prev_steering_deg = steering_deg

                # TRACE LOGGING (Ackermann)
                if return_full_trace:
                    trace_record = {
                        "step": i,
                        "time": i * dt,
                        "Kp": Kp, "Ki": Ki, "Kd": Kd,
                        "target_yaw_deg": target_yaw_deg,
                        "current_yaw_deg": current_yaw_deg,
                        "error_deg": error_deg,
                        "integral_error": integral_error,
                        "dyaw_deg": dyaw_deg,
                        "raw_output_deg": raw_output_deg,
                        "steering_deg": steering_deg,
                        "max_abs_raw": max_abs_raw_output,
                        "sat_steps": sat_steps,
                        "steering_rate_cost": steering_rate_cost,
                        "total_cost": total_cost,
                        "control_mode": "ackermann"
                    }
                    trace_data.append(trace_record)

            else:  # differential drive mode
                # Original PID calculation (derivative-on-measurement to reduce kick)
                error = target_yaw_rad - current_yaw
                integral_error += error * dt
                dyaw = (current_yaw - prev_yaw) / dt
                raw_output = (Kp * error) + (Ki * integral_error) - (Kd * dyaw)

                # Track saturation
                abs_raw_output = abs(raw_output)
                if abs_raw_output > max_abs_raw_output:
                    max_abs_raw_output = abs_raw_output

                turn_speed = float(
                    np.clip(raw_output, -pid_output_limit, pid_output_limit)
                )
                if abs_raw_output > pid_output_limit:
                    sat_steps += 1

                # Anti-windup: stop integrating when saturated
                if raw_output != turn_speed and np.sign(raw_output) == np.sign(
                    error
                ):
                    integral_error -= error * dt

                # Apply control to wheels
                for j in left_wheels:
                    p.setJointMotorControl2(
                        robotId,
                        j,
                        p.VELOCITY_CONTROL,
                        targetVelocity=-turn_speed,
                        force=control_force,
                    )
                for j in right_wheels:
                    p.setJointMotorControl2(
                        robotId,
                        j,
                        p.VELOCITY_CONTROL,
                        targetVelocity=turn_speed,
                        force=control_force,
                    )

                # Calculate cost for differential drive
                saturation_excess = max(0.0, abs_raw_output - pid_output_limit)
                total_cost += (
                    (error**2)
                    + (0.001 * (turn_speed**2))
                    + (pid_sat_penalty * (saturation_excess**2))
                )

                # TRACE LOGGING (Differential)
                if return_full_trace:
                    trace_record = {
                        "step": i,
                        "time": i * dt,
                        "Kp": Kp, "Ki": Ki, "Kd": Kd,
                        "target_yaw_rad": target_yaw_rad,
                        "current_yaw_rad": current_yaw,
                        "error_rad": error,
                        "integral_error": integral_error,
                        "dyaw_rad": dyaw,
                        "raw_output": raw_output,
                        "turn_speed": turn_speed,
                        "max_abs_raw": max_abs_raw_output,
                        "sat_steps": sat_steps,
                        "total_cost": total_cost,
                        "control_mode": "differential"
                    }
                    trace_data.append(trace_record)

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
        hard_penalty_applied = 0.0
        if pid_strict_output_limit:
            if max_abs_raw_output > pid_output_limit:
                excess_ratio = (
                    max_abs_raw_output - pid_output_limit
                ) / pid_output_limit
                hard_penalty_applied = pid_sat_hard_penalty * (excess_ratio**2)
                total_cost += hard_penalty_applied

        # Calculate fitness and saturation info
        fitness = float(total_cost / simulation_steps)
        sat_info = {
            "max_abs_raw_output": float(max_abs_raw_output),
            "sat_fraction": float(sat_steps / simulation_steps),
        }

        # DEBUGGING: Print cost breakdown (only if not realtime to avoid spam)
        if control_mode == "ackermann" and not realtime and label_text:
            avg_fitness = total_cost / simulation_steps
            sat_ratio = sat_steps / simulation_steps
            hard_contrib = (
                hard_penalty_applied / simulation_steps
                if hard_penalty_applied > 0
                else 0
            )
            print(f"\n{'='*60}")
            print(f"COST BREAKDOWN for {label_text}:")
            print(f"  Avg Fitness:     {avg_fitness:.6f}")
            print(
                f"  Sat Fraction:    {sat_ratio:.2%}  ({sat_steps}/{simulation_steps} steps)"
            )
            print(
                f"  Max Steering:    {max_abs_raw_output:.3f}°  (limit: {pid_output_limit:.3f}°)"
            )
            if hard_penalty_applied > 0:
                print(
                    f"  Hard Penalty:    {hard_contrib:.6f}  (⚠ steering exceeded limit)"
                )
            print(f"  PID Params:      Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}")
            print(f"{'='*60}\n")

        return fitness, history, sat_info, trace_data

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

            fitness, history, sat, trace_data = evaluate_pid(
                params,
                label_text=label,
                return_history=return_history,
                realtime=realtime,
                return_full_trace=msg.get("return_full_trace", False),
            )

            resp_q.put(
                {
                    "id": job_id,
                    "fitness": float(fitness),
                    "history": history,
                    "sat": sat,
                    "trace_data": trace_data,
                }
            )

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
        self.ctx = mp.get_context("spawn")
        self.req_q = self.ctx.Queue()
        self.resp_q = self.ctx.Queue()
        self.worker = None
        self.next_id = 0

    def start(self):
        """Start the simulation worker process."""
        self.worker = self.ctx.Process(
            target=bullet_worker,
            args=(self.req_q, self.resp_q, self.config, self.robot_config),
            daemon=False,
        )
        self.worker.start()

    def evaluate(
        self, params, label_text="", return_history=False, realtime=False, return_full_trace=False
    ):
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

        self.req_q.put(
            {
                "type": "eval",
                "id": job_id,
                "params": [float(x) for x in params],
                "label_text": label_text,
                "return_history": bool(return_history),
                "realtime": bool(realtime),
                "return_full_trace": bool(return_full_trace),
            }
        )

        while True:
            resp = self.resp_q.get()
            if resp["id"] == job_id:
                if return_full_trace:
                     return float(resp["fitness"]), resp["history"], resp["sat"], resp["trace_data"]
                return float(resp["fitness"]), resp["history"], resp["sat"]

    def shutdown(self):
        """Shutdown the simulation worker process."""
        self.req_q.put({"type": "shutdown"})
        if self.worker:
            self.worker.join(timeout=3)
