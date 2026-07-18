import math
import os
import socket
import threading
import numpy as np
from controller import Supervisor

HOST = '127.0.0.1'
PORT = 65432
live_telemetry = {"x": 0.0, "y": 0.0}

def telemetry_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        while True:
            try:
                conn, addr = s.accept()
                with conn:
                    data_str = f"{live_telemetry['x']:.6f},{live_telemetry['y']:.6f}"
                    conn.sendall(data_str.encode('utf-8'))
            except Exception:
                pass

threading.Thread(target=telemetry_server, daemon=True).start()

DYNAMIC_WAYPOINT_PATH = "/home/mayank-linux/Desktop/Projects/Custom-2D-Lidar-SLAM-MPC-Navigation-Stack/controllers/turtlebot3_keyboard/dynamic_waypoints.txt"
WAYPOINT_DIST_TOLERANCE = 0.08
MAX_SPEED = 6.67
WHEEL_RADIUS = 0.033
WHEEL_BASE = 0.160

# --- LIDAR Local Avoidance Thresholds ---
LIDAR_SAFETY_DIST = 0.25  # Stop/steer away if an object gets within 25cm
LIDAR_FRONT_ANGLE_DEG = 30 # Front arc window matching steering limits

def load_waypoints(filepath):
    points = []
    if not os.path.exists(filepath):
        return points
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    wx, wy = line.split(",")
                    points.append((float(wx), float(wy)))
    except Exception:
        pass # Handle mid-write read locks safely
    return points

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
robot_node = robot.getSelf()
trans_field = robot_node.getField("translation")  # kept for reference, no longer used for pose

# Initialize Motors
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Initialize Lidar Sensor
lidar = robot.getDevice('LDS-01')
lidar.enable(timestep)

waypoints = []
current_wp_idx = 0
last_mod_time = 0.0

while robot.step(timestep) != -1:
    
    pos = robot_node.getPosition()
    x, y = pos[0], pos[1]
    rot_matrix = robot_node.getOrientation()
    theta = math.atan2(rot_matrix[3], rot_matrix[0])

    live_telemetry["x"] = float(x)
    live_telemetry["y"] = float(y)

    # 2. Check for dynamic path file modifications mid-run
    if os.path.exists(DYNAMIC_WAYPOINT_PATH):
        mod_time = os.path.getmtime(DYNAMIC_WAYPOINT_PATH)
        if mod_time > last_mod_time:
            last_mod_time = mod_time
            waypoints = load_waypoints(DYNAMIC_WAYPOINT_PATH)
            current_wp_idx = 0
            print("-> New live path loaded successfully!")

    # 3. Lidar Reactive Safety Monitor (Local Layer Override)
    ranges = lidar.getRangeImage()
    num_beams = len(ranges)
    obstacle_detected = False

    # Check front window arc bounds
    for i in range(num_beams):
        beam_deg = (i * 360.0) / num_beams
        if beam_deg < LIDAR_FRONT_ANGLE_DEG or beam_deg > (360.0 - LIDAR_FRONT_ANGLE_DEG):
            r = ranges[i]
            if 0.05 < r < LIDAR_SAFETY_DIST:
                obstacle_detected = True
                break

    left_speed = 0.0
    right_speed = 0.0

    if obstacle_detected:
        # EMERGENCY OVERRIDE: Stop and pivot to clear the unmapped hazard safely
        print("Warning: Dynamic obstacle detected! Executing local safety maneuver.")
        left_speed = -1.5
        right_speed = 1.5
    elif current_wp_idx < len(waypoints):
        # Normal Global Path Tracking Mode
        target_x, target_y = waypoints[current_wp_idx]
        dx, dy = target_x - x, target_y - y
        distance_to_wp = math.sqrt(dx**2 + dy**2)

        if distance_to_wp < WAYPOINT_DIST_TOLERANCE:
            current_wp_idx += 1
        else:
            target_theta = math.atan2(dy, dx)
            heading_error = (target_theta - theta + math.pi) % (2 * math.pi) - math.pi

            linear_vel = np.clip(2.5 * distance_to_wp, -0.2, 0.2)
            angular_vel = np.clip(5.0 * heading_error, -1.5, 1.5)

            left_speed = np.clip((linear_vel - (angular_vel * WHEEL_BASE / 2.0)) / WHEEL_RADIUS, -MAX_SPEED, MAX_SPEED)
            right_speed = np.clip((linear_vel + (angular_vel * WHEEL_BASE / 2.0)) / WHEEL_RADIUS, -MAX_SPEED, MAX_SPEED)

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)
