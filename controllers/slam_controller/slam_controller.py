import math
import os
import socket
import threading
import numpy as np
from controller import Supervisor
from PIL import Image

# --- Telemetry & Map Configuration Paths ---
HOST = '127.0.0.1'
PORT = 65432
live_telemetry = {"x": 0.0, "y": 0.0}

MAP_SAVE_PATH = "/home/mayank-linux/Desktop/Projects/Custom-2D-Lidar-SLAM-MPC-Navigation-Stack/controllers/turtlebot3_keyboard/webots_map.png"
DYNAMIC_WAYPOINT_PATH = "/home/mayank-linux/Desktop/Projects/Custom-2D-Lidar-SLAM-MPC-Navigation-Stack/controllers/turtlebot3_keyboard/dynamic_waypoints.txt"

# --- SLAM Map Parameters ---
MAP_SIZE_M = 5.0
MAP_RES_M = 0.02
GRID_SIZE = int(MAP_SIZE_M / MAP_RES_M)

# Log-odds grid initialization
map_log_odds = np.zeros((GRID_SIZE, GRID_SIZE))
L_FREE = -0.15
L_OCCUPIED = 1.5

def world_to_map(wx, wy):
    mx = int((wx + MAP_SIZE_M / 2.0) / MAP_RES_M)
    my = int((wy + MAP_SIZE_M / 2.0) / MAP_RES_M)
    return mx, my

def telemetry_server():
    """Background thread server that shares the robot's real-time position with the pathfinder script."""
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

# Start telemetry broadcast server thread immediately
threading.Thread(target=telemetry_server, daemon=True).start()

# --- GRID Follower Navigation Parameters ---
WAYPOINT_DIST_TOLERANCE = 0.08
MAX_SPEED = 6.67
WHEEL_RADIUS = 0.033
WHEEL_BASE = 0.160

# --- LIDAR Local Avoidance Thresholds ---
LIDAR_SAFETY_DIST = 0.25  
LIDAR_FRONT_ANGLE_DEG = 30 

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
        pass 
    return points

# --- Initialize Supervisor ---
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
robot_node = robot.getSelf()

# Initialize Motors
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Initialize Lidar Sensor
lidar = robot.getDevice('LDS-01')
lidar.enable(timestep)

waypoints = []
current_wp_idx = 0
last_mod_time = 0.0
map_save_counter = 0

print("--- SLAM Controller Node Operational ---")

while robot.step(timestep) != -1:
    # 1. State Tracking Estimation Updates (Supervisor API Ground-Truth)
    pos = robot_node.getPosition()
    x, y = pos[0], pos[1]
    rot_matrix = robot_node.getOrientation()
    theta = math.atan2(rot_matrix[3], rot_matrix[0])

    live_telemetry["x"] = float(x)
    live_telemetry["y"] = float(y)

    # 2. SLAM: Update Log-Odds Occupancy Grid using Lidar Rays
    ranges = lidar.getRangeImage()
    num_beams = len(ranges)
    
    # Compute robot pose in mapping frame space
    lidar_x = x - 0.03 * math.cos(theta)
    lidar_y = y - 0.03 * math.sin(theta)

    for i in range(num_beams):
        r = ranges[i]
        if r >= 3.5 or math.isnan(r) or r <= 0.12:
            continue
            
        beam_angle = theta + math.pi - (i * (2.0 * math.pi / num_beams))
        beam_angle = (beam_angle + math.pi) % (2.0 * math.pi) - math.pi

        wall_x = lidar_x + r * math.cos(beam_angle)
        wall_y = lidar_y + r * math.sin(beam_angle)
        
        wall_mx, wall_my = world_to_map(wall_x, wall_y)

        if 0 <= wall_mx < GRID_SIZE and 0 <= wall_my < GRID_SIZE:
            map_log_odds[wall_my, wall_mx] += L_OCCUPIED

    # 3. Periodically export occupancy grid to PNG file image layout
    map_save_counter += 1
    if map_save_counter % 50 == 0:
        prob_map = 1.0 - (1.0 / (1.0 + np.exp(np.clip(map_log_odds, -5.0, 5.0))))
        gray_map = ((1.0 - prob_map) * 255).astype(np.uint8)
        gray_map = np.flipud(gray_map)  # Row-flip only -- matches pathfinder.py's
                                         # (mx, (GRID_SIZE-1)-my) read-back convention
        
        img = Image.fromarray(gray_map)
        temp_map_path = MAP_SAVE_PATH.replace(".png", "_temp.png")
        img.save(temp_map_path)
        os.replace(temp_map_path, MAP_SAVE_PATH)

    # 4. Check for dynamic path file modifications mid-run
    if os.path.exists(DYNAMIC_WAYPOINT_PATH):
        mod_time = os.path.getmtime(DYNAMIC_WAYPOINT_PATH)
        if mod_time > last_mod_time:
            last_mod_time = mod_time
            waypoints = load_waypoints(DYNAMIC_WAYPOINT_PATH)
            current_wp_idx = 0
            print("-> New live path loaded successfully!")

    # 5. Lidar Reactive Safety Monitor (Local Layer Override Window Check)
    obstacle_detected = False
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
        print("Warning: Dynamic obstacle detected! Executing local safety maneuver.")
        left_speed = -1.5
        right_speed = 1.5
    elif current_wp_idx < len(waypoints):
        target_x, target_y = waypoints[current_wp_idx]
        dx, dy = target_x - x, target_y - y
        distance_to_wp = math.sqrt(dx**2 + dy**2)

        if distance_to_wp < WAYPOINT_DIST_TOLERANCE:
            print(f"Reached Waypoint index {current_wp_idx}: ({target_x:.2f}, {target_y:.2f})")
            current_wp_idx += 1
        else:
            target_theta = math.atan2(dy, dx)
            heading_error = (target_theta - theta + math.pi) % (2 * math.pi) - math.pi

            linear_vel = np.clip(2.5 * distance_to_wp, -0.2, 0.2)
            angular_vel = np.clip(5.0 * heading_error, -1.5, 1.5)

            v_left = (linear_vel - (angular_vel * WHEEL_BASE / 2.0)) / WHEEL_RADIUS
            v_right = (linear_vel + (angular_vel * WHEEL_BASE / 2.0)) / WHEEL_RADIUS

            left_speed = np.clip(v_left, -MAX_SPEED, MAX_SPEED)
            right_speed = np.clip(v_right, -MAX_SPEED, MAX_SPEED)
    else:
        if len(waypoints) > 0:
            print("--- Dynamic Path Tracking Phase Complete ---")
            waypoints = []

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)