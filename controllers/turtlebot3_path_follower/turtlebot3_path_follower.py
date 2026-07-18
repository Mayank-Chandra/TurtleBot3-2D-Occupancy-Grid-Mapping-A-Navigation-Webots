import math
import os
import socket
import threading
import numpy as np
from controller import Supervisor

# --- Telemetry Server Configuration ---
HOST = '127.0.0.1'
PORT = 65432
live_telemetry = {"x": 0.0, "y": 0.0}

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

# Start telemetry broadcast server
threading.Thread(target=telemetry_server, daemon=True).start()

# --- GRID Map & Target Parameters ---
MAP_SIZE_M = 5.0
MAP_RES_M = 0.02
WAYPOINTS_INPUT_PATH = "/home/mayank-linux/Desktop/Projects/Custom-2D-Lidar-SLAM-MPC-Navigation-Stack/controllers/turtlebot3_keyboard/waypoints.txt"

WAYPOINT_DIST_TOLERANCE = 0.08  # Meters (8 cm)
MAX_SPEED = 6.67                # TurtleBot3Burger maximum motor speed

def load_waypoints(filepath):
    points = []
    if not os.path.exists(filepath):
        print(f"Waiting for waypoints file at: {filepath}...")
        return points
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                wx, wy = line.split(",")
                points.append((float(wx), float(wy)))
    return points

# --- Initialize Supervisor ---
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Get handle to the robot's own node for ground-truth tracking
robot_node = robot.getSelf()
trans_field = robot_node.getField("translation")  # kept for reference, no longer used for pose

WHEEL_RADIUS = 0.033
WHEEL_BASE = 0.160

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

waypoints = load_waypoints(WAYPOINTS_INPUT_PATH)
current_wp_idx = 0

if len(waypoints) > 0:
    print(f"Successfully loaded {len(waypoints)} tracking waypoints from file!")
else:
    print("Warning: No waypoints found. Run pathfinder.py first.")

while robot.step(timestep) != -1:
    # 1. Get exact ground truth position and heading from Webots
    # Use getPosition() instead of the translation field -- it returns the
    # FULL global/world-resolved position, whereas the translation field is
    # only relative to the immediate parent node. This was causing the
    # plotted position (pathfinder.py) to disagree with the actual pose
    # shown in the Webots 3D view.
    pos = robot_node.getPosition()
    x = pos[0]
    y = pos[1]

    # Calculate exact heading angle theta from the orientation matrix.
    # getOrientation() returns the 3x3 body->world rotation matrix, row-major:
    # [m00, m01, m02, m10, m11, m12, m20, m21, m22]. The robot's forward
    # (local +X) direction in world coords is the matrix's FIRST COLUMN,
    # i.e. (m00, m10) = (rot_matrix[0], rot_matrix[3]) -- NOT the first row.
    rot_matrix = robot_node.getOrientation()
    theta = math.atan2(rot_matrix[3], rot_matrix[0])

    # Update background network server coordinates
    live_telemetry["x"] = float(x)
    live_telemetry["y"] = float(y)

    left_speed = 0.0
    right_speed = 0.0

    # 2. Autonomous Tracking Control Loop
    if current_wp_idx < len(waypoints):
        target_x, target_y = waypoints[current_wp_idx]

        dx = target_x - x
        dy = target_y - y
        distance_to_wp = math.sqrt(dx**2 + dy**2)

        if distance_to_wp < WAYPOINT_DIST_TOLERANCE:
            print(f"Reached Waypoint index {current_wp_idx}: ({target_x:.2f}, {target_y:.2f})")
            current_wp_idx += 1
        else:
            target_theta = math.atan2(dy, dx)
            heading_error = target_theta - theta
            heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

            k_linear = 2.5
            k_angular = 5.0

            linear_vel = k_linear * distance_to_wp
            angular_vel = k_angular * heading_error

            linear_vel = np.clip(linear_vel, -0.2, 0.2)
            angular_vel = np.clip(angular_vel, -1.5, 1.5)

            v_left = (linear_vel - (angular_vel * WHEEL_BASE / 2.0)) / WHEEL_RADIUS
            v_right = (linear_vel + (angular_vel * WHEEL_BASE / 2.0)) / WHEEL_RADIUS

            left_speed = np.clip(v_left, -MAX_SPEED, MAX_SPEED)
            right_speed = np.clip(v_right, -MAX_SPEED, MAX_SPEED)
    else:
        if len(waypoints) > 0:
            print("--- GOAL DESTINATION TARGET REACHED SUCCESSFULLY! ---")
            waypoints = []

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)