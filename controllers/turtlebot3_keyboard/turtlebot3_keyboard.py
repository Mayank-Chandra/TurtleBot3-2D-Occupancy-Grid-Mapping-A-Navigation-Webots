import math
import numpy as np
from controller import Robot
from PIL import Image

# --- GRID Map Parameters ---
MAP_SIZE_M = 5.0
MAP_RES_M = 0.03
GRID_SIZE = int(MAP_SIZE_M / MAP_RES_M) # 333x333 grid

# Initialize grid map with prior log-odds of 0 (probability 0.5)
map_log_odds = np.zeros((GRID_SIZE, GRID_SIZE))

# Log-odds update values
L_FREE = -0.4
L_OCCUPIED = 0.85

def world_to_map(wx, wy):
    mx = int((wx + MAP_SIZE_M / 2.0) / MAP_RES_M)
    my = int((wy + MAP_SIZE_M / 2.0) / MAP_RES_M)
    return mx, my

def get_line_cells(x0, y0, x1, y1):
    """Bresenham's Line Algorithm to find all cells between robot and obstacles."""
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return cells

# --- Robot Initialization ---
robot = Robot()
timestep = int(robot.getBasicTimeStep())

WHEEL_RADIUS = 0.033
WHEEL_BASE = 0.160

# Get Motors
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Get Positions from Sensors
left_enc = robot.getDevice('left wheel sensor')
right_enc = robot.getDevice('right wheel sensor')
left_enc.enable(timestep)
right_enc.enable(timestep)

# LiDar initialization
lidar = robot.getDevice('LDS-01')
lidar.enable(timestep)

# Enable Keyboard
turtlebot_key = robot.getKeyboard()
turtlebot_key.enable(timestep)

# Odometry Variables
x = 0.0
y = 0.0
theta = 0.0
prev_left_pos = 0.0
prev_right_pos = 0.0
first_step = True

# Mapping States
mapping_enabled = True
prev_m_key_state = False

print('--SLAM MAPPER RUNNING--')
print('--ODOMETRY Controller--')
print('Controls: W/A/S/D to drive | P to save map | M to toggle mapping ON/OFF')

while robot.step(timestep) != -1:
    # 1. Reading the encoders
    left_pos = left_enc.getValue()
    right_pos = right_enc.getValue()
    
    if first_step:
        prev_left_pos = left_pos
        prev_right_pos = right_pos
        first_step = False
        continue 
    
    # 2. Calculate actual encoder ticks difference
    d_left_ticks = left_pos - prev_left_pos
    d_right_ticks = right_pos - prev_right_pos
    
    # Update previous positions for the next loop step
    prev_left_pos = left_pos
    prev_right_pos = right_pos
    
    # 3. Convert encoder rotations to actual wheel travel distance (meters)
    dist_left = d_left_ticks * WHEEL_RADIUS
    dist_right = d_right_ticks * WHEEL_RADIUS
    
    # 4. Compute robot translation and rotation change
    ds = (dist_right + dist_left) / 2.0
    d_theta = (dist_right - dist_left) / WHEEL_BASE
    theta += d_theta
    # Normalizing theta between pi and -pi
    theta = (theta + math.pi) % (2 * math.pi) - math.pi
    
    # Update global coordinates
    x += ds * math.cos(theta + d_theta / 2.0)
    y += ds * math.sin(theta + d_theta / 2.0)

    # Get robot map cell index
    robot_mx, robot_my = world_to_map(x, y)

    # 5. Lidar Data Update
    if mapping_enabled:
        ranges = lidar.getRangeImage()
        num_beams = len(ranges)

        for i in range(0, num_beams, 4):
            r = ranges[i]

            if r >= 3.5 or math.isnan(r) or r <= 0.12:
                continue
            
            beam_angles = theta + (i * (2 * math.pi / num_beams))
            # Global obstacles coordinates
            wall_x = x + r * math.cos(beam_angles)
            wall_y = y + r * math.sin(beam_angles)

            wall_mx, wall_my = world_to_map(wall_x, wall_y)

            # Bound check
            if 0 <= wall_mx < GRID_SIZE and 0 <= wall_my < GRID_SIZE:
                ray_cells = get_line_cells(robot_mx, robot_my, wall_mx, wall_my)

                # Apply L_FREE along the ray (excluding the hit point)
                for (cx, cy) in ray_cells[:-1]:
                    if 0 <= cx < GRID_SIZE and 0 <= cy < GRID_SIZE:
                        map_log_odds[cx, cy] += L_FREE
                
                # Apply L_OCCUPIED only to the exact wall endpoint (OUTSIDE the free cells loop)
                map_log_odds[wall_mx, wall_my] += L_OCCUPIED

        # Clip map values once per time step to save massive amounts of CPU (OUTSIDE the beams loop)
        map_log_odds = np.clip(map_log_odds, -5.0, 5.0)

    prob_map = 1.0 - (1.0 / (1.0 + np.exp(map_log_odds)))

    total_walls_mapped = np.sum(prob_map > 0.7)
    status_str = "ACTIVE" if mapping_enabled else "PAUSED"
    
    # Console output
    print(f"Robot: ({x:.2f}, {y:.2f}) | Mapping: {status_str} | Target Cells Discovered: {total_walls_mapped}")
    print(f"Estimated Pose: X = {x:6.3f} m | Y = {y:6.3f} m | Theta = {math.degrees(theta):6.1f}°")
    
    # 6. Keyboard Driving Logic
    key = turtlebot_key.getKey()
    left_speed = 0.0
    right_speed = 0.0
    
    if key == ord('W') or key == ord('w'):
        left_speed, right_speed = 6.0, 6.0
    elif key == ord('S') or key == ord('s'):
        left_speed, right_speed = -5.0, -5.0
    elif key == ord('A') or key == ord('a'):
        left_speed, right_speed = -3.0, 3.0
    elif key == ord('D') or key == ord('d'):
        left_speed, right_speed = 3.0, -3.0

    # Toggle Mapping
    m_pressed = (key == ord('M') or key == ord('m'))
    if m_pressed and not prev_m_key_state:
        mapping_enabled = not mapping_enabled
        print(f"\n>>> Mapping toggled! New state: {mapping_enabled} <<<\n")
    prev_m_key_state = m_pressed

    # -- Saving the Map -- 
    if key == ord('P') or key == ord('p'):
        print('Saving Map...')
        gray_map = ((1.0 - prob_map) * 255).astype(np.uint8)
        gray_map = np.flipud(gray_map.T)

        img = Image.fromarray(gray_map)
        img.save('webots_map.png')
        print('Map Saved successfully as webots_map.png!')
        
    # Send calculated velocities to motors
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)