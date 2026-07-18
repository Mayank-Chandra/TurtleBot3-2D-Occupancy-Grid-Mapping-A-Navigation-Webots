import math
import numpy as np
from controller import Supervisor
from PIL import Image

# --- GRID Map Parameters (Refined for High Detail) ---
MAP_SIZE_M = 5.0
MAP_RES_M = 0.02
GRID_SIZE = int(MAP_SIZE_M / MAP_RES_M)  # 250x250 grid

# Initialize grid map with prior log-odds of 0 (probability 0.5)
map_log_odds = np.zeros((GRID_SIZE, GRID_SIZE))

# Log-odds update values
L_FREE = -0.15
L_OCCUPIED = 1.5

# --- Display follow-camera ---
# Set to an integer N to show only an N x N cell window centered on the
# robot (it pans/zooms with the robot as it drives). Set to None to show
# the entire GRID_SIZE x GRID_SIZE map instead (old behavior).
FOLLOW_WINDOW_CELLS = 150  # 150 cells * 0.02 m/cell = 3m x 3m visible area


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
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# Ground-truth pose access (same technique as turtlebot3_path_follower).
# Requires this Robot node's "supervisor" field to be TRUE in the scene tree
# -- it already is, since the path-follower controller uses it successfully.
robot_node = robot.getSelf()
trans_field = robot_node.getField('translation')

WHEEL_RADIUS = 0.033
WHEEL_BASE = 0.160

# Get Motors
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# LiDar initialization
lidar = robot.getDevice('LDS-01')
lidar.enable(timestep)

# Display initialization
display = robot.getDevice('map_display')

# Enable Keyboard
turtlebot_key = robot.getKeyboard()
turtlebot_key.enable(timestep)

# Mapping States
mapping_enabled = True
prev_m_key_state = False

print('--SLAM MAPPER RUNNING--')
print('--GROUND-TRUTH (SUPERVISOR) POSE Controller--')
print('Controls: W/A/S/D to drive | P to save map | M to toggle mapping ON/OFF')

while robot.step(timestep) != -1:
    # 1. Read exact ground-truth position and heading straight from Webots.
    # This is the SAME technique turtlebot3_path_follower.py uses, so the
    # SLAM map and the live navigation position now live in one consistent
    # world frame -- no odometry drift, no initial-pose guessing, and no
    # more "map dot in the wrong place" mismatch between the two scripts.
    pos = trans_field.getSFVec3f()
    x = pos[0]
    y = pos[1]

    rot_matrix = robot_node.getOrientation()
    theta = math.atan2(rot_matrix[3], rot_matrix[0])

    # Get robot map cell index
    robot_mx, robot_my = world_to_map(x, y)

    # 5. Lidar Data Update
    if mapping_enabled:
        ranges = lidar.getRangeImage()
        num_beams = len(ranges)

        # Calculate global position of the physically offset LiDAR sensor (-3cm along local x-axis)
        lidar_x = x - 0.03 * math.cos(theta)
        lidar_y = y - 0.03 * math.sin(theta)

        # Get LiDAR map index cell to use as the ray's start point
        lidar_mx, lidar_my = world_to_map(lidar_x, lidar_y)

        for i in range(0, num_beams, 1):
            r = ranges[i]

            if r >= 3.5 or math.isnan(r) or r <= 0.12:
                continue

            # Corrected angle calculation mapping clockwise left-to-right indices
            beam_angles = theta + math.pi - (i * (2.0 * math.pi / num_beams))
            beam_angles = (beam_angles + math.pi) % (2.0 * math.pi) - math.pi

            # Global obstacles coordinates computed from the actual LiDAR position
            wall_x = lidar_x + r * math.cos(beam_angles)
            wall_y = lidar_y + r * math.sin(beam_angles)

            wall_mx, wall_my = world_to_map(wall_x, wall_y)

            # Bound check
            if 0 <= wall_mx < GRID_SIZE and 0 <= wall_my < GRID_SIZE:
                # Ray trace from lidar_mx/y instead of robot center
                ray_cells = get_line_cells(lidar_mx, lidar_my, wall_mx, wall_my)

                # Apply L_FREE along the ray (excluding the hit point)
                for (cx, cy) in ray_cells[:-1]:
                    if 0 <= cx < GRID_SIZE and 0 <= cy < GRID_SIZE:
                        map_log_odds[cx, cy] += L_FREE

                # Apply L_OCCUPIED only to the exact wall endpoint
                map_log_odds[wall_mx, wall_my] += L_OCCUPIED

        # Clip map values once per time step
        map_log_odds = np.clip(map_log_odds, -5.0, 5.0)

    prob_map = 1.0 - (1.0 / (1.0 + np.exp(map_log_odds)))

    # Convert probability to Grayscale (Unexplored = Grey, Free = White, Wall = Black)
    gray_map = ((1.0 - prob_map) * 255).astype(np.uint8)
    gray_map = np.flipud(gray_map.T)  # gray_map[mx, my] -> display[row, col]
    # This transform means: display_row = GRID_SIZE - 1 - my, display_col = mx

    # --- DRAW THE MAP & ROBOT TO THE WEBOTS DISPLAY ---
    # Robot's position in the same (row, col) convention as gray_map above.
    rob_row_disp = GRID_SIZE - 1 - robot_my
    rob_col_disp = robot_mx

    if display is not None:
        disp_w = display.getWidth()
        disp_h = display.getHeight()

        if FOLLOW_WINDOW_CELLS is not None:
            win = min(FOLLOW_WINDOW_CELLS, GRID_SIZE)
            half = win // 2

            row_start = int(np.clip(rob_row_disp - half, 0, GRID_SIZE - win))
            col_start = int(np.clip(rob_col_disp - half, 0, GRID_SIZE - win))

            window = gray_map[row_start:row_start + win, col_start:col_start + win]
            display_img = np.stack((window,) * 3, axis=-1)

            # Robot's position relative to the cropped window
            rob_row_win = rob_row_disp - row_start
            rob_col_win = rob_col_disp - col_start
        else:
            display_img = np.stack((gray_map,) * 3, axis=-1)
            rob_row_win = rob_row_disp
            rob_col_win = rob_col_disp

        # Overlay the robot's current position as a red dot (in window coords)
        win_h, win_w = display_img.shape[0], display_img.shape[1]
        if 0 <= rob_col_win < win_w and 0 <= rob_row_win < win_h:
            for dx_offset in [-1, 0, 1]:
                for dy_offset in [-1, 0, 1]:
                    cx_p = np.clip(rob_col_win + dx_offset, 0, win_w - 1)
                    cy_p = np.clip(rob_row_win + dy_offset, 0, win_h - 1)
                    display_img[cy_p, cx_p] = [255, 0, 0]

        # Resize the (possibly cropped) window up to the display's actual
        # pixel resolution, so the follow-window fills the whole display
        # and appears "zoomed in" rather than a small patch in the corner.
        if display_img.shape[1] != disp_w or display_img.shape[0] != disp_h:
            im = Image.fromarray(display_img)
            im = im.resize((disp_w, disp_h), Image.NEAREST)
            display_img = np.array(im)

        display_bytes = display_img.tobytes()
        ir = display.imageNew(display_bytes, display.RGB, disp_w, disp_h)
        display.imagePaste(ir, 0, 0, False)
        display.imageDelete(ir)

    total_walls_mapped = np.sum(prob_map > 0.7)
    status_str = "ACTIVE" if mapping_enabled else "PAUSED"

    # Console output
    print(f"Robot: ({x:.2f}, {y:.2f}) | Mapping: {status_str} | Target Cells Discovered: {total_walls_mapped}")

    # 6. Keyboard Driving Logic
    key = turtlebot_key.getKey()
    left_speed = 0.0
    right_speed = 0.0

    if key == ord('W') or key == ord('w'):
        left_speed, right_speed = 6.0, 6.0
    elif key == ord('S') or key == ord('s'):
        left_speed, right_speed = -5.0, -5.0
    elif key == ord('A') or key == ord('a'):
        left_speed, right_speed = -2.0, 2.0
    elif key == ord('D') or key == ord('d'):
        left_speed, right_speed = 2.0, -2.0

    # Toggle Mapping
    m_pressed = (key == ord('M') or key == ord('m'))
    if m_pressed and not prev_m_key_state:
        mapping_enabled = not mapping_enabled
        print(f"\n>>> Mapping toggled! New state: {mapping_enabled} <<<\n")
    prev_m_key_state = m_pressed

    # -- Saving the Map --
    if key == ord('P') or key == ord('p'):
        print('Saving Map...')
        img = Image.fromarray(gray_map)
        img.save('webots_map.png')
        print('Map Saved successfully as webots_map.png!')

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)