import math
import os
import socket
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import heapq

# --- Configuration paths ---
MAP_IMAGE_PATH = "/home/mayank-linux/Desktop/Projects/Custom-2D-Lidar-SLAM-MPC-Navigation-Stack/controllers/turtlebot3_keyboard/webots_map.png"
WAYPOINT_OP_PATH = "/home/mayank-linux/Desktop/Projects/Custom-2D-Lidar-SLAM-MPC-Navigation-Stack/controllers/turtlebot3_keyboard/waypoints.txt"
OBSTACLE_THRESHOLD = 50

# Where each run's annotated map + a text summary get saved, so you keep a
# record of where the robot started and where it was told to go each time.
PATH_LOG_DIR = os.path.join(os.path.dirname(MAP_IMAGE_PATH), "path_logs")

# GRID Map Parameters
MAP_SIZE_M = 5.0
MAP_RES_M = 0.02
GRID_SIZE = int(MAP_SIZE_M/MAP_RES_M)

# Physical Safety Buffer
SAFETY_RADIUS_M = 0.12
SAFETY_RADIUS_CELLS = int(round(SAFETY_RADIUS_M / MAP_RES_M))

# Fallback positions if simulation is offline
DEFAULT_X = 0.486022
DEFAULT_Y = 1.206470
# GOAL_NODE is no longer hardcoded -- it's picked interactively by clicking
# on the map (see select_goal_interactively() below).


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


def heuristic(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def get_neighbors(node, grid):
    neighbors = []
    x, y = node
    height, width = grid.shape

    directions = [
        (0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
        (1, 1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (-1, -1, 1.414)
    ]

    for dx, dy, cost in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            if grid[ny, nx] != 0:
                continue

            # Prevent diagonal "corner-cutting": for a diagonal move, also
            # require BOTH orthogonal cells adjacent to it to be free.
            # Otherwise the diagonal step can clip straight through a wall
            # corner even though the diagonal cell itself is clear.
            if dx != 0 and dy != 0:
                if grid[y, x + dx] != 0 or grid[y + dy, x] != 0:
                    continue

            neighbors.append(((nx, ny), cost))
    return neighbors


def get_live_webots_position():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            s.connect(('127.0.0.1', 65432))
            response = s.recv(1024).decode('utf-8')
            wx, wy = map(float, response.split(','))
            print(f"-> Connected to Webots! Current Live Position: X={wx:.4f}, Y={wy:.4f}")
            return wx, wy
    except (socket.timeout, ConnectionRefusedError):
        print(f"-> Webots offline. Using fallback configuration pose: X={DEFAULT_X}, Y={DEFAULT_Y}")
        return DEFAULT_X, DEFAULT_Y


def world_to_map(wx, wy):
    mx = int((wx + MAP_SIZE_M / 2.0) / MAP_RES_M)
    my = int((wy + MAP_SIZE_M / 2.0) / MAP_RES_M)
    return mx, my


def map_to_world(mx, my):
    wx = (mx * MAP_RES_M) - (MAP_SIZE_M / 2.0) + (MAP_RES_M / 2.0)
    wy = (my * MAP_RES_M) - (MAP_SIZE_M / 2.0) + (MAP_RES_M / 2.0)
    return wx, wy


def inflate_obstacles(grid, radius_cells):
    if radius_cells <= 0:
        return grid
    inflated_grid = np.copy(grid)
    height, width = grid.shape
    file_idx, col_idx = np.where(grid == 1)

    for r, c in zip(file_idx, col_idx):
        r_start = max(0, r - radius_cells)
        r_end = min(height, r + radius_cells + 1)
        c_start = max(0, c - radius_cells)
        c_end = min(width, c + radius_cells + 1)
        inflated_grid[r_start:r_end, c_start:c_end] = 1

    return inflated_grid


def a_star_search(grid, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {start: None}
    cost_so_far = {start: 0.0}

    while not frontier.empty():
        current = frontier.get()
        if current == goal:
            break
        for neighbor, move_cost in get_neighbors(current, grid):
            new_cost = cost_so_far[current] + move_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(goal, neighbor)
                frontier.put(neighbor, priority)
                came_from[neighbor] = current

    if goal not in came_from:
        return None

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]

    path.reverse()
    return path


def has_line_of_sight(grid, a, b):
    """Bresenham line check: True if every cell on the straight line from
    a to b is free (and stays in bounds) in the given (inflated) grid."""
    x0, y0 = a
    x1, y1 = b
    height, width = grid.shape

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
    while True:
        if not (0 <= x < width and 0 <= y < height):
            return False
        if grid[y, x] != 0:
            return False
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return True


def smooth_path(grid, path):
    """String-pulling / line-of-sight pruning: greedily jump to the
    farthest path node that's still directly visible (safety-grid clear),
    turning a jagged grid-stepped path into a few clean straight segments."""
    if len(path) < 3:
        return path

    smoothed = [path[0]]
    anchor_idx = 0

    while anchor_idx < len(path) - 1:
        farthest_idx = anchor_idx + 1
        for candidate_idx in range(len(path) - 1, anchor_idx, -1):
            if has_line_of_sight(grid, path[anchor_idx], path[candidate_idx]):
                farthest_idx = candidate_idx
                break
        smoothed.append(path[farthest_idx])
        anchor_idx = farthest_idx

    return smoothed


def select_goal_interactively(display_map, inflated_grid, start_node):
    """Shows the map (with the safety buffer overlaid and the live start
    marked) and lets the user click to pick the goal. Re-prompts if the
    click is out of bounds or inside a wall/safety buffer."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(display_map)
    ax.scatter(start_node[0], start_node[1], color='green', s=100, zorder=5, label='Live Start')
    ax.legend(loc='upper right')
    height, width = inflated_grid.shape

    while True:
        ax.set_title('Click on the map to select your GOAL position')
        plt.draw()
        print("Click on the map to choose the goal position...")
        clicked = plt.ginput(1, timeout=0)

        if not clicked:
            plt.close(fig)
            return None

        gx, gy = int(round(clicked[0][0])), int(round(clicked[0][1]))

        if not (0 <= gx < width and 0 <= gy < height):
            print("That click was outside the map bounds -- try again.")
            continue

        if inflated_grid[gy, gx] != 0:
            print("That spot is inside a wall or its safety buffer -- pick a clear spot.")
            ax.scatter(gx, gy, color='orange', marker='x', s=80, zorder=5)
            continue

        plt.close(fig)
        return (gx, gy)



if __name__ == "__main__":
    try:
        img = Image.open(MAP_IMAGE_PATH).convert("L")
        map_array = np.array(img)
    except FileNotFoundError:
        print(f"Error: Could not find image at {MAP_IMAGE_PATH}")
        exit()

    raw_occupancy_grid = np.zeros_like(map_array, dtype=np.uint8)
    raw_occupancy_grid[map_array < OBSTACLE_THRESHOLD] = 1

    print(f"Inflating obstacles by safety buffer margin: {SAFETY_RADIUS_M}m...")
    inflated_occupancy_grid = inflate_obstacles(raw_occupancy_grid, SAFETY_RADIUS_CELLS)

    # Shared display map (obstacles + safety buffer overlay), reused for both
    # the interactive goal picker and the final result plot.
    display_map = np.stack((map_array,) * 3, axis=-1)
    display_map[(inflated_occupancy_grid == 1) & (raw_occupancy_grid == 0)] = [255, 180, 180]

    # Fetch Webots ground-truth position
    robot_x, robot_y = get_live_webots_position()

    # Convert directly to map array indices matching matrix transformation
    mx, my = world_to_map(robot_x, robot_y)
    START_NODE = (mx, (GRID_SIZE - 1) - my)

    if not (0 <= START_NODE[0] < inflated_occupancy_grid.shape[1] and 0 <= START_NODE[1] < inflated_occupancy_grid.shape[0]):
        print(f"Error: Start coordinate {START_NODE} is outside map bounds!")
        exit()
    if inflated_occupancy_grid[START_NODE[1], START_NODE[0]] != 0:
        print(f"Warning: Start coordinate {START_NODE} is inside an obstacle/safety buffer! "
              f"Path planning may fail or start unsafely.")

    # --- Interactive goal selection: click on the map instead of hardcoding ---
    GOAL_NODE = select_goal_interactively(display_map, inflated_occupancy_grid, START_NODE)
    if GOAL_NODE is None:
        print("No goal selected. Exiting.")
        exit()
    print(f"Goal selected at map cell: {GOAL_NODE}")

    generated_path = a_star_search(inflated_occupancy_grid, START_NODE, GOAL_NODE)

    if generated_path is None:
        print("Path Planning Failure: A valid pathway could not be found.")
    else:
        generated_path = smooth_path(inflated_occupancy_grid, generated_path)
        print(f"Success! Path calculated containing {len(generated_path)} nodes (after smoothing).")

        # Translate pixel path back to Webots world metrics
        world_waypoints = []
        for (mx, my) in generated_path:
            real_my = (GRID_SIZE - 1) - my
            wx, wy = map_to_world(mx, real_my)
            world_waypoints.append((wx, wy))

        with open(WAYPOINT_OP_PATH, "w") as f:
            for wx, wy in world_waypoints:
                f.write(f'{wx:.4f},{wy:.4f}\n')
        print(f"Successfully exported {len(world_waypoints)} safe world coordinates to {WAYPOINT_OP_PATH}")

        world_start = map_to_world(START_NODE[0], (GRID_SIZE - 1) - START_NODE[1])
        world_goal = map_to_world(GOAL_NODE[0], (GRID_SIZE - 1) - GOAL_NODE[1])

        # Visualizer output
        plt.figure(figsize=(8, 8))
        plt.imshow(display_map)
        plt.plot([p[0] for p in generated_path], [p[1] for p in generated_path], color='cyan', linewidth=3, label="Planned Path")
        plt.scatter(START_NODE[0], START_NODE[1], color='green', s=100, zorder=5, label='Live Start')
        plt.scatter(GOAL_NODE[0], GOAL_NODE[1], color='red', s=100, zorder=5, label='Goal')
        plt.title('A* Shortest Route Planning with Safety Buffers')
        plt.legend(loc='upper right')

        # --- Save a record of this run: the annotated map + a text summary ---
        os.makedirs(PATH_LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        map_snapshot_path = os.path.join(PATH_LOG_DIR, f"path_plan_{timestamp}.png")
        plt.savefig(map_snapshot_path, dpi=150, bbox_inches='tight')
        print(f"Saved annotated map snapshot to {map_snapshot_path}")

        summary_path = os.path.join(PATH_LOG_DIR, f"path_plan_{timestamp}.txt")
        with open(summary_path, "w") as f:
            f.write(f"Run timestamp: {timestamp}\n")
            f.write(f"Start (world m): x={world_start[0]:.4f}, y={world_start[1]:.4f}\n")
            f.write(f"Start (map cell): {START_NODE}\n")
            f.write(f"Goal  (world m): x={world_goal[0]:.4f}, y={world_goal[1]:.4f}\n")
            f.write(f"Goal  (map cell): {GOAL_NODE}\n")
            f.write(f"Waypoints exported: {len(world_waypoints)}\n")
            f.write(f"Waypoints file: {WAYPOINT_OP_PATH}\n")
        print(f"Saved run summary to {summary_path}")

        plt.show()