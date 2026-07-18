import time
import os
import socket
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import core structural logic directly from your working pathfinder.py file
from pathfinder import (
    MAP_IMAGE_PATH, OBSTACLE_THRESHOLD, SAFETY_RADIUS_CELLS, GRID_SIZE,
    get_live_webots_position, world_to_map, map_to_world, 
    inflate_obstacles, a_star_search, smooth_path
)

DYNAMIC_WAYPOINT_PATH = "/home/mayank-linux/Desktop/Projects/Custom-2D-Lidar-SLAM-MPC-Navigation-Stack/controllers/turtlebot3_keyboard/dynamic_waypoints.txt"

# --- NEW CONFIGURATION: Auto-saving Directory Path ---
PROJECT_DIR = os.path.dirname(DYNAMIC_WAYPOINT_PATH)
PLOT_LOG_DIR = os.path.join(PROJECT_DIR, "saved_plots")
os.makedirs(PLOT_LOG_DIR, exist_ok=True)


def select_goal_interactively(display_map, inflated_grid, start_node, current_goal=None):
    """Shows the live map and handles clicking selection for a new navigation goal."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(display_map)
    ax.scatter(start_node[0], start_node[1], color='green', s=100, zorder=5, label='Live Start')

    if current_goal:
        ax.scatter(current_goal[0], current_goal[1], color='red', s=100, zorder=5, label='Current Goal')

    ax.legend(loc='upper right')
    height, width = inflated_grid.shape

    while True:
        ax.set_title('Click on the map to select a NEW GOAL position\n(Close the window to let the robot drive there)')
        plt.draw()
        print("Click on the window to select your target coordinates...")
        clicked = plt.ginput(1, timeout=0)

        if not clicked:
            plt.close(fig)
            return current_goal  # Return existing target if closed without clicking

        gx, gy = int(round(clicked[0][0])), int(round(clicked[0][1]))

        if not (0 <= gx < width and 0 <= gy < height):
            print("Click out of bounds! Try again.")
            continue

        if inflated_grid[gy, gx] != 0:
            print("Target falls inside a safety wall inflation zone! Pick a clear area.")
            ax.scatter(gx, gy, color='orange', marker='x', s=80, zorder=5)
            continue

        print(f"New target updated at map index coordinates: ({gx}, {gy})")
        plt.close(fig)
        return (gx, gy)


def main():
    print("--- Starting Dynamic Plotting Planner Node ---")
    try:
        img = Image.open(MAP_IMAGE_PATH).convert("L")
        map_array = np.array(img)
    except FileNotFoundError:
        print(f"Error: Could not find map tracking image layout file at {MAP_IMAGE_PATH}")
        return

    raw_grid = np.zeros_like(map_array, dtype=np.uint8)
    raw_grid[map_array < OBSTACLE_THRESHOLD] = 1
    inflated_grid = inflate_obstacles(raw_grid, SAFETY_RADIUS_CELLS)

    display_map = np.stack((map_array,) * 3, axis=-1)
    display_map[(inflated_grid == 1) & (raw_grid == 0)] = [255, 180, 180]

    # Initial default target goal node index matching tracking configuration
    goal_node = (80, 80) 

    while True:
        # 1. Fetch live metrics from active supervisor communication thread
        robot_x, robot_y = get_live_webots_position()

        # Same convention as pathfinder.py: no axis negation, just the
        # standard image-row vs world-Y flip.
        raw_mx, raw_my = world_to_map(robot_x, robot_y)
        start_node = (raw_mx, (GRID_SIZE - 1) - raw_my)

        # 2. Launch selection overlay window UI
        goal_node = select_goal_interactively(display_map, inflated_grid, start_node, goal_node)

        if goal_node is None:
            print("System closure or exit requested.")
            break

        # 3. Calculate and smooth the A* trajectory layout path vector
        path = a_star_search(inflated_grid, start_node, goal_node)

        if path:
            path = smooth_path(inflated_grid, path)
            world_waypoints = []
            for (mx, my) in path:
                real_my = (GRID_SIZE - 1) - my
                wx, wy = map_to_world(mx, real_my)
                world_waypoints.append((wx, wy))

            # Atomic swap export dump to avoid read/write racing locks
            temp_path = DYNAMIC_WAYPOINT_PATH + ".tmp"
            with open(temp_path, "w") as f:
                for wx, wy in world_waypoints:
                    f.write(f'{wx:.4f},{wy:.4f}\n')
            os.rename(temp_path, DYNAMIC_WAYPOINT_PATH)
            print(f"Exported {len(world_waypoints)} waypoints to follower controller stack.")

            # 4. Generate the tracking trace visual execution graph window
            fig_track = plt.figure(figsize=(8, 8))
            plt.imshow(display_map)
            plt.plot([p[0] for p in path], [p[1] for p in path], color='cyan', linewidth=3, label="Active Track Trajectory")
            plt.scatter(start_node[0], start_node[1], color='green', s=100, zorder=5, label='Live Start')
            plt.scatter(goal_node[0], goal_node[1], color='red', s=100, zorder=5, label='Target Destination')
            plt.title('Execution Track Window\n(Close this window to choose another location)')
            plt.legend(loc='upper right')

            # --- NEW ADDITION: Automated Background Plot Logging ---
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename = os.path.join(PLOT_LOG_DIR, f"dynamic_path_{timestamp}.png")
            plt.savefig(save_filename, dpi=150, bbox_inches='tight')
            print(f"Saved live trajectory visualization snapshot to: {save_filename}")

            plt.show() # Execution blocks here while the robot runs. Close to choose a new target point.
        else:
            print("Planning failed: Target path blocked by inflation safety matrix boundaries.")
            time.sleep(1.0)


if __name__ == "__main__":
    main()