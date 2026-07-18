import time
import os
from datetime import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathfinder import (
    MAP_IMAGE_PATH, OBSTACLE_THRESHOLD, SAFETY_RADIUS_CELLS, GRID_SIZE,
    get_live_webots_position, world_to_map, map_to_world, inflate_obstacles, 
    a_star_search, smooth_path
)

DYNAMIC_WAYPOINT_PATH = "/home/mayank-linux/Desktop/Projects/Custom-2D-Lidar-SLAM-MPC-Navigation-Stack/controllers/turtlebot3_keyboard/dynamic_waypoints.txt"

# Directory settings for saving verification graphs automatically
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

    ax.set_title('Click on the map to select a NEW GOAL position\n(Close window to let the robot plan to it)')
    plt.draw()
    print("Click on the window to select your target coordinates...")
    clicked = plt.ginput(1, timeout=0)

    if not clicked:
        plt.close(fig)
        return current_goal  # Keep current goal if window is closed directly

    gx, gy = int(round(clicked[0][0])), int(round(clicked[0][1]))

    if not (0 <= gx < width and 0 <= gy < height) or inflated_grid[gy, gx] != 0:
        print("Invalid click selection (out of bounds or inside safety wall buffer) -- maintaining previous goal.")
        plt.close(fig)
        return current_goal

    print(f"Goal updated at map index coordinates: ({gx}, {gy})")
    plt.close(fig)
    return (gx, gy)


def main():
    print("--- Starting Adaptive SLAM Plotting Planner Node ---")
    last_map_mtime = 0
    goal_node = (80, 80) # Initial cell coordinate assignment

    while True:
        # Check if map has been updated by the SLAM controller
        if os.path.exists(MAP_IMAGE_PATH):
            current_mtime = os.path.getmtime(MAP_IMAGE_PATH)
            
            # Runs planning phase when a file modification update is recorded
            if current_mtime > last_map_mtime:
                last_map_mtime = current_mtime
                print("\n[SLAM UPDATE] New map matrix detected! Recalculating trajectory...")
                
                # 1. Process updated map array data
                img = Image.open(MAP_IMAGE_PATH).convert("L")
                grid = (np.array(img) < OBSTACLE_THRESHOLD).astype(np.uint8)
                inflated = inflate_obstacles(grid, SAFETY_RADIUS_CELLS)
                
                # Construct fresh RGB display overlay map
                display_map = np.stack((np.array(img),) * 3, axis=-1)
                display_map[(inflated == 1) & (grid == 0)] = [255, 180, 180]
                
                # 2. Extract current real-time robot pose metrics
                rx, ry = get_live_webots_position()
                mx, my = world_to_map(rx, ry)
                start_node = (mx, (GRID_SIZE - 1) - my)
                
                # 3. Open prompt UI window to optionally select a different goal node
                goal_node = select_goal_interactively(display_map, inflated, start_node, goal_node)
                
                # 4. Global A* graph computation
                path = a_star_search(inflated, start_node, goal_node)
                if path:
                    path = smooth_path(inflated, path)
                    
                    # Atomic temporary write block to prevent file racing locks with the active supervisor thread
                    temp_path = DYNAMIC_WAYPOINT_PATH + ".tmp"
                    with open(temp_path, "w") as f:
                        for mx, my in path:
                            wx, wy = map_to_world(mx, (GRID_SIZE - 1) - my)
                            f.write(f'{wx:.4f},{wy:.4f}\n')
                    os.rename(temp_path, DYNAMIC_WAYPOINT_PATH)
                    print(f"Path updated successfully with {len(path)} waypoints.")
                    
                    # 5. Save verification plot snapshot automatically
                    plt.figure(figsize=(8, 8))
                    plt.imshow(display_map)
                    plt.plot([p[0] for p in path], [p[1] for p in path], color='cyan', linewidth=3, label="Planned Path")
                    plt.scatter(start_node[0], start_node[1], color='green', s=100, zorder=5, label='Live Start')
                    plt.scatter(goal_node[0], goal_node[1], color='red', s=100, zorder=5, label='Goal')
                    plt.title('Adaptive SLAM Tracking Frame')
                    plt.legend(loc='upper right')
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_filename = os.path.join(PLOT_LOG_DIR, f"slam_path_{timestamp}.png")
                    plt.savefig(save_filename, dpi=150, bbox_inches='tight')
                    plt.close() # Closes graph frame structure silently so execution is never hung up
                    print(f"Saved run graphic tracking history to: {save_filename}")
                else:
                    print("[Error] Planning failed: Path to target blocked by map elements.")
        
        time.sleep(1.0)


if __name__ == "__main__":
    main()