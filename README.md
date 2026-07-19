# TurtleBot3 2D Occupancy-Grid Mapping & A* Navigation (Webots)

![me](https://github.com/Mayank-Chandra/Custom-2D-Lidar-SLAM-MPC-Navigation-Stack/blob/main/MPC_formation_12-ezgif.com-video-to-gif-converter.gif)

A single-robot navigation stack built in Webots: manual LIDAR-based occupancy-grid
mapping, A* global path planning over the built map, and closed-loop waypoint
following with a reactive LIDAR safety layer for unmapped obstacles.

> **Scope note:** This project uses **ground-truth pose** from Webots' `Supervisor`
> API (`getPosition()` / `getOrientation()`) rather than estimating pose from noisy
> odometry or sensor fusion. That means it is **occupancy-grid mapping with known
> localization**, not full SLAM — there's no odometry drift correction, particle
> filter/EKF pose estimation, scan matching, or loop closure.
> the mapping, planning, and control components
> below are implemented and working as described.

## Basic Explanations

1. **Manual mapping** — drive the robot around the arena with the keyboard while a
   log-odds occupancy grid is built from LIDAR returns (Bresenham ray tracing marks
   free cells along each beam, occupied cells at the hit point). Save the grid as a
   PNG map at any time.
2. **Global path planning** — load the saved map, inflate obstacles by a safety
   radius, click a goal on the map, and compute a smoothed A* path from the robot's
   current position to the goal.
3. **Waypoint following** — a separate controller drives the robot along the
   exported path using proportional heading/distance control, with a reactive LIDAR
   check that stops and pivots away from anything unmapped that gets too close.
4. **Live telemetry bridge** — the driving controller broadcasts the robot's
   ground-truth position over a local TCP socket so the planner can plot "where the
   robot actually is" on top of the map in real time.

## Architecture

```
┌─────────────────────┐        TCP socket         ┌──────────────────────┐
│ turtlebot3_keyboard │  (ground-truth position)  │   pathfinder.py /    │
│  (manual mapping)   │ ─────────────────────────▶│  dynamics_pathfinder │
│  - drive    WASD    │                           │  - A* global planner │
│  - log-odds grid    │                           │  - click-to-set goal │
│  - saves webots_map │                           │  - exports waypoints │
└──────────┬──────────┘                           └───────────┬──────────┘
           │ webots_map.png                                   │ waypoints.txt
           ▼                                                  ▼
                    (loaded by pathfinder.py)       ┌──────────────────────────┐
                                                    │ turtlebot3_path_follower │
                                                    │  / reactive_path_follower│
                                                    │  - waypoint tracking     │
                                                    │  - LIDAR safety override │
                                                    └──────────────────────────┘
```

## Files

| File | Role |
|---|---|
| `turtlebot3_keyboard.py` | Manual WASD-driven mapping controller. Builds the log-odds occupancy grid from LIDAR via Bresenham ray tracing, renders a live minimap to the robot's Webots `Display` device, and saves the map as `webots_map.png` on demand. |
| `pathfinder.py` | Core planning module: A* search with corner-cutting prevention, line-of-sight path smoothing (string-pulling), world↔map coordinate conversion, obstacle inflation, and an interactive click-to-select goal UI. Also runnable standalone for one-shot planning. |
| `dynamics_pathfinder.py` | Live-loop version of the planner — repeatedly re-fetches the robot's current position, lets you pick a new goal, plans, and exports waypoints without needing to restart the script. |
| `turtlebot3_path_follower.py` / `reactive_path_follower.py` | Drives the robot along the exported waypoints using proportional (P) heading and distance control, differential-drive wheel speed mixing, and a front-arc LIDAR check that overrides normal tracking to pivot away from anything unexpectedly close. |

## Working

**Mapping (log-odds occupancy grid):** each LIDAR beam is ray-traced cell-by-cell
from the sensor to its hit point using Bresenham's line algorithm. Cells along the
ray get a small negative log-odds update (evidence of free space); the cell at the
hit point gets a larger positive update (evidence of an obstacle). Log-odds are
converted to a probability and then to grayscale for display/export.

**Planning (A*):** the saved map is thresholded into a binary occupancy grid,
obstacles are inflated by a configurable safety radius (so the planner won't hug
walls), and A* searches the inflated grid with 8-connected movement (diagonal moves
are blocked if either adjacent orthogonal cell is occupied, to prevent the path
clipping through corners). The raw grid path is then smoothed with a line-of-sight
string-pulling pass so the robot doesn't have to follow a jagged grid-stepped path.

**Control (waypoint following):** at each timestep the controller computes distance
and heading error to the current target waypoint, applies proportional gains
(clipped to velocity/angular limits), and converts the resulting linear/angular
velocity into differential wheel speeds. A front-facing LIDAR arc is checked every
step; if anything enters the safety distance, tracking is overridden by a stop-and
-pivot maneuver until the arc clears.

## Possible extensions

- Replace ground-truth pose with real odometry + a particle filter or EKF for actual
  localization under uncertainty.
- Add a local planner (DWA or VFH) between the global path and motor commands for
  smoother, non-blocking obstacle avoidance.
- Loop closure detection for map correction over long runs.

## Requirements

- Webots R2025a (or compatible)
- Python 3, `numpy`, `matplotlib`, `Pillow`

## Setup

1. Open `worlds/Arena.wbt` in Webots and press play.
2. Change the directory for the waypoints and LiDAR map.
3. Set the robot's controller to `turtlebot3_keyboard.py` and drive it around the
   arena with W/A/S/D to build the map. Press `P` or hold `P` to save `webots_map.png`.
4. Switch the controller to `turtlebot3_path_follower.py` (or
   `reactive_path_follower.py`).
5. Run `python3 dynamics_pathfinder.py`` (or `pathfinder.py` for a one-shot run),
   click a goal on the displayed map, and close the window to let the robot drive
   there.
