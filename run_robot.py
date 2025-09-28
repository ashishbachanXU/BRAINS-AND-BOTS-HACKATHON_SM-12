#!/usr/bin/env python3
"""
Combined executable for canvas detection and RRT path planning.

This script performs the following actions in sequence:
1.  Runs canvas detection to identify the start, goal, and obstacles.
2.  Repeats the canvas detection a second time.
3.  Uses the results from the second detection to:
    a. Plan a collision-free path using the RRT algorithm.
    b. Smooth and downsample the path into a series of waypoints.
    c. Send the waypoints as '/move' commands to a local server.
"""

# -------------------------
# Imports
# -------------------------
import requests
import base64
import cv2
import numpy as np
import json
import math
import random
import time
from typing import Tuple, List, Dict, Optional


# -------------------------
# Unified Configuration
# -------------------------
# --- Server ---
SERVER_BASE = "http://localhost:5001"

# --- Canvas Detection ---
MIN_COMPONENT_AREA = 80         # filter tiny blobs
MIN_COMPONENT_DIM = 6           # min width/height in pixels
GAUSSIAN_BLUR = (5, 5)
CLAHE_APPLY = False             # set True if images are very dark/low contrast
CLAHE_CLIP = 2.0
ROBOT_MIN_RADIUS = 8            # pixels — used to filter very small red blobs
# Color ranges (HSV).
BLACK_LOW = np.array([0, 0, 0])
BLACK_HIGH = np.array([180, 255, 60])
GREEN_LOW = np.array([36, 50, 50])
GREEN_HIGH = np.array([86, 255, 255])
RED1_LOW = np.array([0, 80, 40])
RED1_HIGH = np.array([10, 255, 255])
RED2_LOW = np.array([170, 80, 40])
RED2_HIGH = np.array([180, 255, 255])
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# --- RRT Path Planning ---
ROBOT_RADIUS = 9                # pixels (used to inflate obstacles)
RRT_MAX_ITER = 60000
RRT_STEP = 20.0                 # step size in pixels when extending the tree
GOAL_SAMPLE_RATE = 0.10         # probability of sampling the goal directly
GOAL_TOLERANCE = 20.0           # distance to goal to consider success
SMOOTHING_ITERS = 200
WAYPOINT_DIST_MIN = 18.0        # coalesce waypoints closer than this
SPEED_PIXELS_PER_SEC = 80.0     # used to compute wait time between /move commands
WAYPOINT_WAIT_MIN = 0.15        # seconds minimum wait between moves
WAYPOINT_WAIT_MAX = 3.0         # cap wait per waypoint

# --- Server Communication Behavior ---
RESET_BEFORE_RUN = False
PUSH_OBSTACLES_TO_SERVER = False
OBSTACLES_VISIBLE = False
PUSH_GOAL_TO_SERVER = False
POLL_GOAL_STATUS = False


# ==============================================================================
# SECTION 1: CANVAS DETECTION LOGIC (from detect_canvas.py)
# ==============================================================================

def b64_to_cv2_image(image_data_b64: str) -> np.ndarray:
    """Decodes a base64 string into a CV2 image array."""
    if image_data_b64.startswith("data:image"):
        image_data_b64 = image_data_b64.split(",", 1)[1]
    missing = len(image_data_b64) % 4
    if missing:
        image_data_b64 += "=" * (4 - missing)
    img_bytes = base64.b64decode(image_data_b64)
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode base64 image")
    return img

def enhance_contrast(img_bgr: np.ndarray) -> np.ndarray:
    """Applies CLAHE contrast enhancement if enabled."""
    if not CLAHE_APPLY:
        return img_bgr
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def cleanup_mask(mask: np.ndarray, kernel=None, open_iter=1, close_iter=2) -> np.ndarray:
    """Applies morphological operations to clean up a binary mask."""
    if kernel is None:
        kernel = KERNEL
    m = mask.copy()
    if open_iter > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    if close_iter > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    m = cv2.GaussianBlur(m, (3,3), 0)
    _, m = cv2.threshold(m, 128, 255, cv2.THRESH_BINARY)
    return m

def find_components(mask: np.ndarray, min_area=MIN_COMPONENT_AREA) -> List[Dict]:
    """Finds contours in a mask and returns them as a list of component dictionaries."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    comps = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area < min_area or w < MIN_COMPONENT_DIM or h < MIN_COMPONENT_DIM:
            continue
        cx = x + w//2
        cy = y + h//2
        comps.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": float(area), "center": (int(cx), int(cy))})
    return comps

def detect_from_cv_image(img_bgr: np.ndarray) -> Dict:
    """Main detection pipeline that processes an image and returns all found elements."""
    out = {}
    vis = img_bgr.copy()
    img_bgr = enhance_contrast(img_bgr)
    img_blur = cv2.GaussianBlur(img_bgr, GAUSSIAN_BLUR, 0)
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

    H, W = hsv.shape[:2]
    out['canvas_size'] = {"width": W, "height": H}

    # Black obstacles
    mask_black = cv2.inRange(hsv, BLACK_LOW, BLACK_HIGH)
    mask_black = cleanup_mask(mask_black, open_iter=1, close_iter=3)
    obstacles = find_components(mask_black, min_area=MIN_COMPONENT_AREA)
    out['obstacles'] = obstacles
    for obs in obstacles:
        x,y,w,h = obs['x'], obs['y'], obs['w'], obs['h']
        cv2.rectangle(vis, (x,y), (x+w, y+h), (0,0,255), 2)
        cv2.putText(vis, "obs", (x,y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)

    # Green goal
    mask_green = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
    mask_green = cleanup_mask(mask_green, open_iter=1, close_iter=2)
    goals = find_components(mask_green, min_area=MIN_COMPONENT_AREA)
    goal = goals[0] if goals else None
    out['goal'] = goal
    if goal:
        gx,gy,gw,gh = goal['x'], goal['y'], goal['w'], goal['h']
        cv2.rectangle(vis, (gx,gy), (gx+gw, gy+gh), (0,255,0), 2)
        cv2.putText(vis, "goal", (gx,gy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)

    # Red start
    mask_red = cv2.inRange(hsv, RED1_LOW, RED1_HIGH) | cv2.inRange(hsv, RED2_LOW, RED2_HIGH)
    mask_red = cleanup_mask(mask_red, open_iter=1, close_iter=2)
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    start = None
    for cnt in contours:
        (x,y), radius = cv2.minEnclosingCircle(cnt)
        area = cv2.contourArea(cnt)
        if radius >= ROBOT_MIN_RADIUS and area >= MIN_COMPONENT_AREA:
            start = {"center": (int(round(x)), int(round(y))), "radius": int(round(radius)), "area": float(area)}
            break
    out['start'] = start
    if start:
        cv2.circle(vis, start['center'], start['radius'], (255,0,0), 2)
        cv2.putText(vis, "start", (start['center'][0]-20, start['center'][1]-start['radius']-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    # Pixel statistics
    total_pixels = W * H
    black_pixels = int(np.count_nonzero(mask_black))
    green_pixels = int(np.count_nonzero(mask_green))
    red_pixels = int(np.count_nonzero(mask_red))
    free_pixels = total_pixels - (black_pixels + green_pixels + red_pixels)
    out['pixel_stats'] = {
        'total_pixels': int(total_pixels), 'black_pixels': black_pixels,
        'green_pixels': green_pixels, 'red_pixels': red_pixels,
        'free_pixels': int(free_pixels),
        'percent': {
            'black': round(black_pixels / total_pixels * 100, 2),
            'green': round(green_pixels / total_pixels * 100, 2),
            'red': round(red_pixels / total_pixels * 100, 2),
            'free': round(free_pixels / total_pixels * 100, 2),
        }
    }

    cv2.imwrite("detections.png", vis)
    out['annotated_image'] = "detections.png"
    return out

def analyze_image_from_base64(image_data_b64: str) -> Dict:
    """Wrapper to decode a base64 image and run detection."""
    img = b64_to_cv2_image(image_data_b64)
    return detect_from_cv_image(img)


# ==============================================================================
# SECTION 2: RRT AND SERVER LOGIC (from rrt_to_server.py)
# ==============================================================================

# --- Utility geometry helpers ---
def dist(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def point_in_rect(px: float, py: float, rx0: float, ry0: float, rx1: float, ry1: float) -> bool:
    return (rx0 <= px <= rx1) and (ry0 <= py <= ry1)

def seg_seg_intersect(a1, a2, b1, b2) -> bool:
    def orient(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
    def on_segment(p,q,r):
        return min(p[0],r[0]) <= q[0] <= max(p[0],r[0]) and min(p[1],r[1]) <= q[1] <= max(p[1],r[1])
    o1 = orient(a1,a2,b1); o2 = orient(a1,a2,b2)
    o3 = orient(b1,b2,a1); o4 = orient(b1,b2,a2)
    if o1*o2 < 0 and o3*o4 < 0: return True
    if abs(o1) < 1e-9 and on_segment(a1,b1,a2): return True
    if abs(o2) < 1e-9 and on_segment(a1,b2,a2): return True
    if abs(o3) < 1e-9 and on_segment(b1,a1,b2): return True
    if abs(o4) < 1e-9 and on_segment(b1,a2,b2): return True
    return False

def seg_rect_collision(a, b, rx0, ry0, rx1, ry1) -> bool:
    if point_in_rect(a[0], a[1], rx0, ry0, rx1, ry1) or point_in_rect(b[0], b[1], rx0, ry0, rx1, ry1):
        return True
    r1=(rx0,ry0); r2=(rx1,ry0); r3=(rx1,ry1); r4=(rx0,ry1)
    if seg_seg_intersect(a,b,r1,r2) or seg_seg_intersect(a,b,r2,r3) or \
       seg_seg_intersect(a,b,r3,r4) or seg_seg_intersect(a,b,r4,r1):
        return True
    return False

def segment_collides_obstacles(a: Tuple[float,float], b: Tuple[float,float],
                               obstacles: List[Dict], robot_radius: float) -> bool:
    """Checks if a line segment collides with any inflated obstacles."""
    for obs in obstacles:
        if all(k in obs for k in ("x","y","w","h")):
            ox, oy, ow, oh = float(obs['x']), float(obs['y']), float(obs['w']), float(obs['h'])
            # Expand rect by robot_radius for collision check
            rx0, ry0 = ox - robot_radius, oy - robot_radius
            rx1, ry1 = ox + ow + robot_radius, oy + oh + robot_radius
            if seg_rect_collision(a, b, rx0, ry0, rx1, ry1):
                return True
    return False

# --- RRT Planner Class ---
class RRTPlanner:
    def __init__(self, start: Tuple[float,float], goal: Tuple[float,float],
                 obstacles: List[Dict], canvas_w: int, canvas_h: int,
                 step_size: float = RRT_STEP,
                 max_iter: int = RRT_MAX_ITER,
                 goal_sample_rate: float = GOAL_SAMPLE_RATE,
                 goal_tolerance: float = GOAL_TOLERANCE,
                 robot_radius: float = ROBOT_RADIUS):
        self.start, self.goal = start, goal
        self.obstacles = obstacles
        self.W, self.H = canvas_w, canvas_h
        self.step_size = float(step_size)
        self.max_iter = int(max_iter)
        self.goal_sample_rate = float(goal_sample_rate)
        self.goal_tol = float(goal_tolerance)
        self.robot_radius = float(robot_radius)
        self.nodes = [start]
        self.parent = {0: None}

    def sample_free(self) -> Tuple[float,float]:
        return self.goal if random.random() < self.goal_sample_rate else \
               (random.uniform(0, self.W), random.uniform(0, self.H))

    def nearest_index(self, sample: Tuple[float,float]) -> int:
        return min(range(len(self.nodes)), key=lambda i: dist(self.nodes[i], sample))

    def steer(self, from_p: Tuple[float,float], to_p: Tuple[float,float]) -> Tuple[float,float]:
        d = dist(from_p, to_p)
        if d <= self.step_size:
            return to_p
        dx = (to_p[0] - from_p[0]) / d
        dy = (to_p[1] - from_p[1]) / d
        return (from_p[0] + dx * self.step_size, from_p[1] + dy * self.step_size)

    def plan(self) -> Optional[List[Tuple[float,float]]]:
        if segment_collides_obstacles(self.start, self.start, self.obstacles, self.robot_radius):
            print("[RRT] Start is in collision. Aborting.")
            return None
        if segment_collides_obstacles(self.goal, self.goal, self.obstacles, self.robot_radius):
            print("[RRT] Goal is in collision. Aborting.")
            return None

        for it in range(self.max_iter):
            sample = self.sample_free()
            nearest_i = self.nearest_index(sample)
            nearest = self.nodes[nearest_i]
            newp = self.steer(nearest, sample)

            if not segment_collides_obstacles(nearest, newp, self.obstacles, self.robot_radius):
                new_index = len(self.nodes)
                self.nodes.append(newp)
                self.parent[new_index] = nearest_i

                if dist(newp, self.goal) <= self.goal_tol and \
                   not segment_collides_obstacles(newp, self.goal, self.obstacles, self.robot_radius):
                    path = [self.goal, newp]
                    cur = new_index
                    while cur is not None:
                        path.append(self.nodes[cur])
                        cur = self.parent.get(cur)
                    path.reverse()
                    print(f"[RRT] Found path in {it} iterations. Path length: {len(path)}")
                    return path
        print("[RRT] No path found within iteration limit.")
        return None

    def smooth_path(self, path: List[Tuple[float,float]], iters: int = SMOOTHING_ITERS) -> List[Tuple[float,float]]:
        if not path or len(path) <= 2: return path
        p = path[:]
        for _ in range(iters):
            n = len(p)
            if n <= 2: break
            i = random.randint(0, n-2)
            j = random.randint(i+1, n-1)
            if j - i > 1 and not segment_collides_obstacles(p[i], p[j], self.obstacles, self.robot_radius):
                p = p[:i+1] + p[j:]
        return p

    def downsample(self, path: List[Tuple[float,float]], min_dist: float = WAYPOINT_DIST_MIN) -> List[Dict]:
        if not path: return []
        out = [path[0]]
        for p in path[1:]:
            if dist(out[-1], p) >= min_dist:
                out.append(p)
        if dist(out[-1], path[-1]) > 1e-3:
            out.append(path[-1])
        return [{"x": float(round(pt[0],2)), "y": float(round(pt[1],2))} for pt in out]

# --- Server communication helpers ---
def post_move(x: float, y: float):
    try:
        r = requests.post(f"{SERVER_BASE}/move", json={"x": float(x), "y": float(y)}, timeout=3)
        return r.ok, r.json() if r.headers.get('content-type','').startswith('application/json') else {}
    except Exception as e:
        return False, {"error": str(e)}

# --- Main RRT runner ---
def run_from_detections(detections: Dict):
    """Parses detections, runs RRT, and sends waypoints to the server."""
    try:
        canvas = detections['canvas_size']
        canvas_w = int(canvas['width'])
        canvas_h = int(canvas['height'])
        start_obj = detections['start']
        goal_obj = detections['goal']
        obstacles = detections.get('obstacles', [])
        
        if not start_obj or not goal_obj:
            print("[ERROR] 'start' or 'goal' missing from detections.")
            return {"status":"failed", "reason":"missing start or goal"}

        sx, sy = float(start_obj['center'][0]), float(start_obj['center'][1])
        gx = float(goal_obj['x'] + goal_obj['w'] / 2.0)
        gy = float(goal_obj['y'] + goal_obj['h'] / 2.0)
    except (KeyError, TypeError) as e:
        print(f"[ERROR] Could not parse detections dictionary: {e}")
        return {"status":"failed", "reason": "parsing_error"}
    
    print(f"[INFO] Canvas {canvas_w}x{canvas_h}, start=({sx:.1f},{sy:.1f}), goal=({gx:.1f},{gy:.1f}), obstacles={len(obstacles)}")

    planner = RRTPlanner(start=(sx,sy), goal=(gx,gy), obstacles=obstacles,
                         canvas_w=canvas_w, canvas_h=canvas_h)
    path = planner.plan()
    if path is None:
        print("[ERROR] RRT planning failed. Exiting.")
        return {"status":"failed", "reason":"no_path"}

    path_sm = planner.smooth_path(path)
    waypoints = planner.downsample(path_sm)
    print(f"[INFO] Original path: {len(path)} pts; Smoothed: {len(path_sm)} pts; Waypoints: {len(waypoints)}")

    for idx, wp in enumerate(waypoints):
        x, y = wp['x'], wp['y']
        print(f"[MOVE] waypoint {idx+1}/{len(waypoints)} -> ({x:.2f},{y:.2f})")
        ok, resp = post_move(x, y)
        print(f"  > Server response: {'OK' if ok else 'FAIL'} {resp}")
        
        if idx > 0:
            prev = (waypoints[idx-1]['x'], waypoints[idx-1]['y'])
            d = dist(prev, (x,y))
            wait_t = max(WAYPOINT_WAIT_MIN, min(d / SPEED_PIXELS_PER_SEC, WAYPOINT_WAIT_MAX))
            time.sleep(wait_t)

    print("[DONE] All waypoints sent.")
    return {"status":"done", "waypoints_sent": len(waypoints)}


# ==============================================================================
# SECTION 3: MAIN EXECUTION ORCHESTRATOR
# ==============================================================================

def perform_detection() -> Optional[Dict]:
    """
    Requests a canvas capture from the server, runs the detection pipeline,
    and returns the analysis dictionary.
    """
    capture_url = f"{SERVER_BASE}/capture"
    print(f"Requesting canvas capture from {capture_url}...")
    try:
        r = requests.get(capture_url, timeout=5)
        r.raise_for_status()
        res = r.json()
        if res.get("status") != "success":
            print(f"Error from server: {res}")
            return None
            
        print("Image received. Analyzing...")
        analysis = analyze_image_from_base64(res['image_data'])
        analysis['server_meta'] = {k: res.get(k) for k in ['timestamp', 'canvas_size', 'robot_position']}
        
        # Save JSON to file for debugging and simulation
        with open("detections.json", "w") as f:
            json.dump(analysis, f, indent=2)
        print(" Detection successful. Saved results to detections.json and detections.png")
        return analysis

    except requests.exceptions.RequestException as e:
        print(f" Error: Could not connect to the server at {SERVER_BASE}. Is it running?")
        print(f"   Details: {e}")
        return None
    except Exception as e:
        print(f" An unexpected error occurred during detection: {e}")
        return None


def main():
    """Main function to run the complete detection and planning pipeline."""
    print("--- Running First Detection ---")
    first_detection_results = perform_detection()
    if first_detection_results:
        print(json.dumps(first_detection_results['pixel_stats'], indent=2))
    else:
        print("First detection failed. Aborting.")
        return

    print("\n-------------------------------------\n")
    print("--- Running Second Detection ---")
    second_detection_results = perform_detection()
    if not second_detection_results:
        print("Second detection failed. Aborting.")
        return

    print("\n-------------------------------------\n")
    print("--- Starting RRT Path Planning and Execution ---")
    print("Using results from the second detection.")
    
    rrt_result = run_from_detections(second_detection_results)
    
    print("\n--- RRT Process Finished ---")
    print(json.dumps(rrt_result, indent=2))


if __name__ == "__main__":
    main()