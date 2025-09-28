# 🧭 Obstacle Robot Game — SM12 (Canvas Detection + RRT Planning)

## 📌 Overview
This project combines **computer vision** and **path planning** to navigate a robot in an environment where obstacle positions are **not given in advance**.

The program works in two main stages:
1. **Canvas Detection** → Captures the canvas from a local server and uses OpenCV to detect start, goal, and obstacles.  
2. **Path Planning & Execution** → Runs the **Rapidly-Exploring Random Tree (RRT)** algorithm to find a path, smooth it into waypoints, and send `/move` commands to the server.  

---

## 🔹 Features
- Automatic **obstacle, goal, and robot detection** from canvas images  
- **Base64 image decoding** and OpenCV-based color segmentation  
  - Red = Start  
  - Green = Goal  
  - Black = Obstacles  
- Robust **morphological cleanup** of masks for accurate detection  
- **RRT planner** with:
  - Collision checking against inflated obstacles  
  - Path smoothing and downsampling  
  - Configurable parameters like step size, max iterations, goal tolerance  
- **Server communication**:
  - `/capture` → fetches images  
  - `/move` → sends robot waypoint commands  

---

## ⚙️ Parameters
- `step_size` → 10  
- `max_iter` → 10,000  
- `goal_tolerance` → 15  
- Robot radius considered during collision checks  

---

## 🛠️ Tech Stack
- Python 3.11+  
- OpenCV (image segmentation and morphology)  
- NumPy (geometry and math)  
- Requests (server communication)  

---

## ▶️ How to Run
```bash
pip install -r requirements.txt

python run_robot.py
