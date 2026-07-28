360-Degree Camera Data Retrieval & AI Processing Pipeline
📖 Overview
This pipeline is an end-to-end Python framework designed to extract, synchronize, and analyze omnidirectional (6-lens) video data collected from moving vehicles. It bridges high-frequency IMU/GPS sensor logs with 30 FPS video feeds, performs mathematically rigorous lens undistortion, feature-based SIFT stitching, and runs cascading YOLO-based AI models for vehicle and pedestrian detection.

⚙️ Prerequisites & Setup
1. Install Dependencies
Ensure you have Python 3.8+ installed. You will need the following libraries:

Bash
pip install numpy opencv-contrib-python ultralytics supervision
(Note: opencv-contrib-python is specifically required for the cv2.omnidir fisheye undistortion module).

2. Configure Hardcoded Paths
Before running the script for the first time, open the Python file, scroll to the main() function at the bottom, and ensure these 4 paths match your local machine:

OMNI_JSON: Path to your camera's calibration_omni.json file.

ROOT_OUTPUT: The master folder where all processed data will be saved (e.g., C:\IITM\DATA_RETRIEVAL_TASK).

YOLO_VEHICLE_MODEL: Path to your trained vehicle .pt model.

YOLO_PEDESTRIAN_MODEL: Path to your pedestrian .pt model (e.g., yolo11m.pt).

🚀 How to Use the Pipeline
Run the script via your terminal or IDE:

Bash
python data_pipeline.py
You will be greeted by an interactive dashboard with 7 options.

Option 1: Upload CSV & Generate Master Mapping
When to use: Run this first whenever you are analyzing a new hard drive of raw data.
What it does: It ingests your high-frequency IMU/GPS .csv file, strips the milliseconds to prevent duplicates, and maps every exact second of physical time to a specific video folder and frame number. It saves a Master_Mapping.csv to your ROOT_OUTPUT directory.

Input 1: Base Directory (e.g., D:\MUMMAS DATA COLLECTION\02022026)

Input 2: IMU CSV Path (e.g., D:\data\imu_all_topics.csv)

Note: Upon restarting the script later, this Master Mapping auto-loads instantly!

🎯 Target Selection (For Options 2 through 6)
Whenever you select an extraction or processing option, the script will ask how you want to find your target frame. You have two choices:

Path A (Search by Timestamp): The recommended method. Paste an exact time from your IMU log.

Example Inputs: 2026-02-02 09:36:59 (IST/UTC) or 1770005572.000 (Unix).

Path B (Manual Override): Use this if you don't have GPS data and just want to pull a frame directly from a video folder.

Input 1: Folder Name (e.g., 22012026091535 or VID_001)

Input 2: Choose Frame Number (e.g., 450) OR Playback Time (e.g., 15.5 seconds).

Option 2: Extract Undistorted Lenses
What it does: Finds your target frame, reads all 6 raw .mp4 lenses, and mathematically flattens the spherical fisheye distortion into standard planar images.
Outputs: Generates a folder named Pipeline_YYYY-MM-DD_HH-MM-SS_F<frame> containing an undistorted/ folder with 6 .jpg images and localized pinhole intrinsic JSONs.

Option 3: Extract & Stitch Panorama
What it does: Performs Option 2 (Extraction), and then automatically passes the 6 flattened images into the SIFT Stitcher. It matches microscopic keypoints in the overlapping edges, adjusts camera yaw/pitch, and blends them into a 360-degree equirectangular panorama.
Outputs: Saves Final_Static_Stitch.jpg and stitched_data.json (containing the stitching errors and final angles).

Option 4: Extract, Stitch, & Detect Vehicles
What it does: Performs Options 2 + 3, and then runs the panoramic image through the custom YOLO Vehicle model. It applies an Ego-Polygon mask (to hide the camera car's hood) and uses Dynamic Thresholding (lower confidence required for far-away horizon objects, higher confidence for close objects).
Outputs: Saves Final_Static_Stitch_detected.jpg (with bounding boxes) and a .json containing exact traffic counts and coordinates.

Option 5: Extract, Stitch, Detect Vehicles & Pedestrians
What it does: The ultimate single-frame pipeline. Performs Options 2, 3, 4, and then runs the YOLO Pedestrian model. It utilizes a Spatial Overlap Filter: if a detected human's feet fall inside a vehicle's bounding box (meaning they are a passenger inside a car), they are aggressively deleted from the pedestrian count.
Outputs: Saves Final_Static_Stitch_pedestrians.jpg and updates the JSON metadata.

Option 6: Temporal Burst (Future Vehicle Dynamics)
When to use: Use this when you need time-series data for velocity estimation, trajectory prediction, or optical flow.
What it does: Instead of pulling a single frame, it asks for a temporal offset x (e.g., 15 frames). It then extracts, stitches, and processes a burst of 3 frames: [Center - x], [Center], and [Center + x].
Outputs: Creates a Master Burst Folder containing a master temporal_metadata.json ledger, alongside the individual pipeline folders for all 3 frames.

Option 7: Quit
Safely exits the CLI dashboard and clears the terminal.

🧠 Advanced System Features (Under the Hood)
The pipeline executes several advanced quality-control measures silently in the background:

Dynamic Sharpness Scout: Cars hit bumps, causing motion blur. Before extracting a requested frame, the pipeline opens the forward-facing lens and scans a 1-second window. It calculates the Variance of the Laplacian and automatically shifts your request to the mathematically sharpest frame available.

Math-Based Consistency Lock: If the Sharpness Scout shifts your frame from 450 to 455 during Option 2, and you come back tomorrow to run Option 4 on the same timestamp, the script calculates the exact local time, scans your hard drive, recognizes the anchor, and permanently locks onto 455 to prevent duplicate data generation.

Intelligent File Caching: If you ask the script to perform Option 5 on a frame you already processed yesterday, it will ask Overwrite? [y/n]. If you type n, it instantly skips extraction and stitching, loading the existing panorama from your hard drive directly into the AI modules in milliseconds.