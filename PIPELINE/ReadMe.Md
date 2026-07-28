# 360 Camera Data Retrieval & AI Processing Hub 🎥🚙

Welcome! This application is designed to take raw data from a 6-lens 360-degree camera, sync it perfectly with sensor data (IMU/GPS), stitch the 6 video feeds together into a massive panorama, and use AI to automatically count vehicles and pedestrians.

If you have never used this tool before, this guide will walk you through exactly how to set up your files, organize your data, and run your first extraction.

## 🛠️ Step 1: Preparing Your System
Before you run the app, make sure your computer has the basic software required to run Python and process videos:
* **Python 3.8+** installed on your system.
* **FFmpeg** installed and added to your system's PATH (this is the engine the app uses to quickly trim video clips).
* **Required Python Libraries:** Open your terminal or command prompt and run this command to install the core libraries, plus the advanced slicing and tracking engines:
```bash
pip install opencv-python numpy pandas ultralytics supervision torch sahi deep-sort-realtime
```

## 📁 Step 2: Required AI & Calibration Files
Before running the app, you must have specific configuration files downloaded to your computer. These files contain the complex math and AI "brains" required to flatten the videos and detect objects.

If you open the Python script and scroll to the very bottom, you will see a `main()` function with file paths. You must update these paths in the code to point to where these files are saved on your computer:

```python
def main():
    OMNI_JSON = r"C:\Your\Path\To\calibration_omni.json"
    YOLO_VEHICLE_MODEL = r"C:\Your\Path\To\vehicle_model_best.pt"
    YOLO_PEDESTRIAN_MODEL = r"C:\Your\Path\To\yolo11m.pt"

class ClipVehicleDetector:
    def process_video():
        tracker=r"C:\Your\Path\To\FINAL_custom_track.yaml" # (Around line 1575)
```

* **The Omnidirectional Calibration Files:** The raw videos from the 6 lenses look like circular "fish-eyes." The script uses a master `calibration_omni.json` to perfectly stretch and flatten them. **NEW:** You must also place the 4 lens-specific calibration files (`lens1_calibration_omni.json`, `lens3...`, `lens4...`, `lens6...`) in this *exact same directory*. 
    * *(Note: Calibration files for Lens 2 and Lens 5 are intentionally omitted. The app contains advanced fallback logic to automatically borrow math from Lenses 1 and 6 to cover these blind spots).*
* **The YOLO Vehicle Detection Model (`best.pt`):** This is the AI trained specifically to recognize 6 classes of vehicles (Car, Auto, Bus, Truck, Motorbike) from the perspective of a 360-degree camera.
* **The YOLO Pedestrian Detection Model (`yolo11m.pt`):** This AI runs a secondary pass over the video specifically to track humans.
* **DeepSORT Tracker:** The app dynamically downloads a lightweight mobilenet ReID model on its first run to handle appearance-based tracking. Ensure you have an active internet connection the very first time you process a video.

## 🗂️ Step 3: Organizing Your Data Drive
For the app's automated search engine to work, your external hard drive containing the raw camera data must follow this exact folder structure:

```text
[HARD_DRIVE_ROOT]\               <-- e.g., D:\
├── 01022026\                    <-- Date folders (Format: DDMMYYYY or MMDDYYYY)
│   └── run_20260201_090111\     <-- Session folder
│       ├── session_meta.json    <-- Required metadata file
│       ├── LENS1\video_lens1.mp4
│       └── ... (Through LENS6)
└── All date all other sensors\  
    └── imu\imu_all_topics.csv   <-- IMU sensor data
```

## 🚀 Step 4: Starting the App
Run the script from your terminal:
```bash
python data_retrieval_hub.py
```
When the app starts, it will ask you for two paths:
1.  **Master Output Directory:** Where do you want the final stitched images/videos and Excel files saved? (e.g., `C:\Results`)
2.  **Hard Drive Root:** Where is your raw camera data located? (e.g., `D:\`)

## 🎯 Step 5: Choosing How to Find Your Data (Main Menu)
Once the app is running, you will see a Main Menu with three different ways to retrieve your data.

* **Option 1: Batch Process via CSV/Excel**
    Use this if you have a spreadsheet with hundreds or thousands of timestamps you want to process automatically.
    * *Input Format:* Provide the path to a `.csv` or `.xlsx` file.
    * *File Requirements:* Your spreadsheet must have a column named `timestamp` (e.g., `2026-01-28 14:36:18`).
    * *Optional:* If you want to extract video clips instead of static images, add a column named `duration` and put the length in seconds (e.g., `5.0`).
* **Option 2: Single Process: By Timestamp**
    Use this if you want to look up one specific moment in time. The app will search your entire hard drive to find the matching video frame.
    * *Example Inputs:* Standard (`2026-01-02 09:19:45`), ISO Format (`2026-01-02T09:19:45Z`), Unix (`1769917185.0`)
* **Option 3: Single Process: By Folder Path & Time/Frame**
    Use this if you already know exactly which folder the video is in, and you want to bypass the automated search.
    * *Input Format 1:* Provide the exact path to the run folder: `D:\01022026\run_20260201_090111_8757`
    * *Input Format 2:* The app will ask if you want to use a specific [f]rame number or playback [s]econd. Type `f` and enter `33392`, or type `s` and enter `12.5`.

## ⚙️ Step 6: Choosing the Processing Level (Pipeline Menu)
After you provide your input, the app will ask what you want it to do with that data.

**First, it will ask:**
`Process as: 1) Image Frame  2) Video Clip`
*(Note: If you are using a Batch CSV with a "duration" column, it skips this question and automatically chooses Video).*

**Next, choose how deep the AI goes:**
1.  **Extract Undistorted Lenses:** Just flattens the fish-eye effect from the 6 lenses and saves them.
2.  **Extract & Stitch Panorama:** Flattens the lenses and uses AI to seamlessly stitch them into one massive 360-degree flat image/video.
3.  **Extract, Stitch, & Detect Vehicles:** Does all the above, plus runs YOLO AI to draw boxes around Cars, Autos, Buses, Trucks, and Motorbikes.
4.  **Extract, Stitch, Detect Vehicles & Pedestrians:** Does all the above, plus runs a highly filtered human-tracking AI to count true pedestrians.
5.  **Temporal Burst (Images Only):** Extracts a frame, plus a few frames before and after it, so you can see how fast objects are moving.

### 💾 The "Save Media" Prompt
If you are doing a Batch Process (Option 1), the app will ask:
`Do you want to save the media (.mp4/.jpg) and .json files to disk? [y/n]`
* **If you press 'y':** Every stitched image, video, and AI file will be saved permanently to your Output folder.
* **If you press 'n':** The app will run the AI, update your CSV spreadsheet with the vehicle/pedestrian counts, and then automatically delete the heavy media files. This allows you to process massive datasets just to get the data counts without filling up your hard drive.
*(Note: If an existing file is found on your computer during processing, the app will ask if you want to overwrite it. If you don't type anything for 15 seconds, it will automatically say "yes" and keep going).*

## 📊 Step 7: Viewing Your Results
Once the app finishes, navigate to your Master Output Directory. Here is what you will find:
* **Your Updated Spreadsheet (`[name]_processed.csv`):** If you ran a batch process, a new copy of your CSV is generated. It will have an `available` column alongside new columns for CAR, BUS, PEDESTRIAN, etc., filled with the exact counts the AI found.
* **Master Mapping (`Master_Mapping.csv`):** A master log the app builds in the background linking exact sensor times to specific video frame numbers.
* **Pipeline Folders:** If you chose to save media, you will see folders containing:
    * `/undistorted`: Your flattened lens videos/images.
    * `Final_Static_Stitch.jpg / .mp4`: The raw stitched panorama.
    * `Final_Static_Stitch_detected.jpg / .mp4`: The final panorama with AI tracking boxes. Each pedestrian is assigned a persistent tracking ID color.
    * `*.json` files: Raw text data containing the exact pixel coordinates, confidence scores, and calculated pixel-speeds of every detected object.

## 🧠 Step 8: Under the Hood (Advanced Logic)
This pipeline uses several advanced optimizations to ensure high performance and accuracy:

* **Date-Routed Smart Stitching:** The pipeline automatically reads the timestamp of your footage. For any data captured *after March 1, 2026*, the app routes to a lightning-fast "Optimized Stitcher" that uses hardcoded, mathematically perfected zoom (1.25x) and orientation values—completely bypassing heavy calculations. For older footage, it seamlessly routes to a "Legacy Stitcher" that utilizes SIFT feature-matching to build the panorama.
* **SAHI (Slicing Aided Hyper Inference):** Because the 360 panoramas are massive, standard YOLO models cannot see distant pedestrians. The app slices the image into smaller overlapping grids (832x832), analyzes them individually, and stitches the detections back together.
* **DeepSORT ReID Tracking:** Standard trackers fail when a pedestrian walks behind a pole or crosses a camera stitching seam. This app uses DeepSORT to extract the "visual signature" of a pedestrian (clothing, colors) to remember them even if they disappear from the frame for up to 5 seconds.
* **Rider vs. Pedestrian Filtering:** To prevent counting people riding on motorcycles or sitting inside cars as pedestrians, the app generates a dynamic "danger zone" padding around every detected vehicle. Any human overlapping this zone is instantly discarded.
* **Biological Geometry & Kinematic Filtering:** The app calculates the aspect ratio and physical pixel width of every human detection. It uses Optical Flow to estimate camera movement and ensures that large "humans" are actually moving organically. This prevents the AI from falsely counting movie posters, billboards, and wall paintings as pedestrians.