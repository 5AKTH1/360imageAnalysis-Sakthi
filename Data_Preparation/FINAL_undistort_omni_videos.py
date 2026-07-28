import os
import json
import numpy as np
import cv2
from typing import Tuple

# --- Keep your original loading logic ---
def load_omni_intrinsics(json_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Intrinsics JSON not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    K = np.asarray(data["K"], dtype=np.float64)
    D = np.asarray(data["D"], dtype=np.float64).reshape(-1, 1)
    xi = float(data["xi"])
    return K, D, xi

def process_all_lenses(base_dir, intrinsics_file, fov_scale=0.5):
    # Load parameters once
    K, D, xi = load_omni_intrinsics(intrinsics_file)
    xi_vec = np.array([xi], dtype=np.float64)
    
    # Identify the central Intrinsics folder
    intrinsics_dir = os.path.dirname(intrinsics_file)

    # List of your 6 processed folders
    lenses = [f"lens {i}" for i in range(1, 7)]

    for lens_folder in lenses:
        lens_path = os.path.join(base_dir, lens_folder)
        video_files = [f for f in os.listdir(lens_path) if f.startswith("processed_") and f.endswith(".mp4")]

        for v_file in video_files:
            input_video = os.path.join(lens_path, v_file)
            output_video = os.path.join(lens_path, v_file.replace("processed_", "undistorted_"))

            cap = cv2.VideoCapture(input_video)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # --- CALCULATE NEW INTRINSICS ---
            K_new = K.copy()
            K_new[0, 0] *= fov_scale
            K_new[1, 1] *= fov_scale
            K_new[0, 2] = w / 2.0
            K_new[1, 2] = h / 2.0

            # --- SAVE JSON TO CENTRAL INTRINSICS FOLDER ---
            # Standardize name to 'lens1', 'lens2', etc.
            lens_id = lens_folder.replace(" ", "").lower() 
            pinhole_json_name = f"calibration_pinhole_{lens_id}.json"
            pinhole_json_path = os.path.join(intrinsics_dir, pinhole_json_name)
            
            pinhole_intr = {
                "K": K_new.tolist(),
                "D": [0.0, 0.0, 0.0, 0.0, 0.0], # Sigifies zero distortion after rectification
                "model": "pinhole_from_omni",
                "source_omni_json": os.path.basename(intrinsics_file),
                "image_size": [int(w), int(h)],
            }
            
            with open(pinhole_json_path, "w") as f:
                json.dump(pinhole_intr, f, indent=2)
            print(f"[WRITE] Generated JSON in Intrinsics folder: {pinhole_json_path}")

            # --- VIDEO UNDISTORTION ---
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

            print(f"Undistorting {v_file}...")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                undistorted = cv2.omnidir.undistortImage(
                    frame, K, D, xi_vec,
                    cv2.omnidir.RECTIFY_PERSPECTIVE,
                    Knew=K_new,
                    new_size=(w, h)
                )

                if undistorted.dtype != np.uint8:
                    undistorted = cv2.normalize(undistorted, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                out.write(undistorted)

            cap.release()
            out.release()
            print(f"Saved Video: {output_video}")

if __name__ == "__main__":
    CLIPS_PATH = r"C:\IITM\CAMERA_Short_Clips"
    # The JSONs will be saved in this folder:
    OMNI_JSON = r"C:\IITM\Vehicle_detection_360\Intrinsics\calibration_omni.json"
    process_all_lenses(CLIPS_PATH, OMNI_JSON)



#for original lens
'''
import os
import json
import numpy as np
import cv2
from typing import Tuple

def load_omni_intrinsics(json_path: str):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Intrinsics JSON not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    
    K = np.asarray(data["K"], dtype=np.float64)
    D = np.asarray(data["D"], dtype=np.float64).reshape(-1, 1)
    xi = float(data["xi"])
    
    raw_size = data.get("image_size", None)
    orig_size = None
    if raw_size:
        if isinstance(raw_size, dict):
            w = raw_size.get("width") or raw_size.get("w")
            h = raw_size.get("height") or raw_size.get("h")
            if w and h: orig_size = [float(w), float(h)]
        elif isinstance(raw_size, (list, tuple)) and len(raw_size) >= 2:
            orig_size = [float(raw_size[0]), float(raw_size[1])]
            
    return K, D, xi, orig_size

def undistort_specific_video(video_path, intrinsics_file, lens_name, fov_scale=0.4, duration_sec=120):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return

    # 1. Setup Video Capture
    cap = cv2.VideoCapture(video_path)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if vid_w == 0 or vid_h == 0:
        print("Error: Could not read video dimensions.")
        return

    # --- ROTATION DIMENSION SWAP ---
    w = vid_h  # New Width is old Height
    h = vid_w  # New Height is old Width
    
    if duration_sec:
        max_frames = int(fps * duration_sec)
    else:
        max_frames = total_frames

    print(f"[INFO] Processing {lens_name.upper()} | Input: {vid_w}x{vid_h} -> Rotated: {w}x{h}")

    # 2. Load Calibration
    K_orig, D, xi, orig_size = load_omni_intrinsics(intrinsics_file)
    xi_vec = np.array([xi], dtype=np.float64)

    # 3. Scale K based on the ROTATED Resolution
    K_scaled = K_orig.copy()
    
    if orig_size:
        orig_w, orig_h = orig_size
        scale_factor = w / orig_w 
        K_scaled = K_orig * scale_factor
        K_scaled[2, 2] = 1.0 
    
    # 4. Calculate Output Pinhole Matrix (K_new)
    K_new = np.eye(3)
    K_new[0, 0] = w * fov_scale 
    K_new[1, 1] = w * fov_scale 
    K_new[0, 2] = w / 2.0
    K_new[1, 2] = h / 2.0

    # 5. Save Pinhole JSON
    lens_id = lens_name.replace(" ", "").lower()
    pinhole_json_name = f"calibration_pinhole_{lens_id}.json"
    # Save to the same folder as video
    pinhole_json_path = os.path.join(os.path.dirname(video_path), pinhole_json_name)
    
    pinhole_intr = {
        "K": K_new.tolist(),
        "D": [0.0]*5,
        "model": "pinhole_from_omni",
        "source_omni_json": os.path.basename(intrinsics_file),
        "image_size": [w, h],
    }
    
    with open(pinhole_json_path, "w") as f:
        json.dump(pinhole_intr, f, indent=2)
    print(f"[CONFIG] Saved JSON to: {pinhole_json_path}")

    # 6. Setup Video Writer
    base_folder = os.path.dirname(video_path)
    filename = os.path.basename(video_path)
    
    if "processed_" in filename:
        out_name = filename.replace("processed_", "undistorted_")
    else:
        out_name = "undistorted_" + filename
        
    output_path = os.path.join(base_folder, out_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"[START] Rotated Undistortion -> {out_name}")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx >= max_frames:
            print(f"\n[INFO] Reached {duration_sec}s limit. Stopping.")
            break

        # --- STEP 1: ROTATE -90 (Counter Clockwise) ---
        frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # --- STEP 2: UNDISTORT ---
        undistorted = cv2.omnidir.undistortImage(
            frame_rotated, K_scaled, D, xi_vec,
            cv2.omnidir.RECTIFY_PERSPECTIVE,
            Knew=K_new,
            new_size=(w, h)
        )

        out.write(undistorted)
        
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"       Processed {frame_idx}/{max_frames} frames...", end='\r')

    cap.release()
    out.release()
    print(f"\n[DONE] Saved to: {output_path}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    OMNI_JSON = r"C:\IITM\Vehicle_detection_360\Intrinsics\calibration_omni.json"
    BASE_VIDEO_PATH = r"D:\Camera\17012026"

    # --- LOOP 1 TO 6 ---
    for i in range(1, 7):
        # Construct filename dynamically: origin_1..., origin_2..., etc.
        video_file_name = f"origin_{i}_20260117_103923_lrv.mp4"
        video_full_path = os.path.join(BASE_VIDEO_PATH, video_file_name)
        
        lens_name = f"lens {i}"
        
        # Check if file exists before processing
        if os.path.exists(video_full_path):
            undistort_specific_video(video_full_path, OMNI_JSON, lens_name, fov_scale=0.4, duration_sec=120)
        else:
            print(f"[SKIP] File not found: {video_full_path}")
'''