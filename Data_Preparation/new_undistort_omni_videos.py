import os
import json
import numpy as np
import cv2
import glob
from typing import Tuple

def load_omni_intrinsics(json_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Intrinsics JSON not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    K = np.asarray(data["K"], dtype=np.float64)
    D = np.asarray(data["D"], dtype=np.float64).reshape(-1, 1)
    xi = float(data["xi"])
    return K, D, xi

def process_all_lenses(base_dir, intrinsics_dir, fov_scale=0.5):
    # Iterate through origin_0 to origin_5
    for i in range(6):
        # Look for files matching the pattern in the flat directory
        search_pattern = os.path.join(base_dir, f"origin_{i}_*.mp4")
        video_files = glob.glob(search_pattern)

        for input_video in video_files:
            v_file_name = os.path.basename(input_video)
            output_video = os.path.join(base_dir, f"undistorted_{v_file_name}")

            # Map the 0-indexed filename to physical lens 1-6
            lens_num = i + 1
            
            # --- FALLBACK LOGIC FOR MISSING CALIBRATIONS ---
            if lens_num == 2:
                json_num = 1
                print(f"Warning: Missing calibration for origin_1 (Lens 2). Falling back to Lens 1.")
            elif lens_num == 5:
                json_num = 6
                print(f"Warning: Missing calibration for origin_4 (Lens 5). Falling back to Lens 6.")
            else:
                json_num = lens_num

            # Construct the correct JSON filename
            json_filename = f"lens{json_num}_calibration_omni.json"
            intrinsics_file = os.path.join(intrinsics_dir, json_filename)
            
            # Load parameters specific to this lens
            K, D, xi = load_omni_intrinsics(intrinsics_file)
            xi_vec = np.array([xi], dtype=np.float64)

            cap = cv2.VideoCapture(input_video)
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # --- GEOMETRY UPDATE: ROTATION IMPLICATIONS ---
            # Rotating -90 degrees (counter-clockwise) swaps width and height
            rot_w = orig_h
            rot_h = orig_w

            # --- CALCULATE NEW INTRINSICS ---
            # We use the updated rotated dimensions to find the new optical center
            K_new = K.copy()
            K_new[0, 0] *= fov_scale
            K_new[1, 1] *= fov_scale
            K_new[0, 2] = rot_w / 2.0
            K_new[1, 2] = rot_h / 2.0

            # --- SAVE PHOLE JSON TO CENTRAL INTRINSICS FOLDER ---
            pinhole_json_name = f"calibration_pinhole_lens{lens_num}.json"
            pinhole_json_path = os.path.join(intrinsics_dir, pinhole_json_name)
            
            pinhole_intr = {
                "K": K_new.tolist(),
                "D": [0.0, 0.0, 0.0, 0.0, 0.0], 
                "model": "pinhole_from_omni",
                "source_omni_json": json_filename, 
                "image_size": [int(rot_w), int(rot_h)], # Save rotated dimensions
            }
            
            with open(pinhole_json_path, "w") as f:
                json.dump(pinhole_intr, f, indent=2)
            print(f"[WRITE] Generated JSON: {pinhole_json_name}")

            # --- VIDEO UNDISTORTION ---
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Initialize VideoWriter with rotated dimensions
            out = cv2.VideoWriter(output_video, fourcc, fps, (rot_w, rot_h))

            print(f"Processing {v_file_name} using {json_filename}...")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # 1. ROTATE THE FRAME -90 DEGREES (Counter-Clockwise)
                rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # 2. UNDISTORT THE ROTATED FRAME
                undistorted = cv2.omnidir.undistortImage(
                    rotated_frame, K, D, xi_vec,
                    cv2.omnidir.RECTIFY_PERSPECTIVE,
                    Knew=K_new,
                    new_size=(rot_w, rot_h)
                )

                if undistorted.dtype != np.uint8:
                    undistorted = cv2.normalize(undistorted, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                out.write(undistorted)

            cap.release()
            out.release()
            print(f"Saved Video: {output_video}\n")

if __name__ == "__main__":
    # Updated paths based on your new directory structure
    CLIPS_PATH = r"C:\IITM\CAMERA_Short_Clips\March'26 Clips"
    INTRINSICS_DIR = r"C:\IITM\IMU-GPS" 
    
    process_all_lenses(CLIPS_PATH, INTRINSICS_DIR)