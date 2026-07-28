import os
import cv2
import json
import csv
import time
import numpy as np
from datetime import datetime, timezone, timedelta

# ==========================================
# 1. UNDISTORTION & CAMERA MATH
# ==========================================

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

def process_and_undistort(frame, K_orig, D, xi, orig_size, fov_scale=0.4):
    img_h, img_w = frame.shape[:2]
    w, h = img_h, img_w  
    xi_vec = np.array([xi], dtype=np.float64)

    K_scaled = K_orig.copy()
    if orig_size:
        scale_factor = w / orig_size[0] 
        K_scaled = K_orig * scale_factor
        K_scaled[2, 2] = 1.0 
    
    K_new = np.eye(3)
    K_new[0, 0] = w * fov_scale 
    K_new[1, 1] = w * fov_scale 
    K_new[0, 2] = w / 2.0
    K_new[1, 2] = h / 2.0

    frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    undistorted = cv2.omnidir.undistortImage(
        frame_rotated, K_scaled, D, xi_vec,
        cv2.omnidir.RECTIFY_PERSPECTIVE,
        Knew=K_new,
        new_size=(w, h)
    )
    return undistorted, K_new, w, h


# ==========================================
# 2. REGISTRY & MAPPING CORE
# ==========================================

def build_session_registry(base_root_dir):
    """Scans the root directory to map out all video clips and their exact time windows."""
    print(f"\n[INFO] Scanning '{base_root_dir}' to build video registry...")
    registry = []
    
    for root_path, _, files in os.walk(base_root_dir):
        if "session_meta.json" in files:
            meta_path = os.path.join(root_path, "session_meta.json")
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    
                utc_str = meta.get("created_utc")
                if not utc_str: continue
                
                utc_dt = datetime.strptime(utc_str, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
                ist_start_dt = (utc_dt + timedelta(hours=5, minutes=30)).replace(tzinfo=None)
                
                vid_path = os.path.join(root_path, "LENS1", "video_lens1.mp4")
                if not os.path.exists(vid_path): continue
                
                cap = cv2.VideoCapture(vid_path)
                if not cap.isOpened(): continue
                fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()
                
                duration_sec = frames / fps
                ist_end_dt = ist_start_dt + timedelta(seconds=duration_sec)
                
                registry.append({
                    "folder_path": root_path,
                    "folder_name": os.path.basename(root_path),
                    "start_ist": ist_start_dt,
                    "end_ist": ist_end_dt,
                    "fps": fps
                })
            except Exception as e:
                print(f"[WARNING] Skipping {meta_path}: {e}")
                
    print(f"[SUCCESS] Mapped {len(registry)} valid video sessions.")
    return registry

def find_session_by_time(registry, target_ist_dt):
    """Returns the session that contains the target time."""
    for session in registry:
        if session["start_ist"] <= target_ist_dt <= session["end_ist"]:
            return session
    return None

def process_imu_gps_file(csv_path, registry, output_mapping_path):
    """Reads IMU-GPS, syncs it to videos, and creates the Master Mapping CSV."""
    print(f"\n[INFO] Processing GPS data and calculating frame mapping...")
    
    mapped_data = []
    unmapped_count = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Look specifically for 'local_timestamp'
        time_col = None
        for col in reader.fieldnames:
            if 'local_timestamp' in col.lower():
                time_col = col
                break
        
        if not time_col:
            print(f"[ERROR] Could not find 'local_timestamp' column. Found: {reader.fieldnames}")
            return None
            
        for row in reader:
            time_str = row[time_col].strip()
            if not time_str: continue
            
            try:
                ist_dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
                
            session = find_session_by_time(registry, ist_dt)
            if not session:
                unmapped_count += 1
                continue
                
            utc_dt = ist_dt - timedelta(hours=5, minutes=30)
            utc_dt = utc_dt.replace(tzinfo=timezone.utc)
            unix_time = utc_dt.timestamp()
            
            elapsed_sec = (ist_dt - session["start_ist"]).total_seconds()
            frame_no = int(elapsed_sec * session["fps"])
            
            mapped_data.append({
                "ist_time": ist_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "utc_time": utc_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "unix_time": f"{unix_time:.3f}",
                "source_folder": session["folder_name"],
                "folder_path": session["folder_path"],
                "playback_time_sec": f"{elapsed_sec:.3f}",
                "frame_no": frame_no
            })
            
    with open(output_mapping_path, 'w', newline='') as f:
        fieldnames = ["ist_time", "utc_time", "unix_time", "source_folder", "folder_path", "playback_time_sec", "frame_no"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mapped_data)
        
    print(f"[SUCCESS] Master Mapping generated: {output_mapping_path}")
    print(f" -> Successfully mapped {len(mapped_data)} rows.")
    if unmapped_count > 0:
        print(f" -> Note: {unmapped_count} GPS rows fell outside of recorded video times.")
        
    return mapped_data


# ==========================================
# 3. EXTRACTION ENGINE
# ==========================================

def run_extraction(folder_path, frame_no, target_label, omni_json, output_root, meta_info=None, fov_scale=0.4):
    """Extracts, undistorts, and saves the 6 lenses for a given frame."""
    print(f"\n[INFO] Starting Extraction for: {target_label} (Frame {frame_no})")
    
    if meta_info is None:
        meta_info = {}
    
    K_orig, D, xi, orig_size = load_omni_intrinsics(omni_json)
    
    safe_label = target_label.replace(":", "-").replace(" ", "_")
    output_dir = os.path.join(output_root, f"Extraction_{safe_label}")
    
    undistorted_dir = os.path.join(output_dir, "undistorted")
    intrinsics_dir = os.path.join(output_dir, "INTRINSICS")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(undistorted_dir, exist_ok=True)
    os.makedirs(intrinsics_dir, exist_ok=True)

    extracted_any = False
    
    start_time = time.perf_counter()
    
    for i in range(1, 7):
        video_path = os.path.join(folder_path, f"LENS{i}", f"video_lens{i}.mp4")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"  [!] LENS{i} video not found.")
            continue
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        
        if ret:
            undistorted, K_new, w, h = process_and_undistort(frame, K_orig, D, xi, orig_size, fov_scale)
            img_filename = f"undistorted_LENS{i}_{safe_label}_frame_{frame_no}.jpg"
            
            cv2.imwrite(os.path.join(undistorted_dir, img_filename), undistorted)
            
            pinhole_intr = {
                "K": K_new.tolist(), "D": [0.0]*5, "model": "pinhole_from_omni",
                "source_omni_json": os.path.basename(omni_json), "image_size": [w, h],
            }
            with open(os.path.join(intrinsics_dir, f"calibration_pinhole_lens{i}.json"), "w") as jf:
                json.dump(pinhole_intr, jf, indent=2)
                
            extracted_any = True
            print(f"  -> LENS{i} Extracted & Undistorted.")
        else:
            print(f"  [!] LENS{i} failed to read frame {frame_no} (End of video?)")
            
        cap.release()
        
    end_time = time.perf_counter()
    processing_duration = end_time - start_time
    
    if extracted_any:
        frame_data = {
            "local_time_ist": meta_info.get("ist_time", "Manual Override"),
            "unix_time": meta_info.get("unix_time", "N/A"),
            "frame_number": frame_no,
            "source_folder": os.path.basename(folder_path),
            "playback_time_sec": meta_info.get("playback_time_sec", "Manual Override"),
            "processing_time_sec": round(processing_duration, 4)
        }
        
        meta_filepath = os.path.join(undistorted_dir, "frame_data.json")
        with open(meta_filepath, "w") as f:
            json.dump(frame_data, f, indent=4)
            
        print(f"[SUCCESS] All extracted files saved to: {output_dir}")
        print(f"[INFO] 6-Lens Processing Time: {processing_duration:.2f} seconds")
    else:
        print(f"[ERROR] Extraction failed. No frames were retrieved.")


# ==========================================
# 4. INTERACTIVE TERMINAL LOOP
# ==========================================

def main():
    OMNI_JSON = r"C:\IITM\Results\FINAL_Intrinsics\calibration_omni.json"
    ROOT_OUTPUT = r"C:\IITM\DATA_RETRIEVAL_TASK"
    
    registry = []
    mapped_data = []
    base_dir = ""

    print("="*60)
    print(" 360 CAMERA DATA PROCESSING & EXTRACTION SUITE ")
    print("="*60)

    while True:
        print("\n--- MAIN MENU ---")
        print("1. Upload & Process IMU-GPS File (Generates Master Mapping)")
        print("2. Extract Frame from Video")
        print("3. Quit")
        choice = input("Select an option (1/2/3): ").strip()

        if choice == '1':
            base_dir = input("\nEnter Base Data Directory (e.g., C:\\data\\22nd jan): ").strip()
            csv_path = input("Enter path to IMU-GPS CSV file: ").strip()
            
            if not os.path.exists(csv_path):
                print("[ERROR] CSV file not found.")
                continue
                
            registry = build_session_registry(base_dir)
            mapping_output = os.path.join(base_dir, "Master_GPS_Frame_Mapping.csv")
            mapped_data = process_imu_gps_file(csv_path, registry, mapping_output)
            
            if mapped_data:
                print("\n[INFO] IMU data uploaded and mapped successfully.")
                print("Transitioning to Extraction Menu...")
                choice = '2' 

        if choice == '2':
            print("\n--- EXTRACTION MENU ---")
            print("A) Search by Timestamp (IST, UTC, or Unix)")
            print("B) Enter Direct Folder & Frame Number")
            sub_choice = input("Select an option (A/B): ").strip().upper()
            
            if sub_choice == 'A':
                if not mapped_data:
                    print("[!] No Master Mapping loaded. Please run Option 1 first to map timestamps.")
                    continue
                    
                query = input("\nEnter Timestamp (YYYY-MM-DD HH:MM:SS) or Unix epoch: ").strip()
                
                match = None
                for row in mapped_data:
                    if query in [row["ist_time"], row["utc_time"], row["unix_time"]]:
                        match = row
                        break
                        
                if match:
                    run_extraction(
                        folder_path=match["folder_path"], 
                        frame_no=int(match["frame_no"]), 
                        target_label=match["ist_time"], 
                        omni_json=OMNI_JSON, 
                        output_root=ROOT_OUTPUT,
                        meta_info=match 
                    )
                else:
                    print("[ERROR] Timestamp not found in the Master Mapping CSV.")
                    
            elif sub_choice == 'B':
                direct_folder = input("\nEnter the full path to the video session folder: ").strip()
                
                if not os.path.exists(os.path.join(direct_folder, "session_meta.json")):
                    print("[ERROR] Invalid folder. 'session_meta.json' not found.")
                    continue
                    
                frame_input = input("Enter Frame Number: ").strip()
                if not frame_input.isdigit():
                    print("[ERROR] Frame number must be an integer.")
                    continue
                
                manual_meta = {
                    "source_folder": os.path.basename(direct_folder)
                }
                    
                label = f"{os.path.basename(direct_folder)}_Frame_{frame_input}"
                run_extraction(
                    folder_path=direct_folder, 
                    frame_no=int(frame_input), 
                    target_label=label, 
                    omni_json=OMNI_JSON, 
                    output_root=ROOT_OUTPUT,
                    meta_info=manual_meta
                )
            else:
                print("[ERROR] Invalid selection.")

        elif choice == '3':
            print("\nExiting Data Retrieval Suite. Goodbye!")
            break
            
        elif choice != '1' and choice != '2':
            print("[ERROR] Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()