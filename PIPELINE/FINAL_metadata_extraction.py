import os
import json
import glob
import cv2
import logging
import shutil
import csv
import itertools
from datetime import datetime, timedelta
import pytz

# --- CONFIGURATION ---
ROOT_DIR = r"F:\MUMMAS DATA COLLECTION\MUMMAS AQI BUILDING" 
IST = pytz.timezone('Asia/Kolkata')

SENSOR_FOLDER_ALIASES = {
    "aqi": ["AQI", "AQIDATA"],
    "imu-gps": ["IMU", "IMUGPS"],
    "lidar1": ["LIDAR", "LIDAR1", "L1"],
    "lidar2": ["LIDAR", "LIDAR2", "L2"],
    "princeton_gps": ["PRINCETONGPS", "PRINCETON", "PGPS", "GPS"],
    "weather": ["WEATHER", "WEATHERDATA"]
}
CAMERA_ALIASES = ["CAMERA", "CAMERAS", "CAM"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def normalize_name(name):
    return name.replace(" ", "").replace("_", "").replace("-", "").upper()

def resolve_folder(parent_dir, aliases):
    if not os.path.exists(parent_dir): return None
    normalized_aliases = [normalize_name(a) for a in aliases]
    for item in os.listdir(parent_dir):
        item_path = os.path.join(parent_dir, item)
        if os.path.isdir(item_path):
            if normalize_name(item) in normalized_aliases:
                return item_path
    return None

def find_sensor_file(folder_path, timestamp_str):
    if not folder_path or not os.path.exists(folder_path): return "N/A"
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if timestamp_str in f: return os.path.join(root, f)
    return "N/A"

def get_video_info(video_path):
    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        return "N/A", 0, 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        return "N/A", 0, 0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    duration = int(frames / fps) if fps > 0 else 0
    return frames, duration, round(fps, 2)

def fetch_imu_start_data(imu_path, target_utc):
    if imu_path == "N/A" or not target_utc: return None, None, None
    clean_target = str(target_utc).replace("T", " ").replace("Z", "").strip()
    try:
        with open(imu_path, 'r', encoding='utf-8') as f:
            header = next((line for line in f if 'timestamp_utc' in line or 'timestamp_local' in line), None)
            if not header: return None, None, None
            reader = csv.DictReader(itertools.chain([header], f))
            for row in reader:
                if clean_target in str(row.get('timestamp_utc', '')).replace("T", " ").replace("Z", "").strip():
                    return row.get('timestamp_local'), row.get('timestamp_utc'), row.get('t_unix')
    except Exception as e: logging.error(f"IMU Read Error: {e}")
    return None, None, None

def process_date_folder(date_folder_path, date_folder_name):
    metadata = {
        "date": date_folder_name, "no. of trips": 0, "no. of runs": 0, "trips": {}, 
        "timestamp_metadata": os.path.join(date_folder_path, f"timestamps_metadata_{date_folder_name}.csv")
    }
    csv_rows, csv_headers = [], ["run_id", "ist_timestamp", "utc_timestamp", "unix_timestamp", "playback_time"]
    
    trip_folders = [d for d in os.listdir(date_folder_path) 
                    if os.path.isdir(os.path.join(date_folder_path, d)) and "TRIP" in d.upper()] or ["."]
    
    metadata["no. of trips"] = len(trip_folders)
    
    for trip_idx, trip_name in enumerate(trip_folders, 1):
        trip_path = os.path.join(date_folder_path, trip_name) if trip_name != "." else date_folder_path
        resolved_dirs = {k: resolve_folder(trip_path, v) for k, v in SENSOR_FOLDER_ALIASES.items()}
        camera_dir = resolve_folder(trip_path, CAMERA_ALIASES)
        
        if not camera_dir: continue
            
        run_folders = [d for d in os.listdir(camera_dir) if os.path.isdir(os.path.join(camera_dir, d)) and d.startswith("run_")]
        
        for run_name in run_folders:
            metadata["no. of runs"] += 1
            run_path = os.path.join(camera_dir, run_name)
            ts = f"{run_name.split('_')[1][:4]}-{run_name.split('_')[1][4:6]}-{run_name.split('_')[1][6:8]}_{run_name.split('_')[2][:2]}-{run_name.split('_')[2][2:4]}-{run_name.split('_')[2][4:6]}"
            
            meta_path = os.path.join(run_path, "session_meta.json")
            meta = {}
            if os.path.exists(meta_path) and os.path.getsize(meta_path) > 0:
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f: meta = json.load(f)
                except: logging.error(f"Corrupt JSON: {meta_path}")
            
            created_utc = meta.get("created_utc")
            sensors = {k: find_sensor_file(resolved_dirs[k], ts) for k in SENSOR_FOLDER_ALIASES}
            
            # --- UPDATED: Camera availability and FPS logic ---
            camera_data, lens1_dur, found_lenses = {}, 0, 0
            for i in range(1, 7):
                f_path = glob.glob(os.path.join(run_path, f"LENS{i}", "*.mp4"))
                if f_path:
                    fc, dur, fps = get_video_info(f_path[0])
                    camera_data[f"lens{i}_frame_count"] = fc
                    camera_data[f"lens{i}_fps"] = fps
                    found_lenses += 1
                    if i == 1: lens1_dur = dur
                else:
                    camera_data[f"lens{i}_frame_count"] = "N/A"
                    camera_data[f"lens{i}_fps"] = "N/A"

            if found_lenses == 6: camera_data["available"] = "available"
            elif found_lenses > 0: camera_data["available"] = "partially available"
            else: camera_data["available"] = "N/A"

            ist_s, utc_s, unix_s = fetch_imu_start_data(sensors["imu-gps"], created_utc)
            if not ist_s and created_utc:
                try:
                    dt = datetime.fromisoformat(str(created_utc).replace("Z", "+00:00"))
                    ist_s, utc_s, unix_s = dt.astimezone(IST).strftime('%Y-%m-%d %H:%M:%S'), dt.strftime('%Y-%m-%d %H:%M:%S'), dt.timestamp()
                except: ist_s = None
            
            ist_s_safe = str(ist_s) if ist_s else "N/A"

            metadata["trips"].setdefault(str(trip_idx), []).append({
                "run_id": run_name, "start_time_ist": ist_s_safe, 
                "end_time_ist": (datetime.strptime(ist_s_safe, '%Y-%m-%d %H:%M:%S') + timedelta(seconds=lens1_dur)).strftime('%Y-%m-%d %H:%M:%S') if ist_s_safe != "N/A" else "N/A",
                **sensors, "camera": camera_data
            })
            
            # CSV generation...
            if ist_s_safe != "N/A":
                dt_ist = datetime.strptime(ist_s_safe, '%Y-%m-%d %H:%M:%S')
                dt_utc = datetime.strptime(utc_s, '%Y-%m-%d %H:%M:%S')
                for t in range(lens1_dur):
                    csv_rows.append({"run_id": run_name, "ist_timestamp": (dt_ist+timedelta(seconds=t)).strftime('%Y-%m-%d %H:%M:%S'), "utc_timestamp": (dt_utc+timedelta(seconds=t)).strftime('%Y-%m-%d %H:%M:%S'), "unix_timestamp": round(float(unix_s)+t, 3), "playback_time": t})
            else:
                csv_rows.append({"run_id": run_name, "ist_timestamp": "N/A", "utc_timestamp": "N/A", "unix_timestamp": "N/A", "playback_time": "N/A"})

    with open(os.path.join(date_folder_path, "metadata.json"), 'w') as f: json.dump(metadata, f, indent=4)
    with open(metadata["timestamp_metadata"], 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers); writer.writeheader(); writer.writerows(csv_rows)

if __name__ == "__main__":
    for d in os.listdir(ROOT_DIR):
        if d.isdigit() and len(d) == 8: process_date_folder(os.path.join(ROOT_DIR, d), d)