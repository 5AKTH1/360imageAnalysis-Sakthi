import os
msvc_path = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64" # UPDATE THIS TO YOUR EXACT PATH
if os.path.exists(msvc_path):
    os.environ["PATH"] += os.pathsep + msvc_path
import cv2
import json
import csv
import time
import numpy as np
import subprocess
import shutil
from datetime import datetime, timezone, timedelta
import traceback
from collections import defaultdict, Counter
import select
import sys
import threading
import concurrent.futures
import queue
from multiprocessing import Manager
from tqdm import tqdm
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from deep_sort_realtime.deepsort_tracker import DeepSort

# Dataframes used for CSV Batch Processing
try:
    import pandas as pd
except ImportError:
    print("[WARNING] 'pandas' library not found. CSV Batch Processing will fail.")

# Optional import: gracefully handle YOLO missing
try:
    from ultralytics import YOLO
except ImportError:
    print("[WARNING] 'ultralytics' library not found. AI Options will fail.")

# Optional import: gracefully handle supervision missing
try:
    import supervision as sv
    import torch
except ImportError:
    print("[WARNING] 'supervision' or 'torch' not found. Tracking will fail.")

# ==========================================
# GPU ACCELERATION & SIFT CONFIGURATION
# ==========================================
try:
    import cupy as cp
    _GPU = cp.cuda.runtime.getDeviceCount() > 0
    if _GPU:
        print(f"[GPU] CuPy detected — using GPU (device 0: {cp.cuda.Device(0).use()})")
        
        # --- HOTFIX: BYPASS CUDA 12.1 vs NEW VISUAL STUDIO VERSION BLOCKS ---
        import cupy.cuda.compiler
        _orig_compile = cupy.cuda.compiler.compile_using_nvcc
        
        def _patched_compile(source, options=(), *args, **kwargs):
            options = list(options)
            # 1. Tell NVIDIA to ignore the new Microsoft compiler
            if '-allow-unsupported-compiler' not in options:
                options.append('-allow-unsupported-compiler')
            # 2. Tell Microsoft to ignore the older NVIDIA CUDA toolkit
            if '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH' not in options:
                options.append('-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH')
                
            return _orig_compile(source, tuple(options), *args, **kwargs)
            
        cupy.cuda.compiler.compile_using_nvcc = _patched_compile
        # -------------------------------------------------------------------
        
except Exception:
    cp = None
    _GPU = False

# PySIFT — GPU-resident SIFT (DSP-SIFT + RootSIFT); falls back to cv2.SIFT
try:
    from pysift import PySIFT
    _PYSIFT_AVAILABLE = True
    print("[PySIFT] GPU SIFT loaded (DSP-SIFT + RootSIFT)")
except Exception as _pysift_err:
    _PYSIFT_AVAILABLE = False
    print(f"[PySIFT] Unavailable ({_pysift_err}), using OpenCV SIFT fallback")

# ==========================================
# 0. UI/UX DASHBOARD HELPERS
# ==========================================
if os.name == 'nt':
    os.system('') 
    import msvcrt 

class UI:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    @staticmethod
    def clear(): 
        os.system('cls' if os.name == 'nt' else 'clear')

    @staticmethod
    def header(title, base_dir=None, out_dir=None):
        UI.clear()
        print(f"{UI.MAGENTA}{UI.BOLD}{'='*65}{UI.RESET}")
        print(f"{UI.CYAN}{UI.BOLD}{title.center(65)}{UI.RESET}")
        print(f"{UI.MAGENTA}{UI.BOLD}{'='*65}{UI.RESET}")
        print(f"{UI.YELLOW}📂 Hard Drive Root : {UI.RESET}{base_dir if base_dir else 'Not Set'}")
        print(f"{UI.YELLOW}💾 Master Output   : {UI.RESET}{out_dir if out_dir else 'Not Set'}")
        print(f"{UI.MAGENTA}{'-'*65}{UI.RESET}")

    @staticmethod
    def info(msg): 
        print(f"{UI.CYAN}[INFO]{UI.RESET} {msg}")
        
    @staticmethod
    def success(msg): 
        print(f"{UI.GREEN}{UI.BOLD}[SUCCESS]{UI.RESET} {msg}")
        
    @staticmethod
    def warn(msg): 
        print(f"{UI.YELLOW}[WARNING]{UI.RESET} {msg}")
        
    @staticmethod
    def error(msg): 
        print(f"{UI.RED}{UI.BOLD}[ERROR]{UI.RESET} {msg}")
        
    @staticmethod
    def input(msg): 
        return input(f"{UI.BOLD}{msg}{UI.RESET}").strip()
        
    @staticmethod
    def input_dir(msg, create_if_missing=False):
        while True:
            path = UI.input(msg)
            if os.path.isdir(path):
                return path
            if create_if_missing:
                try:
                    os.makedirs(path, exist_ok=True)
                    return path
                except Exception as e:
                    UI.error(f"Failed to create directory: {e}")
            UI.error("Directory does not exist. Please provide a valid directory path.")

    @staticmethod
    def input_file(msg):
        while True:
            path = UI.input(msg)
            if os.path.isfile(path):
                return path
            UI.error("File does not exist. Please provide a valid file path.")
        
    @staticmethod
    def pause(): 
        print(f"\n{UI.CYAN}{'-'*65}{UI.RESET}")
        input(f"{UI.BOLD}Press Enter to return to the menu...{UI.RESET}")

    @staticmethod
    def ask_yes_no(msg, timeout=15):
        print(f"{UI.YELLOW}{UI.BOLD}? {msg} [y/n] (Auto-Yes in {timeout}s): {UI.RESET}", end="", flush=True)
        start_time = time.time()
        ans = ""
        
        if os.name == 'nt':
            while True:
                if msvcrt.kbhit():
                    char = msvcrt.getwche()
                    if char in ('\r', '\n'):
                        print()
                        break
                    ans += char
                if time.time() - start_time > timeout:
                    print(f"\n{UI.MAGENTA}Timeout reached. Auto-selecting: YES{UI.RESET}")
                    return True
                time.sleep(0.05)
        else:            
            while True:
                i, o, e = select.select([sys.stdin], [], [], 0.1)
                if i:
                    ans = sys.stdin.readline().strip()
                    break
                if time.time() - start_time > timeout:
                    print(f"\n{UI.MAGENTA}Timeout reached. Auto-selecting: YES{UI.RESET}")
                    return True

        ans = ans.strip().lower()
        if ans in ['n', 'no']: return False
        return True


# ==========================================
# 1. CORE CAMERA MATH (Shared)
# ==========================================

def load_omni_intrinsics(json_path: str):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Intrinsics JSON not found at: {json_path}")
        
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"Intrinsics JSON is corrupted: {json_path}")
        
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

def get_intrinsics_for_lens(lens_num, omni_json_path, is_optimized):
    base_dir = os.path.dirname(omni_json_path)
    json_filename = os.path.basename(omni_json_path)
    
    if is_optimized:
        json_num = lens_num
        if lens_num == 2:
            json_num = 1
            print(f"   [Warning] Missing calibration for Lens 2. Falling back to Lens 1.")
        elif lens_num == 5:
            json_num = 6
            print(f"   [Warning] Missing calibration for Lens 5. Falling back to Lens 6.")
            
        opt_filename = f"lens{json_num}_calibration_omni.json"
        opt_path = os.path.join(base_dir, opt_filename)
        
        if os.path.exists(opt_path):
            K, D, xi, orig_size = load_omni_intrinsics(opt_path)
            return K, D, xi, orig_size, opt_filename
        else:
            UI.warn(f"Missing {opt_filename} in {base_dir}. Falling back to default {json_filename}")
            
    K, D, xi, orig_size = load_omni_intrinsics(omni_json_path)
    return K, D, xi, orig_size, json_filename

def process_and_undistort(frame, K_orig, D, xi, orig_size, fov_scale=0.4, downscale=0.5):
    try:
        if downscale != 1.0:
            new_w = int(frame.shape[1] * downscale)
            new_h = int(frame.shape[0] * downscale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
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
        
        # Switched to initUndistortRectifyMap for sharper INTER_CUBIC interpolation
        map1, map2 = cv2.omnidir.initUndistortRectifyMap(
            K_scaled, D, xi_vec, np.eye(3), K_new,
            (w, h), cv2.CV_32FC1, cv2.omnidir.RECTIFY_PERSPECTIVE
        )
        undistorted = cv2.remap(frame_rotated, map1, map2, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return undistorted, K_new, w, h
    except cv2.error as e:
        return None, None, None, None


# ==========================================
# 2. ASSET DISCOVERY & REGISTRY
# ==========================================
def parse_target_timestamp(ts_str):
    """Dynamically detects and parses IST, UTC, and Unix timestamp strings to standard IST."""
    ts_str = str(ts_str).strip()
    
    # Try standard IST
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        pass
        
    # Try Unix
    try:
        unix_val = float(ts_str)
        # Convert unix to UTC then to IST
        utc_dt = datetime.fromtimestamp(unix_val, timezone.utc)
        ist_dt = utc_dt + timedelta(hours=5, minutes=30)
        return ist_dt.replace(tzinfo=None)
    except ValueError:
        pass
        
    # Try typical UTC ISO format
    try:
        clean_str = ts_str.replace("T", " ").replace("Z", "")
        utc_dt = datetime.strptime(clean_str, "%Y-%m-%d %H:%M:%S")
        return utc_dt + timedelta(hours=5, minutes=30)
    except ValueError:
        pass

    return None

class AssetDiscovery:
    @staticmethod
    def get_csv_bounds(csv_path):
        """Ultra-fast O(1) CSV reader. Jumps to the end of the file to find bounds instantly."""
        try:
            first_time_str = None
            last_time_str = None
            time_col_idx = -1
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip() and not line.strip().startswith('#'):
                        header = next(csv.reader([line.strip()]))
                        for i, col in enumerate(header):
                            c_name = col.lower().strip()
                            if 'timestamp_local' in c_name or 'local_timestamp' in c_name or c_name == 'timestamp':
                                time_col_idx = i
                                break
                        break
                
                if time_col_idx == -1: return None, None
                
                for line in f:
                    if line.strip() and not line.strip().startswith('#'):
                        cols = next(csv.reader([line.strip()]))
                        if len(cols) > time_col_idx:
                            first_time_str = cols[time_col_idx].strip().replace("T", " ").split('.')[0]
                        break
                        
            if not first_time_str: return None, None
            
            # Jump straight to the end of the file for the last row
            with open(csv_path, 'rb') as f:
                f.seek(0, 2) 
                file_size = f.tell()
                chunk_size = min(file_size, 2048) 
                f.seek(file_size - chunk_size)
                last_chunk = f.read().decode('utf-8', errors='ignore').splitlines()
                
                for line in reversed(last_chunk):
                    if line.strip() and not line.strip().startswith('#'):
                        cols = next(csv.reader([line.strip()]))
                        if len(cols) > time_col_idx:
                            last_time_str = cols[time_col_idx].strip().replace("T", " ").split('.')[0]
                            break
                            
            if first_time_str and last_time_str:
                try:
                    start_dt = datetime.strptime(first_time_str, "%Y-%m-%d %H:%M:%S")
                    end_dt = datetime.strptime(last_time_str, "%Y-%m-%d %H:%M:%S")
                    return start_dt, end_dt
                except ValueError:
                    return None, None
        except Exception:
            return None, None
        return None, None

    @staticmethod
    def search_for_timestamp(root_drive, target_dt, verbose=True):
        target_csv_date = target_dt.strftime("%Y-%m-%d")   
        swapped_csv_date = f"{target_dt.year}-{target_dt.day:02d}-{target_dt.month:02d}"

        search_dirs = [
            os.path.join(root_drive, target_dt.strftime("%d%m%Y")), 
            os.path.join(root_drive, target_dt.strftime("%m%d%Y")), 
            os.path.join(root_drive, "All date all other sensors"),
            os.path.join(root_drive, "All other All dates")
        ]

        # 1. Search for IMU-GPS CSV (Best Match Logic)
        if verbose: UI.info("Scanning for corresponding IMU-GPS CSV...")
        best_csv = None
        min_csv_diff = float('inf')
        
        for s_dir in search_dirs:
            if not os.path.exists(s_dir): continue
            for root, dirs, files in os.walk(s_dir):
                for file in files:
                    if file.endswith(".csv"):
                        if target_csv_date in file or swapped_csv_date in file:
                            potential_csv = os.path.join(root, file)
                            start_dt, end_dt = AssetDiscovery.get_csv_bounds(potential_csv)
                            
                            if start_dt and end_dt:
                                # 1. Perfect Internal Match (Target is inside the CSV)
                                if start_dt <= target_dt <= end_dt:
                                    best_csv = potential_csv
                                    min_csv_diff = 0
                                    break
                                # 2. Closest Border Match (Fallback)
                                else:
                                    diff = min(abs((start_dt - target_dt).total_seconds()), abs((end_dt - target_dt).total_seconds()))
                                    if diff < min_csv_diff and diff <= 15 * 60:
                                        min_csv_diff = diff
                                        best_csv = potential_csv
                if min_csv_diff == 0: break
            if min_csv_diff == 0: break

        if not best_csv:
            if verbose: UI.error("imu-gps file not available for this timestamp.")
            return None, None, None

        # 2. Search for JSON / Video Session (Best Match Logic)
        if verbose: UI.info("Scanning for corresponding session_meta.json...")
        best_json_folder = None
        best_meta = None
        min_json_diff = float('inf')
        
        for s_dir in search_dirs:
            if not os.path.exists(s_dir): continue
            for root, dirs, files in os.walk(s_dir):
                if "session_meta.json" in files:
                    json_path = os.path.join(root, "session_meta.json")
                    try:
                        with open(json_path, 'r') as f:
                            meta = json.load(f)
                            utc_str = meta.get("created_utc")
                            
                        if utc_str:
                            try: utc_dt = datetime.strptime(utc_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                            except ValueError:
                                try: utc_dt = datetime.strptime(utc_str, "%Y-%m-%dT%H:%M:%SZ")
                                except ValueError: utc_dt = datetime.strptime(utc_str.split('.')[0], "%Y-%m-%dT%H:%M:%S")

                            start_ist = (utc_dt.replace(tzinfo=timezone.utc) + timedelta(hours=5, minutes=30)).replace(tzinfo=None)
                            
                            vid_path = os.path.join(root, "LENS1", "video_lens1.mp4")
                            if os.path.exists(vid_path):
                                cap = cv2.VideoCapture(vid_path)
                                fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
                                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                                cap.release()
                                
                                if frames > 0:
                                    duration_sec = frames / fps
                                    end_ist = start_ist + timedelta(seconds=duration_sec)
                                    
                                    # Perfect Internal Match
                                    if start_ist <= target_dt <= end_ist:
                                        best_json_folder = root
                                        best_meta = meta
                                        min_json_diff = 0
                                        break
                                    # Closest Border Match (Fallback)
                                    else:
                                        diff = min(abs((start_ist - target_dt).total_seconds()), abs((end_ist - target_dt).total_seconds()))
                                        if diff < min_json_diff and diff <= 15 * 60:
                                            min_json_diff = diff
                                            best_json_folder = root
                                            best_meta = meta
                    except Exception:
                        continue
            if min_json_diff == 0: break

        if not best_json_folder:
            if verbose: UI.error("meta.json files not available to map with the csv files for this timestamp.")
            return None, None, None

        missing_lenses = []
        for i in range(1, 7):
            if not os.path.exists(os.path.join(best_json_folder, f"LENS{i}", f"video_lens{i}.mp4")):
                missing_lenses.append(i)

        if missing_lenses:
            if verbose: UI.error(f"mp4 files for the particular timestamp are missing. (Missing Lenses: {missing_lenses})")
            return None, None, None

        return best_csv, best_json_folder, best_meta

def build_session_registry(base_root_dir):
    registry = []
    if not os.path.exists(base_root_dir):
        return registry
        
    for root_path, _, files in os.walk(base_root_dir):
        if "session_meta.json" in files:
            try:
                with open(os.path.join(root_path, "session_meta.json"), 'r') as f: 
                    meta = json.load(f)
                    
                utc_str = meta.get("created_utc")
                if not utc_str: continue
                    
                try:
                    utc_dt = datetime.strptime(utc_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                except ValueError:
                    try:
                        utc_dt = datetime.strptime(utc_str, "%Y-%m-%dT%H:%M:%SZ")
                    except ValueError:
                        utc_dt = datetime.strptime(utc_str.split('.')[0], "%Y-%m-%dT%H:%M:%S")
                
                ist_start_dt = (utc_dt.replace(tzinfo=timezone.utc) + timedelta(hours=5, minutes=30)).replace(tzinfo=None)
                
                vid_path = os.path.join(root_path, "LENS1", "video_lens1.mp4")
                if not os.path.exists(vid_path): continue
                    
                cap = cv2.VideoCapture(vid_path)
                if not cap.isOpened(): continue
                    
                fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()
                
                if frames == 0: continue 
                
                registry.append({
                    "folder_path": root_path, 
                    "folder_name": os.path.basename(root_path),
                    "start_ist": ist_start_dt, 
                    "end_ist": ist_start_dt + timedelta(seconds=frames/fps), 
                    "fps": fps, 
                    "total_frames": frames
                })
            except Exception as e: 
                UI.warn(f"Error reading registry for {root_path}: {e}")
                
    return registry

def process_imu_gps_file(csv_path, registry, master_mapping_path):
    mapped_data = []
    
    try:
        def skip_comments(file_obj):
            for line in file_obj:
                if line.strip() and not line.strip().startswith('#'):
                    yield line

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(skip_comments(f))
            
            if not reader.fieldnames:
                UI.error("CSV appears to be empty after skipping comments.")
                return None
                
            time_col = next((c for c in reader.fieldnames if c and ('timestamp_local' in c.lower() or 'local_timestamp' in c.lower() or c.strip().lower() == 'timestamp')), None)
            if not time_col: 
                UI.error(f"Timestamp column missing. Found: {reader.fieldnames}")
                return None
            
            seen_in_this_file = set()
            for row in reader:
                time_str = row[time_col].strip()
                if not time_str: continue
                
                try: 
                    clean_time_str = time_str.replace("T", " ").split('.')[0]
                    if clean_time_str in seen_in_this_file:
                        continue
                        
                    ist_dt = datetime.strptime(clean_time_str, "%Y-%m-%d %H:%M:%S")
                except ValueError: 
                    continue
                    
                # --- STRICT SESSION MATCHING ---
                best_session = None
                min_s_diff = float('inf')
                
                for s in registry:
                    if s["start_ist"] <= ist_dt <= s["end_ist"]:
                        best_session = s
                        break
                    
                    diff = abs((s["start_ist"] - ist_dt).total_seconds())
                    if diff < min_s_diff and diff <= 15 * 60:
                        min_s_diff = diff
                        best_session = s
                        
                if not best_session: continue
                    
                utc_dt = (ist_dt - timedelta(hours=5, minutes=30)).replace(tzinfo=timezone.utc)
                elapsed_sec = (ist_dt - best_session["start_ist"]).total_seconds()
                
                if elapsed_sec < 0: elapsed_sec = 0.0
                
                mapped_data.append({
                    "ist_time": clean_time_str, 
                    "utc_time": utc_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "unix_time": f"{utc_dt.timestamp():.3f}", 
                    "source_folder": best_session["folder_name"],
                    "folder_path": best_session["folder_path"], 
                    "playback_time_sec": f"{elapsed_sec:.3f}",
                    "frame_no": int(elapsed_sec * best_session["fps"])
                })
                seen_in_this_file.add(clean_time_str)
                
        if not mapped_data:
            UI.warn("CSV was read, but no timestamps matched the available video clips.")
            return None
            
        file_exists = os.path.exists(master_mapping_path)
        existing_timestamps = set()
        
        if file_exists:
            try:
                with open(master_mapping_path, 'r', encoding='utf-8') as f:
                    existing_timestamps = {r['ist_time'] for r in csv.DictReader(f) if 'ist_time' in r}
            except Exception: pass
            
        new_rows = [r for r in mapped_data if r['ist_time'] not in existing_timestamps]
        
        if new_rows:
            with open(master_mapping_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["ist_time", "utc_time", "unix_time", "source_folder", "folder_path", "playback_time_sec", "frame_no"])
                if not file_exists:
                    writer.writeheader()
                writer.writerows(new_rows)
            
        return mapped_data
        
    except FileNotFoundError:
        UI.error(f"CSV File not found: {csv_path}")
        return None
    except Exception as e:
        UI.error(f"Failed to process CSV: {e}")
        traceback.print_exc()
        return None


# ==========================================
# 3. UNIVERSAL TARGET SELECTOR & MAPPING
# ==========================================

def find_sharpest_frame_offset(video_path, base_frame, offsets=[0, 10, 20]):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        return base_frame
    
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if base_frame >= total_frames:
        cap.release()
        return base_frame

    best_frame = base_frame
    max_sharpness = -1.0
    max_offset = max(offsets)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, base_frame)
    current_frame = base_frame
    
    for _ in range(max_offset + 1):
        ret, frame = cap.read()
        if not ret: break
            
        offset_val = current_frame - base_frame
        if offset_val in offsets:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness = round(sharpness, 3)
            
            if sharpness > max_sharpness: 
                max_sharpness = sharpness
                best_frame = current_frame
                
        current_frame += 1
                
    cap.release()
    return best_frame

def get_exact_frame_label(meta, f_num):
    if "start_ist_str" in meta and "fps" in meta:
        start_ist = datetime.fromisoformat(meta["start_ist_str"])
        exact_time = start_ist + timedelta(seconds=f_num / meta["fps"])
        return f"{exact_time.strftime('%Y-%m-%d_%H-%M-%S')}_F{f_num}"
    else:
        safe_time = meta.get("ist_time", "Unknown").replace(":", "-").replace(" ", "_")
        return f"{safe_time}_F{f_num}"

def get_target_frame(mapped_data, registry, root_output, target_dt, override_meta=None, is_clip=False, batch_dur=None):
    if batch_dur is None:
        print(f"\n{UI.CYAN}--- Target Mapping ---{UI.RESET}")

    # 1. If override_meta has a valid pre-calculated frame (Option 3), use it
    if override_meta is not None and override_meta.get("frame_no", 0) > 0:
        return override_meta["folder_path"], int(override_meta["frame_no"]), override_meta

    # 2. PURE MATHEMATICAL TIME CALCULATION
    best_session = None
    min_diff = float('inf')
    
    for s in registry:
        # Heal dates to isolate TIME difference (handles DD/MM CSV swap bugs)
        s_start_healed = s["start_ist"].replace(year=target_dt.year, month=target_dt.month, day=target_dt.day)
        s_end_healed = s["end_ist"].replace(year=target_dt.year, month=target_dt.month, day=target_dt.day)
        
        if s_start_healed <= target_dt <= s_end_healed:
            best_session = s
            break
        else:
            diff = min(abs((s_start_healed - target_dt).total_seconds()), abs((s_end_healed - target_dt).total_seconds()))
            if diff < min_diff and diff <= 15 * 60: # Within 15 minutes of the video bounds
                min_diff = diff
                best_session = s

    if best_session:
        # Mathematically calculate elapsed seconds from the start of the video
        s_start_healed = best_session["start_ist"].replace(year=target_dt.year, month=target_dt.month, day=target_dt.day)
        elapsed_sec = (target_dt - s_start_healed).total_seconds()
        
        if elapsed_sec < 0: elapsed_sec = 0.0
        
        # Exact Temporal Calculation (MSEC) for perfect Multi-Lens Sync
        target_msec = elapsed_sec * 1000.0
        frame_no = int(elapsed_sec * best_session["fps"])
        
        if batch_dur is None:
            UI.info(f"Target locked mathematically! Source: {best_session['folder_name']} | Time: {elapsed_sec:.3f}s (Frame ~{frame_no})")

        meta = {
            "ist_time": target_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "source_folder": best_session["folder_name"],
            "folder_path": best_session["folder_path"],
            "playback_time_sec": f"{elapsed_sec:.3f}",
            "playback_time_msec": target_msec,  # <--- EXACT TIME INJECTED HERE
            "frame_no": frame_no,
            "start_ist_str": best_session["start_ist"].isoformat(),
            "fps": best_session["fps"]
        }

        # Handle video clip logic if requested
        if is_clip:
            if batch_dur is not None:
                max_avail = (best_session["total_frames"] - frame_no) / best_session["fps"]
                meta["clip_duration_sec"] = min(float(batch_dur), max_avail)
            else:
                while True:
                    try:
                        dur = float(UI.input("\nEnter duration of clip in seconds (e.g., 5.0): "))
                        max_avail = (best_session["total_frames"] - frame_no) / best_session["fps"]
                        if dur > max_avail:
                            UI.warn(f"Only {max_avail:.1f}s available from this timestamp to end of video.")
                            if UI.ask_yes_no(f"Clip using available {max_avail:.1f}s?"):
                                meta["clip_duration_sec"] = max_avail
                                break
                        else:
                            meta["clip_duration_sec"] = dur
                            break
                    except ValueError:
                        UI.error("Invalid duration.")

        return best_session["folder_path"], frame_no, meta

    # 3. Ultimate Fallback 
    UI.error("Could not mathematically map this timestamp to any available video session.")
    return None, None, None


# ==========================================
# GPU BLENDING & REMAP HELPERS
# ==========================================

def _blend_frames(warpeds, weights):
    """Brightness-match and weighted-blend 6 warped lens frames.
    Uses CuPy GPU arrays when available, NumPy otherwise.
    Returns NumPy uint8 array."""
    xp = cp if _GPU else np
    H, W = warpeds[0].shape[:2]
    total_img    = xp.zeros((H, W, 3), dtype=xp.float32)
    total_weight = xp.zeros((H, W),    dtype=xp.float32)

    xwarpeds = [xp.asarray(w, dtype=xp.float32) for w in warpeds]
    xweights = [xp.asarray(w) for w in weights]

    ref_mask = xweights[0] > 0.1
    for i in range(1, 6):
        overlap = ref_mask & (xweights[i] > 0.1)
        if int(overlap.sum()) > 100:
            for c in range(3):
                ref_m = float(xwarpeds[0][:, :, c][overlap].mean())
                src_m = float(xwarpeds[i][:, :, c][overlap].mean())
                if src_m > 1.0:
                    xwarpeds[i][:, :, c] *= (ref_m / src_m)

    for i in range(6):
        w3 = xp.dstack([xweights[i]] * 3)
        total_img    += xwarpeds[i] * w3
        total_weight += xweights[i]

    total_weight[total_weight == 0] = 1.0
    result = xp.clip(total_img / xp.dstack([total_weight] * 3), 0, 255)

    if _GPU:
        result = cp.asnumpy(result)
    return result.astype(np.uint8)

_BICUBIC_REMAP_KERNEL = None
def _get_bicubic_remap_kernel():
    global _BICUBIC_REMAP_KERNEL
    if _BICUBIC_REMAP_KERNEL is not None:
        return _BICUBIC_REMAP_KERNEL
    _BICUBIC_REMAP_KERNEL = cp.RawKernel(r'''
    extern "C" __global__
    void bicubic_remap(
        const unsigned char* src, int src_h, int src_w,
        const float* map_x, const float* map_y,
        unsigned char* dst, int dst_h, int dst_w)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= dst_h * dst_w) return;

        int oy = idx / dst_w;
        int ox = idx % dst_w;

        float fx = map_x[idx];
        float fy = map_y[idx];

        if (fx < 1.0f || fx >= (float)(src_w - 2) ||
            fy < 1.0f || fy >= (float)(src_h - 2)) {
            dst[(oy * dst_w + ox) * 3]     = 0;
            dst[(oy * dst_w + ox) * 3 + 1] = 0;
            dst[(oy * dst_w + ox) * 3 + 2] = 0;
            return;
        }

        int ix = (int)floorf(fx);
        int iy = (int)floorf(fy);
        float dx = fx - (float)ix;
        float dy = fy - (float)iy;

        #define KEYS(t) ((t) <= 1.0f \
            ? ((1.5f*(t) - 2.5f)*(t)*(t) + 1.0f) \
            : ((-0.5f*(t) + 2.5f)*(t) - 4.0f)*(t) + 2.0f)

        float wy[4], wx[4];
        for (int j = 0; j < 4; j++) {
            float ty = fabsf(dy - (float)(j - 1));
            wy[j] = (ty < 2.0f) ? KEYS(ty) : 0.0f;
            float tx = fabsf(dx - (float)(j - 1));
            wx[j] = (tx < 2.0f) ? KEYS(tx) : 0.0f;
        }
        #undef KEYS

        for (int c = 0; c < 3; c++) {
            float val = 0.0f;
            for (int jj = 0; jj < 4; jj++) {
                int sy = iy - 1 + jj;
                sy = max(0, min(sy, src_h - 1));
                float row_val = 0.0f;
                for (int ii = 0; ii < 4; ii++) {
                    int sx = ix - 1 + ii;
                    sx = max(0, min(sx, src_w - 1));
                    row_val += wx[ii] * (float)src[(sy * src_w + sx) * 3 + c];
                }
                val += wy[jj] * row_val;
            }
            dst[(oy * dst_w + ox) * 3 + c] =
                (unsigned char)fminf(fmaxf(val, 0.0f), 255.0f);
        }
    }
    ''', 'bicubic_remap')
    return _BICUBIC_REMAP_KERNEL

def _gpu_remap(frame_np, map_x_np, map_y_np, gpu_maps=None):
    if not _GPU:
        return cv2.remap(frame_np, map_x_np, map_y_np, cv2.INTER_CUBIC)

    kernel = _get_bicubic_remap_kernel()
    src = cp.asarray(np.ascontiguousarray(frame_np))
    src_h, src_w = frame_np.shape[:2]
    dst_h, dst_w = map_x_np.shape[:2]

    if gpu_maps is not None:
        gx, gy = gpu_maps
    else:
        gx = cp.asarray(map_x_np)
        gy = cp.asarray(map_y_np)

    dst = cp.empty((dst_h, dst_w, 3), dtype=cp.uint8)
    n = dst_h * dst_w
    block = 256
    grid = (n + block - 1) // block
    kernel((grid,), (block,), (src, src_h, src_w, gx, gy, dst, dst_h, dst_w))
    return dst 

def _upload_maps_to_gpu(maps_x, maps_y):
    if not _GPU: return None
    return [(cp.asarray(mx), cp.asarray(my)) for mx, my in zip(maps_x, maps_y)]

# ==========================================
# FFMPEG / NVENC ENCODER HELPERS
# ==========================================

_NVENC_AVAILABLE = None 
def _check_nvenc_once():
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE
    try:
        probe = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True, timeout=5)
        if 'h264_nvenc' not in probe.stdout:
            print("[Encoder] libx264 (NVENC not in FFmpeg build)")
            _NVENC_AVAILABLE = False
            return False

        test_frame = bytes(256 * 128 * 3)
        test_cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24', '-s', '256x128', '-r', '1', '-i', 'pipe:',
            '-c:v', 'h264_nvenc', '-pix_fmt', 'yuv420p',
            '-frames:v', '1', '-f', 'null', '-'
        ]
        test = subprocess.run(test_cmd, input=test_frame, capture_output=True, timeout=10)
        if test.returncode == 0:
            print("[Encoder] h264_nvenc (NVIDIA GPU)")
            _NVENC_AVAILABLE = True
            return True
            
    except Exception: pass
    print("[Encoder] libx264 (NVENC unavailable)")
    _NVENC_AVAILABLE = False
    return False

def _build_ffmpeg_cmd(output_path, W, H, fps):
    base = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{W}x{H}',
        '-r', str(fps),
        '-i', 'pipe:',
    ]
    if _check_nvenc_once():
        return base + ['-c:v', 'h264_nvenc', '-preset', _cfg("output", "nvenc_preset", "p4"),
                       '-rc', 'vbr', '-cq', str(_cfg("output", "nvenc_cq", 23)), '-b:v', '0',
                       '-pix_fmt', 'yuv420p', output_path]
    return base + ['-c:v', 'libx264', '-preset', _cfg("output", "cpu_preset", "medium"), '-crf', str(_cfg("output", "crf", 23)),
                   '-pix_fmt', 'yuv420p', output_path]


# ==========================================
# 4A. STATIC IMAGE STITCHER CLASSES (Date Routed)
# ==========================================

class OptimizedImageStitcher:
    def __init__(self, intrinsics_dir):
        self.intrinsics_dir = intrinsics_dir
        self.W = 1920
        self.H = 960
        self.CROP_TOP = 191      
        self.CROP_BOTTOM = 193   
        
        self.LENS_ZOOMS = [1.2500, 1.2200, 1.0800, 1.2500, 1.2500, 1.2500]
        self.LENS_ORIENTATIONS = [
            (-73.1664, 0.0000),  # Lens 1
            (-135.8652, 0.0000), # Lens 2
            (165.3833, 0.0000),  # Lens 3
            (117.0267, 0.0000),  # Lens 4
            (47.8264, 0.0000),   # Lens 5
            (-15.5961, 0.0000),  # Lens 6
        ]
        
        self.maps_x = []
        self.maps_y = []
        self.weights = []
        self.gpu_maps = None
        self.base_Ks = [] 
        self.Ks = []      
        self.img_dims = []
        self.final_errors = [0.0] * 6 
        
        self._pysift = PySIFT(dsp=True, rootsift=True) if _PYSIFT_AVAILABLE else None

        self._load_intrinsics()
        print(f"[Optimized Image Stitcher] Initializing Geometry ({self.W}x{self.H})...")
        self._init_geometry_and_weights()

    def _load_intrinsics(self):
        self.base_Ks = []
        for i in range(1, 7):
            path = os.path.join(self.intrinsics_dir, f"calibration_pinhole_lens{i}.json")
            if not os.path.exists(path):
                print(f"[WARNING] {path} not found. Using generic fallback.")
                K = np.array([[1000,0,960],[0,1000,540],[0,0,1]], dtype=np.float32)
                self.img_dims.append((1920,1080))
                self.base_Ks.append(K)
            else:
                with open(path, 'r') as f:
                    data = json.load(f)
                K = np.array(data['K'], dtype=np.float32)
                self.base_Ks.append(K)
                self.img_dims.append(tuple(data['image_size']))

    def _init_geometry_and_weights(self):
        self.maps_x = []
        self.maps_y = []
        self.weights = []
        self.Ks = []
        
        for i in range(6):
            K = self.base_Ks[i].copy()
            K[0,0] *= self.LENS_ZOOMS[i]
            K[1,1] *= self.LENS_ZOOMS[i]
            self.Ks.append(K)

        u_grid, v_grid = np.meshgrid(np.arange(self.W), np.arange(self.H))
        theta = (u_grid / self.W) * 2 * np.pi
        theta = (theta + np.pi) % (2 * np.pi) 
        phi = (np.pi / 2) - ((v_grid / self.H) * np.pi)
        
        x_s = np.cos(phi) * np.sin(theta)
        y_s = np.sin(phi) 
        z_s = np.cos(phi) * np.cos(theta)
        rays_global = np.dstack((x_s, -y_s, z_s)) 

        for i in range(6):
            yaw_deg, pitch_deg = self.LENS_ORIENTATIONS[i]
            yaw = np.radians(yaw_deg)
            pitch = np.radians(pitch_deg)
            
            Ry = np.array([
                [np.cos(yaw), 0, -np.sin(yaw)],
                [0, 1, 0],
                [np.sin(yaw), 0, np.cos(yaw)]
            ], dtype=np.float32)
            
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)]
            ], dtype=np.float32)
            
            R_inv = (Ry @ Rx).T
            rays_local = rays_global.reshape(-1, 3) @ R_inv
            
            z_vals = rays_local[:, 2]
            z_vals[z_vals <= 0.01] = 0.01 
            
            K = self.Ks[i]
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            
            u_src = (fx * (rays_local[:, 0] / z_vals)) + cx
            v_src = (fy * (rays_local[:, 1] / z_vals)) + cy
            
            map_x = u_src.reshape(self.H, self.W).astype(np.float32)
            map_y = v_src.reshape(self.H, self.W).astype(np.float32)
            
            src_w, src_h = self.img_dims[i]
            valid_mask = (map_x >= 0) & (map_x < src_w) & \
                         (map_y >= 0) & (map_y < src_h) & \
                         (rays_local[:, 2].reshape(self.H, self.W) > 0.1)
            
            dist_x = np.abs(map_x - (src_w / 2)) / (src_w / 2)
            dist_y = np.abs(map_y - (src_h / 2)) / (src_h / 2)
            linear_weight = 1.0 - np.maximum(dist_x, dist_y)
            linear_weight = np.clip(linear_weight, 0, 1)
            weight = np.power(linear_weight, 2.0) 
            weight[~valid_mask] = 0
            
            self.maps_x.append(map_x)
            self.maps_y.append(map_y)
            self.weights.append(weight.astype(np.float32))

        self.gpu_maps = _upload_maps_to_gpu(self.maps_x, self.maps_y)

    def find_shift_with_sift(self, img1, img2):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

        if self._pysift is not None:
            try:
                kp1, des1 = self._pysift.detectAndCompute(gray1, None)
                kp2, des2 = self._pysift.detectAndCompute(gray2, None)
            except Exception as e:
                import traceback
                print(f"\n[WARNING] GPU SIFT failed! Here is the REAL error:")
                traceback.print_exc()  # <--- THIS WILL REVEAL THE CULPRIT
                print("[INFO] Safely falling back to CPU OpenCV SIFT.")
                self._pysift = None
                sift = cv2.SIFT_create()
                kp1, des1 = sift.detectAndCompute(gray1, None)
                kp2, des2 = sift.detectAndCompute(gray2, None)
        else:
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(gray1, None)
            kp2, des2 = sift.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5: return None
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
                
        if len(good) < 4: return None
            
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        
        dx = pts1[:, 0] - pts2[:, 0]
        dy = pts1[:, 1] - pts2[:, 1]
        
        valid_matches = np.abs(dy) < 15 
        if np.sum(valid_matches) < 3: return None
        return np.median(dx[valid_matches])

    def auto_tune_zooms(self, frames):
        print("\n[INFO] Auto-tuning zooms...")
        TEST_ZOOMS = [1.00, 1.04, 1.08, 1.12, 1.16, 1.20, 1.22, 1.23, 1.24, 1.25]
        
        for i in range(6):
            best_zoom = self.LENS_ZOOMS[i]
            best_error = float('inf')

            for test_z in TEST_ZOOMS:
                self.LENS_ZOOMS[i] = test_z
                self._init_geometry_and_weights() 

                warped_i = cv2.remap(frames[i], self.maps_x[i], self.maps_y[i], cv2.INTER_LINEAR)
                
                prev_i = (i - 1) % 6
                next_i = (i + 1) % 6

                warped_prev = cv2.remap(frames[prev_i], self.maps_x[prev_i], self.maps_y[prev_i], cv2.INTER_LINEAR)
                warped_next = cv2.remap(frames[next_i], self.maps_x[next_i], self.maps_y[next_i], cv2.INTER_LINEAR)

                def get_seam_error(w1, w2):
                    mask1 = cv2.cvtColor(w1, cv2.COLOR_BGR2GRAY) > 10
                    mask2 = cv2.cvtColor(w2, cv2.COLOR_BGR2GRAY) > 10
                    overlap = mask1 & mask2
                    if np.sum(overlap) == 0: return float('inf')
                    
                    y, x = np.where(overlap)
                    min_x, max_x = np.min(x), np.max(x)
                    if (max_x - min_x) < 10: return float('inf')
                    
                    s1 = w1[:, min_x:max_x]
                    s2 = w2[:, min_x:max_x]
                    
                    shift = self.find_shift_with_sift(s1, s2)
                    if shift is None: return float('inf') 
                    return abs(shift)

                err_left = get_seam_error(warped_prev, warped_i)
                err_right = get_seam_error(warped_i, warped_next)
                total_err = err_left + err_right
                
                if total_err < best_error:
                    best_error = total_err
                    best_zoom = test_z

            self.LENS_ZOOMS[i] = best_zoom

        self._init_geometry_and_weights()

    def calibrate_angles_with_sift(self, frames):
        MAX_ITERATIONS = 50        
        ERROR_THRESHOLD_DEG = 0.4 
        LEARNING_RATE = 0.25      
        MAX_CORRECTION_STEP = 1.5 
        
        print(f"[INFO] Calibrating angles...")
        pixels_per_deg = self.W / 360.0

        for iteration in range(MAX_ITERATIONS):
            warped = []
            for i in range(6):
                w = cv2.remap(frames[i], self.maps_x[i], self.maps_y[i], cv2.INTER_LINEAR)
                warped.append(w)
                
            new_orientations = list(self.LENS_ORIENTATIONS)
            max_error_deg = 0.0
            total_corrections = 0
            
            for i in range(6):
                next_i = (i + 1) % 6
                
                mask_curr = cv2.cvtColor(warped[i], cv2.COLOR_BGR2GRAY) > 10
                mask_next = cv2.cvtColor(warped[next_i], cv2.COLOR_BGR2GRAY) > 10
                overlap = mask_curr & mask_next
                
                if np.sum(overlap) == 0: continue
                
                y, x = np.where(overlap)
                min_x, max_x = np.min(x), np.max(x)
                if (max_x - min_x) < 10: continue
                
                strip_curr = warped[i][:, min_x:max_x]
                strip_next = warped[next_i][:, min_x:max_x]
                
                shift_px = self.find_shift_with_sift(strip_curr, strip_next)
                if shift_px is None: continue
                    
                shift_deg = shift_px / pixels_per_deg
                
                if abs(shift_deg) > max_error_deg:
                    max_error_deg = abs(shift_deg)
                
                if abs(shift_deg) > 0.1: 
                    old_yaw, pitch = new_orientations[next_i]
                    correction = shift_deg * LEARNING_RATE
                    correction = np.clip(correction, -MAX_CORRECTION_STEP, MAX_CORRECTION_STEP)
                    new_yaw = old_yaw + correction 
                    new_orientations[next_i] = (new_yaw, pitch)
                    total_corrections += 1

            if max_error_deg <= ERROR_THRESHOLD_DEG or (total_corrections == 0 and max_error_deg > ERROR_THRESHOLD_DEG):
                break

            if iteration < MAX_ITERATIONS - 1:
                self.LENS_ORIENTATIONS = new_orientations
                self._init_geometry_and_weights()

        self.final_errors = [round(float(max_error_deg), 3)] * 6

    def stitch_frames(self, frames, output_path, run_sift=False, save_media=True, resize_to=(1920, 960)):
        if run_sift:
            self.auto_tune_zooms(frames)
            self.calibrate_angles_with_sift(frames)

        print(f"\n[INFO] Blending images into final panorama ({'GPU' if _GPU else 'CPU'})...")
        warpeds = [_gpu_remap(frames[i], self.maps_x[i], self.maps_y[i],
                   gpu_maps=self.gpu_maps[i] if self.gpu_maps else None)
                   for i in range(6)]
        
        final_pano = _blend_frames(warpeds, self.weights)
        final_pano = final_pano[self.CROP_TOP : self.H - self.CROP_BOTTOM, :]
        
        if resize_to:
            final_pano = cv2.resize(final_pano, resize_to, interpolation=cv2.INTER_AREA)

        if save_media:
            success = cv2.imwrite(output_path, final_pano)
            if success:
                print(f"[DONE] Saved high-quality stitched panorama to: {output_path}")
            else:
                UI.error(f"Failed to save stitched image to disk. Check permissions: {output_path}")
        return final_pano


class LegacyImageStitcher:
    def __init__(self, intrinsics_dir):
        self.intrinsics_dir = intrinsics_dir
        self.W = 1920
        self.H = 960
        self.CROP_TOP = 191      
        self.CROP_BOTTOM = 193   
                
        self.LENS_ZOOMS = [1.078, 1.127, 1.105, 1.044, 1.118, 1.08]
        self.LENS_ORIENTATIONS = [
            (-60, 0), (-120, 0), (180, 0), (120, 0), (60, 0), (0, 0)       
        ]
        
        self.maps_x = []
        self.maps_y = []
        self.weights = []
        self.gpu_maps = None
        self.Ks = []
        self.img_dims = []
        self.final_errors = [0.0] * 6 

        self._pysift = PySIFT(dsp=True, rootsift=True) if _PYSIFT_AVAILABLE else None

        self._load_intrinsics()
        print(f"[Legacy Image Stitcher] Initializing Geometry ({self.W}x{self.H})...")
        self._init_geometry_and_weights()

    def _load_intrinsics(self):
        for i in range(1, 7):
            path = os.path.join(self.intrinsics_dir, f"calibration_pinhole_lens{i}.json")
            zoom_factor = self.LENS_ZOOMS[i-1]

            if not os.path.exists(path):
                K = np.array([[1000,0,960],[0,1000,540],[0,0,1]], dtype=np.float32)
                self.img_dims.append((1920,1080))
                K[0,0] *= zoom_factor
                K[1,1] *= zoom_factor
                self.Ks.append(K)
            else:
                with open(path, 'r') as f:
                    data = json.load(f)
                K = np.array(data['K'], dtype=np.float32)
                K[0,0] *= zoom_factor 
                K[1,1] *= zoom_factor 
                self.Ks.append(K)
                self.img_dims.append(tuple(data['image_size']))

    def _init_geometry_and_weights(self):
        self.maps_x = []
        self.maps_y = []
        self.weights = []

        u_grid, v_grid = np.meshgrid(np.arange(self.W), np.arange(self.H))
        theta = (u_grid / self.W) * 2 * np.pi
        theta = (theta + np.pi) % (2 * np.pi) 
        phi = (np.pi / 2) - ((v_grid / self.H) * np.pi)
        
        x_s = np.cos(phi) * np.sin(theta)
        y_s = np.sin(phi) 
        z_s = np.cos(phi) * np.cos(theta)
        rays_global = np.dstack((x_s, -y_s, z_s)) 

        for i in range(6):
            yaw_deg, pitch_deg = self.LENS_ORIENTATIONS[i]
            yaw = np.radians(yaw_deg)
            pitch = np.radians(pitch_deg)
            
            Ry = np.array([
                [np.cos(yaw), 0, -np.sin(yaw)],
                [0, 1, 0],
                [np.sin(yaw), 0, np.cos(yaw)]
            ], dtype=np.float32)
            
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)]
            ], dtype=np.float32)
            
            R_inv = (Ry @ Rx).T
            rays_local = rays_global.reshape(-1, 3) @ R_inv
            
            z_vals = rays_local[:, 2]
            z_vals[z_vals <= 0.01] = 0.01 
            
            K = self.Ks[i]
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            
            u_src = (fx * (rays_local[:, 0] / z_vals)) + cx
            v_src = (fy * (rays_local[:, 1] / z_vals)) + cy
            
            map_x = u_src.reshape(self.H, self.W).astype(np.float32)
            map_y = v_src.reshape(self.H, self.W).astype(np.float32)
            
            src_w, src_h = self.img_dims[i]
            valid_mask = (map_x >= 0) & (map_x < src_w) & \
                         (map_y >= 0) & (map_y < src_h) & \
                         (rays_local[:, 2].reshape(self.H, self.W) > 0.1)
            
            dist_x = np.abs(map_x - (src_w / 2)) / (src_w / 2)
            dist_y = np.abs(map_y - (src_h / 2)) / (src_h / 2)
            linear_weight = 1.0 - np.maximum(dist_x, dist_y)
            linear_weight = np.clip(linear_weight, 0, 1)
            weight = np.power(linear_weight, 2.0) 
            weight[~valid_mask] = 0
            
            self.maps_x.append(map_x)
            self.maps_y.append(map_y)
            self.weights.append(weight.astype(np.float32))

        self.gpu_maps = _upload_maps_to_gpu(self.maps_x, self.maps_y)

    def find_shift_with_sift(self, img1, img2):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

        if self._pysift is not None:
            try:
                kp1, des1 = self._pysift.detectAndCompute(gray1, None)
                kp2, des2 = self._pysift.detectAndCompute(gray2, None)
            except Exception as e:
                import traceback
                print(f"\n[WARNING] GPU SIFT failed! Here is the REAL error:")
                traceback.print_exc()  # <--- THIS WILL REVEAL THE CULPRIT
                print("[INFO] Safely falling back to CPU OpenCV SIFT.")
                self._pysift = None
                sift = cv2.SIFT_create()
                kp1, des1 = sift.detectAndCompute(gray1, None)
                kp2, des2 = sift.detectAndCompute(gray2, None)
        else:
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(gray1, None)
            kp2, des2 = sift.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5: return None
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
                
        if len(good) < 4: return None
            
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        
        dx = pts1[:, 0] - pts2[:, 0]
        dy = pts1[:, 1] - pts2[:, 1]
        
        valid_matches = np.abs(dy) < 15 
        if np.sum(valid_matches) < 3: return None
        return np.median(dx[valid_matches])

    def calibrate_angles_with_sift(self, frames):
        MAX_ITERATIONS = 50        
        ERROR_THRESHOLD_DEG = 0.3
        LEARNING_RATE = 0.25      
        MAX_CORRECTION_STEP = 1.5 
        
        print(f"\n--- STARTING AUTO-CALIBRATION (Safe Mode) ---")
        pixels_per_deg = self.W / 360.0

        for iteration in range(MAX_ITERATIONS):
            warped = []
            for i in range(6):
                w = cv2.remap(frames[i], self.maps_x[i], self.maps_y[i], cv2.INTER_LINEAR)
                warped.append(w)
                
            new_orientations = list(self.LENS_ORIENTATIONS)
            max_error_deg = 0.0
            total_corrections = 0
            
            for i in range(6):
                next_i = (i + 1) % 6
                
                mask_curr = cv2.cvtColor(warped[i], cv2.COLOR_BGR2GRAY) > 10
                mask_next = cv2.cvtColor(warped[next_i], cv2.COLOR_BGR2GRAY) > 10
                overlap = mask_curr & mask_next
                if np.sum(overlap) == 0: continue
                
                y, x = np.where(overlap)
                min_x, max_x = np.min(x), np.max(x)
                if (max_x - min_x) < 10: continue
                
                strip_curr = warped[i][:, min_x:max_x]
                strip_next = warped[next_i][:, min_x:max_x]
                
                shift_px = self.find_shift_with_sift(strip_curr, strip_next)
                shift_deg = shift_px / pixels_per_deg
                
                if abs(shift_deg) > max_error_deg:
                    max_error_deg = abs(shift_deg)
                
                if abs(shift_deg) > 0.1:
                    status = "OK" if abs(shift_deg) <= ERROR_THRESHOLD_DEG else "FIXING"
                    print(f"  Lens {i+1} -> {next_i+1}: Error {shift_px:.1f}px ({shift_deg:.2f}°) [{status}]")
                    
                    old_yaw, pitch = new_orientations[next_i]
                    correction = shift_deg * LEARNING_RATE
                    correction = np.clip(correction, -MAX_CORRECTION_STEP, MAX_CORRECTION_STEP)
                    new_yaw = old_yaw + correction 
                    new_orientations[next_i] = (new_yaw, pitch)
                    total_corrections += 1

            if max_error_deg <= ERROR_THRESHOLD_DEG:
                print(f"  >>> SUCCESS! All errors are within range.")
                break
            
            if total_corrections == 0 and max_error_deg > ERROR_THRESHOLD_DEG:
                print("  >>> No clear SIFT matches found to improve further. Stopping.")
                break

            if iteration < MAX_ITERATIONS - 1:
                self.LENS_ORIENTATIONS = new_orientations
                self._init_geometry_and_weights()
            else:
                print("  >>> Reached Max Iterations.")
                
        self.final_errors = [round(float(max_error_deg), 3)] * 6

    def stitch_frames(self, frames, output_path, run_sift=True, save_media=True, resize_to=(1920, 960)):
        if run_sift:
            self.calibrate_angles_with_sift(frames)
        else:
            print("[INFO] SIFT Bypassed. Using cached daily geometry for instant stitching.")
        
        print(f"\n[INFO] Blending images into final panorama ({'GPU' if _GPU else 'CPU'})...")
        warpeds = [_gpu_remap(frames[i], self.maps_x[i], self.maps_y[i],
                   gpu_maps=self.gpu_maps[i] if self.gpu_maps else None)
                   for i in range(6)]
        
        final_pano = _blend_frames(warpeds, self.weights)
        final_pano = final_pano[self.CROP_TOP : self.H - self.CROP_BOTTOM, :]
        
        if resize_to:
            final_pano = cv2.resize(final_pano, resize_to, interpolation=cv2.INTER_AREA)
        
        if save_media:
            success = cv2.imwrite(output_path, final_pano)
            if success:
                print(f"[DONE] Saved high-quality stitched panorama to: {output_path}")
            else:
                UI.error(f"Failed to save stitched image to disk. Check permissions: {output_path}")
        return final_pano

# ==========================================
# 4B. VIDEO STITCHER CLASSES (Date Routed)
# ==========================================

class OptimizedClipVideoStitcher:
    def __init__(self, intrinsics_dir):
        self.intrinsics_dir = intrinsics_dir
        
        self.W = 1920
        self.H = 960
        self.CROP_TOP = 191      
        self.CROP_BOTTOM = 193   
        
        self.LENS_ZOOMS = [1.2500, 1.2200, 1.0800, 1.2500, 1.2500, 1.2500]
        self.LENS_ORIENTATIONS = [
            (-73.1664, 0.0000), # Lens 1
            (-135.8652, 0.0000), # Lens 2
            (165.3833, 0.0000), # Lens 3
            (117.0267, 0.0000), # Lens 4
            (47.8264, 0.0000), # Lens 5
            (-15.5961, 0.0000), # Lens 6
        ]
        
        self.maps_x = []
        self.maps_y = []
        self.weights = []
        self.gpu_maps = None
        self.base_Ks = [] 
        self.Ks = []      
        self.img_dims = []
        self.final_errors = [0.0] * 6 

        self._pysift = PySIFT(dsp=True, rootsift=True) if _PYSIFT_AVAILABLE else None

        self._load_intrinsics()
        print(f"Initializing Geometry ({self.W}x{self.H})...")
        self._init_geometry_and_weights()
        self.FACTORY_ORIENTATIONS = list(self.LENS_ORIENTATIONS) 

    def _load_intrinsics(self):
        self.base_Ks = []
        for i in range(1, 7):
            path = os.path.join(self.intrinsics_dir, f"calibration_pinhole_lens{i}.json")

            if not os.path.exists(path):
                print(f"Warning: {path} not found. Using generic.")
                K = np.array([[1000,0,960],[0,1000,540],[0,0,1]], dtype=np.float32)
                self.img_dims.append((1920,1080))
                self.base_Ks.append(K)
            else:
                with open(path, 'r') as f:
                    data = json.load(f)
                K = np.array(data['K'], dtype=np.float32)
                self.base_Ks.append(K)
                self.img_dims.append(tuple(data['image_size']))

    def _init_geometry_and_weights(self):
        self.maps_x = []
        self.maps_y = []
        self.weights = []

        self.Ks = []
        for i in range(6):
            K = self.base_Ks[i].copy()
            K[0,0] *= self.LENS_ZOOMS[i]
            K[1,1] *= self.LENS_ZOOMS[i]
            self.Ks.append(K)

        u_grid, v_grid = np.meshgrid(np.arange(self.W), np.arange(self.H))
        theta = (u_grid / self.W) * 2 * np.pi
        theta = (theta + np.pi) % (2 * np.pi) 
        phi = (np.pi / 2) - ((v_grid / self.H) * np.pi)
        
        x_s = np.cos(phi) * np.sin(theta)
        y_s = np.sin(phi) 
        z_s = np.cos(phi) * np.cos(theta)
        
        rays_global = np.dstack((x_s, -y_s, z_s)) 

        for i in range(6):
            yaw_deg, pitch_deg = self.LENS_ORIENTATIONS[i]
            yaw = np.radians(yaw_deg)
            pitch = np.radians(pitch_deg)
            
            Ry = np.array([
                [np.cos(yaw), 0, -np.sin(yaw)],
                [0, 1, 0],
                [np.sin(yaw), 0, np.cos(yaw)]
            ], dtype=np.float32)
            
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)]
            ], dtype=np.float32)
            
            R_inv = (Ry @ Rx).T
            rays_local = rays_global.reshape(-1, 3) @ R_inv
            
            z_vals = rays_local[:, 2]
            z_vals[z_vals <= 0.01] = 0.01 
            
            K = self.Ks[i]
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            
            u_src = (fx * (rays_local[:, 0] / z_vals)) + cx
            v_src = (fy * (rays_local[:, 1] / z_vals)) + cy
            
            map_x = u_src.reshape(self.H, self.W).astype(np.float32)
            map_y = v_src.reshape(self.H, self.W).astype(np.float32)
            
            src_w, src_h = self.img_dims[i]
            valid_mask = (map_x >= 0) & (map_x < src_w) & \
                         (map_y >= 0) & (map_y < src_h) & \
                         (rays_local[:, 2].reshape(self.H, self.W) > 0.1)
            
            dist_x = np.abs(map_x - (src_w / 2)) / (src_w / 2)
            dist_y = np.abs(map_y - (src_h / 2)) / (src_h / 2)
            linear_weight = 1.0 - np.maximum(dist_x, dist_y)
            linear_weight = np.clip(linear_weight, 0, 1)
            weight = np.power(linear_weight, 2.0) 
            weight[~valid_mask] = 0
            
            self.maps_x.append(map_x)
            self.maps_y.append(map_y)
            self.weights.append(weight.astype(np.float32))

        self.gpu_maps = _upload_maps_to_gpu(self.maps_x, self.maps_y)

    def find_shift_with_sift(self, img1, img2):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

        if self._pysift is not None:
            try:
                kp1, des1 = self._pysift.detectAndCompute(gray1, None)
                kp2, des2 = self._pysift.detectAndCompute(gray2, None)
            except Exception as e:
                import traceback
                print(f"\n[WARNING] GPU SIFT failed! Here is the REAL error:")
                traceback.print_exc()  # <--- THIS WILL REVEAL THE CULPRIT
                print("[INFO] Safely falling back to CPU OpenCV SIFT.")
                self._pysift = None
                sift = cv2.SIFT_create()
                kp1, des1 = sift.detectAndCompute(gray1, None)
                kp2, des2 = sift.detectAndCompute(gray2, None)
        else:
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(gray1, None)
            kp2, des2 = sift.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5: return None
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
                
        if len(good) < 4: return None
            
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        
        dx = pts1[:, 0] - pts2[:, 0]
        dy = pts1[:, 1] - pts2[:, 1]
        
        valid_matches = np.abs(dy) < 15 
        if np.sum(valid_matches) < 3: return None
        return np.median(dx[valid_matches])
    
    def auto_tune_zooms(self, frames):
        print("\n--- STARTING AUTO-ZOOM TUNING ---")
        TEST_ZOOMS = [1.00, 1.04, 1.08, 1.12, 1.16, 1.20, 1.22, 1.23, 1.24, 1.25]
        
        for i in range(6):
            best_zoom = self.LENS_ZOOMS[i]
            best_error = float('inf')

            for test_z in TEST_ZOOMS:
                self.LENS_ZOOMS[i] = test_z
                self._init_geometry_and_weights() 

                warped_i = cv2.remap(frames[i], self.maps_x[i], self.maps_y[i], cv2.INTER_LINEAR)
                prev_i = (i - 1) % 6
                next_i = (i + 1) % 6

                warped_prev = cv2.remap(frames[prev_i], self.maps_x[prev_i], self.maps_y[prev_i], cv2.INTER_LINEAR)
                warped_next = cv2.remap(frames[next_i], self.maps_x[next_i], self.maps_y[next_i], cv2.INTER_LINEAR)

                def get_seam_error(w1, w2):
                    mask1 = cv2.cvtColor(w1, cv2.COLOR_BGR2GRAY) > 10
                    mask2 = cv2.cvtColor(w2, cv2.COLOR_BGR2GRAY) > 10
                    overlap = mask1 & mask2
                    if np.sum(overlap) == 0: return float('inf')
                    
                    y, x = np.where(overlap)
                    min_x, max_x = np.min(x), np.max(x)
                    if (max_x - min_x) < 10: return float('inf')
                    
                    s1 = w1[:, min_x:max_x]
                    s2 = w2[:, min_x:max_x]
                    
                    shift = self.find_shift_with_sift(s1, s2)
                    if shift is None: return float('inf') 
                    return abs(shift)

                err_left = get_seam_error(warped_prev, warped_i)
                err_right = get_seam_error(warped_i, warped_next)
                total_err = err_left + err_right
                
                if total_err < best_error:
                    best_error = total_err
                    best_zoom = test_z

            self.LENS_ZOOMS[i] = best_zoom

        self._init_geometry_and_weights()

    def calibrate_angles_with_sift(self, frames):
        MAX_ITERATIONS = 50        
        ERROR_THRESHOLD_DEG = 0.4 
        LEARNING_RATE = 0.25      
        MAX_CORRECTION_STEP = 1.5 
        
        print(f"\n--- STARTING ANGLE CALIBRATION ---")
        pixels_per_deg = self.W / 360.0

        for iteration in range(MAX_ITERATIONS):
            warped = []
            for i in range(6):
                w = cv2.remap(frames[i], self.maps_x[i], self.maps_y[i], cv2.INTER_LINEAR)
                warped.append(w)
                
            new_orientations = list(self.LENS_ORIENTATIONS)
            max_error_deg = 0.0
            total_corrections = 0
            
            for i in range(6):
                next_i = (i + 1) % 6
                
                mask_curr = cv2.cvtColor(warped[i], cv2.COLOR_BGR2GRAY) > 10
                mask_next = cv2.cvtColor(warped[next_i], cv2.COLOR_BGR2GRAY) > 10
                overlap = mask_curr & mask_next
                if np.sum(overlap) == 0: continue
                
                y, x = np.where(overlap)
                min_x, max_x = np.min(x), np.max(x)
                if (max_x - min_x) < 10: continue
                
                strip_curr = warped[i][:, min_x:max_x]
                strip_next = warped[next_i][:, min_x:max_x]
                
                shift_px = self.find_shift_with_sift(strip_curr, strip_next)
                if shift_px is None: continue
                    
                shift_deg = shift_px / pixels_per_deg
                
                if abs(shift_deg) > max_error_deg:
                    max_error_deg = abs(shift_deg)
                
                if abs(shift_deg) > 0.1: 
                    if next_i == 0: continue # MASTER ANCHOR LOCK
                        
                    old_yaw, pitch = new_orientations[next_i]
                    correction = shift_deg * LEARNING_RATE
                    correction = np.clip(correction, -MAX_CORRECTION_STEP, MAX_CORRECTION_STEP)
                    new_yaw = old_yaw + correction 
                    
                    base_yaw = self.FACTORY_ORIENTATIONS[next_i][0]
                    new_yaw = np.clip(new_yaw, base_yaw - 4.5, base_yaw + 4.5)
                    
                    new_orientations[next_i] = (new_yaw, pitch)
                    total_corrections += 1

            if max_error_deg <= ERROR_THRESHOLD_DEG or (total_corrections == 0 and max_error_deg > ERROR_THRESHOLD_DEG):
                break

            if iteration < MAX_ITERATIONS - 1:
                self.LENS_ORIENTATIONS = new_orientations
                self._init_geometry_and_weights()

        self.final_errors = [round(float(max_error_deg), 3)] * 6

    def check_drift_and_update(self, frames):
        pixels_per_deg = self.W / 360.0
        TARGET_MAX_ERROR = 0.40   
        MAX_ITERATIONS = 8        
        
        best_orientations = list(self.LENS_ORIENTATIONS)
        best_error = float('inf')
        
        for iteration in range(MAX_ITERATIONS):
            max_drift = 0.0
            new_orientations = list(self.LENS_ORIENTATIONS)
            warped = [cv2.remap(frames[i], self.maps_x[i], self.maps_y[i], cv2.INTER_LINEAR) for i in range(6)]
            
            for i in range(6):
                next_i = (i + 1) % 6
                mask_curr = cv2.cvtColor(warped[i], cv2.COLOR_BGR2GRAY) > 10
                mask_next = cv2.cvtColor(warped[next_i], cv2.COLOR_BGR2GRAY) > 10
                overlap = mask_curr & mask_next
                
                if np.sum(overlap) == 0: continue
                y, x = np.where(overlap)
                min_x, max_x = np.min(x), np.max(x)
                
                padding = 40 
                search_min_x = max(0, min_x - padding)
                search_max_x = min(self.W, max_x + padding)
                
                strip_curr = warped[i][:, search_min_x:search_max_x]
                strip_next = warped[next_i][:, search_min_x:search_max_x]
                
                shift_px = self.find_shift_with_sift(strip_curr, strip_next)
                if shift_px is None: continue
                    
                shift_deg = shift_px / pixels_per_deg
                if abs(shift_deg) > 3.5: continue
                if abs(shift_deg) > max_drift: max_drift = abs(shift_deg)
                
                if abs(shift_deg) > 0.05:
                    if next_i == 0: continue # MASTER ANCHOR
                        
                    old_yaw, pitch = new_orientations[next_i]
                    correction = np.clip(shift_deg * 0.40, -0.6, 0.6)
                    new_yaw = old_yaw + correction
                    
                    base_yaw = self.FACTORY_ORIENTATIONS[next_i][0]
                    new_yaw = np.clip(new_yaw, base_yaw - 4.5, base_yaw + 4.5)
                    
                    new_orientations[next_i] = (new_yaw, pitch)

            if max_drift < best_error:
                best_error = max_drift
                best_orientations = list(self.LENS_ORIENTATIONS)
            elif max_drift > best_error + 0.05:
                self.LENS_ORIENTATIONS = best_orientations
                self._init_geometry_and_weights()
                break

            self.LENS_ORIENTATIONS = new_orientations
            self._init_geometry_and_weights()
            if max_drift <= TARGET_MAX_ERROR: break

    def stitch_video(self, input_paths, output_path, resize_to=(1920, 960), run_sift=False):
        caps = []
        for p in input_paths:
            caps.append(cv2.VideoCapture(p))
            
        if any(not c.isOpened() for c in caps): 
            return False
            
        if run_sift:
            init_frames = []
            for cap in caps:
                ret, frame = cap.read()
                init_frames.append(frame)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
                
            self.auto_tune_zooms(init_frames)
            self.calibrate_angles_with_sift(init_frames)
        else:
            print("\n   [INFO] Bypassing SIFT. Using cached geometry for speed.")
            
        fps = caps[0].get(cv2.CAP_PROP_FPS) or 29.97
        
        crop_h = self.H - self.CROP_TOP - self.CROP_BOTTOM
        out_w, out_h = resize_to if resize_to else (self.W, crop_h)
        
        ffmpeg_cmd = _build_ffmpeg_cmd(output_path, out_w, out_h, fps)
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        _nvenc_in_use = 'h264_nvenc' in ffmpeg_cmd
        
        RESET_FRAME_INTERVAL = int(7 * 60 * fps) 
        frame_idx = 0
        try:
            while True:
                frames = []
                for cap in caps:
                    ret, frame = cap.read()
                    if not ret: 
                        frames = None
                        break
                    frames.append(frame)
                
                if frames is None: break
                
                if frame_idx > 0 and frame_idx % RESET_FRAME_INTERVAL == 0:
                    print(f"\n\n[🚨 HARD RESET] 7 Minutes Reached (Frame {frame_idx}). Wiping out accumulative drift...")
                    print("[INFO] Re-running complete calibration pipeline from scratch...")
                    
                    self.auto_tune_zooms(frames)
                    self.calibrate_angles_with_sift(frames)
                    
                    print("[SUCCESS] New baseline geometry locked in. Resuming stitching seamlessly.\n")
                
                elif frame_idx > 0 and frame_idx % 150 == 0:
                    print(f"\n   [Frame {frame_idx}] Checking thermal drift...")
                    self.check_drift_and_update(frames)
                
                warpeds = [_gpu_remap(frames[i], self.maps_x[i], self.maps_y[i],
                           gpu_maps=self.gpu_maps[i] if self.gpu_maps else None)
                           for i in range(6)]
                
                final_pano = _blend_frames(warpeds, self.weights)
                final_pano = final_pano[self.CROP_TOP : self.H - self.CROP_BOTTOM, :]

                if resize_to:
                    final_pano = cv2.resize(final_pano, resize_to, interpolation=cv2.INTER_AREA)

                if _nvenc_in_use and ffmpeg_proc.poll() is not None:
                    print("[WARNING] NVENC failed mid-encode. Falling back to CPU...")
                    cpu_cmd = [
                        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                        '-pix_fmt', 'bgr24', '-s', f'{out_w}x{out_h}', '-r', str(fps),
                        '-i', 'pipe:', '-c:v', 'libx264', '-preset', 'medium',
                        '-crf', '23', '-pix_fmt', 'yuv420p', output_path
                    ]
                    ffmpeg_proc = subprocess.Popen(cpu_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    _nvenc_in_use = False

                try:
                    ffmpeg_proc.stdin.write(final_pano.tobytes())
                except OSError:
                    pass

                frame_idx += 1
                if frame_idx % 10 == 0: print(f"   Processed {frame_idx} frames...", end='\r')
                    
        except KeyboardInterrupt: pass
        finally:
            for cap in caps: cap.release()
            try: ffmpeg_proc.stdin.close()
            except Exception: pass
            ffmpeg_proc.wait()
            print("\nDone!")
            
        return True


class LegacyClipVideoStitcher:
    def __init__(self, intrinsics_dir):
        self.intrinsics_dir = intrinsics_dir
        
        self.W = 1920
        self.H = 960
        self.CROP_TOP = 191      
        self.CROP_BOTTOM = 193   
        
        self.LENS_ZOOMS = [1.078, 1.127, 1.105, 1.044, 1.118, 1.08]
        
        self.LENS_ORIENTATIONS = [
            (-60, 0),    # Lens 1 
            (-120, 0),   # Lens 2
            (180, 0),    # Lens 3
            (120, 0),    # Lens 4
            (60, 0),     # Lens 5
            (0, 0)       # Lens 6 (Front)
        ]
        
        self.maps_x = []
        self.maps_y = []
        self.weights = []
        self.gpu_maps = None
        self.Ks = []
        self.img_dims = []
        self.final_errors = [0.0] * 6 

        self._pysift = PySIFT(dsp=True, rootsift=True) if _PYSIFT_AVAILABLE else None

        self._load_intrinsics()
        print(f"Initializing Geometry ({self.W}x{self.H})...")
        self._init_geometry_and_weights()
        self.FACTORY_ORIENTATIONS = list(self.LENS_ORIENTATIONS) 

    def _load_intrinsics(self):
        for i in range(1, 7):
            path = os.path.join(self.intrinsics_dir, f"calibration_pinhole_lens{i}.json")
            zoom_factor = self.LENS_ZOOMS[i-1]

            if not os.path.exists(path):
                print(f"Warning: {path} not found. Using generic.")
                K = np.array([[1000,0,960],[0,1000,540],[0,0,1]], dtype=np.float32)
                self.img_dims.append((1920,1080))
                K[0,0] *= zoom_factor
                K[1,1] *= zoom_factor
                self.Ks.append(K)
            else:
                with open(path, 'r') as f:
                    data = json.load(f)
                K = np.array(data['K'], dtype=np.float32)
                print(f"Lens {i}: Applying Zoom Factor {zoom_factor}")
                K[0,0] *= zoom_factor 
                K[1,1] *= zoom_factor 
                self.Ks.append(K)
                self.img_dims.append(tuple(data['image_size']))

    def _init_geometry_and_weights(self):
        self.maps_x = []
        self.maps_y = []
        self.weights = []

        u_grid, v_grid = np.meshgrid(np.arange(self.W), np.arange(self.H))
        
        theta = (u_grid / self.W) * 2 * np.pi
        theta = (theta + np.pi) % (2 * np.pi) 

        phi = (np.pi / 2) - ((v_grid / self.H) * np.pi)
        
        x_s = np.cos(phi) * np.sin(theta)
        y_s = np.sin(phi) 
        z_s = np.cos(phi) * np.cos(theta)
        
        rays_global = np.dstack((x_s, -y_s, z_s)) 

        for i in range(6):
            yaw_deg, pitch_deg = self.LENS_ORIENTATIONS[i]
            
            yaw = np.radians(yaw_deg)
            pitch = np.radians(pitch_deg)
            
            Ry = np.array([
                [np.cos(yaw), 0, -np.sin(yaw)],
                [0, 1, 0],
                [np.sin(yaw), 0, np.cos(yaw)]
            ], dtype=np.float32)
            
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)]
            ], dtype=np.float32)
            
            R_inv = (Ry @ Rx).T
            
            rays_local = rays_global.reshape(-1, 3) @ R_inv
            
            z_vals = rays_local[:, 2]
            z_vals[z_vals <= 0.01] = 0.01 
            
            K = self.Ks[i]
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            
            u_src = (fx * (rays_local[:, 0] / z_vals)) + cx
            v_src = (fy * (rays_local[:, 1] / z_vals)) + cy
            
            map_x = u_src.reshape(self.H, self.W).astype(np.float32)
            map_y = v_src.reshape(self.H, self.W).astype(np.float32)
            
            src_w, src_h = self.img_dims[i]
            valid_mask = (map_x >= 0) & (map_x < src_w) & \
                         (map_y >= 0) & (map_y < src_h) & \
                         (rays_local[:, 2].reshape(self.H, self.W) > 0.1)
            
            dist_x = np.abs(map_x - (src_w / 2)) / (src_w / 2)
            dist_y = np.abs(map_y - (src_h / 2)) / (src_h / 2)
            
            linear_weight = 1.0 - np.maximum(dist_x, dist_y)
            linear_weight = np.clip(linear_weight, 0, 1)
            
            weight = np.power(linear_weight, 2.0) 
            weight[~valid_mask] = 0
            
            self.maps_x.append(map_x)
            self.maps_y.append(map_y)
            self.weights.append(weight.astype(np.float32))

        self.gpu_maps = _upload_maps_to_gpu(self.maps_x, self.maps_y)

    def find_shift_with_sift(self, img1, img2):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

        if self._pysift is not None:
            try:
                kp1, des1 = self._pysift.detectAndCompute(gray1, None)
                kp2, des2 = self._pysift.detectAndCompute(gray2, None)
            except Exception as e:
                import traceback
                print(f"\n[WARNING] GPU SIFT failed! Here is the REAL error:")
                traceback.print_exc()  # <--- THIS WILL REVEAL THE CULPRIT
                print("[INFO] Safely falling back to CPU OpenCV SIFT.")
                self._pysift = None
                sift = cv2.SIFT_create()
                kp1, des1 = sift.detectAndCompute(gray1, None)
                kp2, des2 = sift.detectAndCompute(gray2, None)
        else:
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(gray1, None)
            kp2, des2 = sift.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5: return None
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
                
        if len(good) < 4: return None
            
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        
        dx = pts1[:, 0] - pts2[:, 0]
        dy = pts1[:, 1] - pts2[:, 1]
        
        valid_matches = np.abs(dy) < 15 
        if np.sum(valid_matches) < 3: return None
        return np.median(dx[valid_matches])

    def calibrate_angles_with_sift(self, frames):
        MAX_ITERATIONS = 50        
        ERROR_THRESHOLD_DEG = 0.4 
        LEARNING_RATE = 0.25      
        MAX_CORRECTION_STEP = 1.5 
        
        print(f"\n--- STARTING ANGLE CALIBRATION ---")
        pixels_per_deg = self.W / 360.0

        for iteration in range(MAX_ITERATIONS):
            warped = []
            for i in range(6):
                w = cv2.remap(frames[i], self.maps_x[i], self.maps_y[i], cv2.INTER_LINEAR)
                warped.append(w)
                
            new_orientations = list(self.LENS_ORIENTATIONS)
            max_error_deg = 0.0
            total_corrections = 0
            
            for i in range(6):
                next_i = (i + 1) % 6
                
                mask_curr = cv2.cvtColor(warped[i], cv2.COLOR_BGR2GRAY) > 10
                mask_next = cv2.cvtColor(warped[next_i], cv2.COLOR_BGR2GRAY) > 10
                overlap = mask_curr & mask_next
                if np.sum(overlap) == 0: continue
                
                y, x = np.where(overlap)
                min_x, max_x = np.min(x), np.max(x)
                if (max_x - min_x) < 10: continue
                
                strip_curr = warped[i][:, min_x:max_x]
                strip_next = warped[next_i][:, min_x:max_x]
                
                shift_px = self.find_shift_with_sift(strip_curr, strip_next)
                if shift_px is None: continue
                    
                shift_deg = shift_px / pixels_per_deg
                
                if abs(shift_deg) > max_error_deg:
                    max_error_deg = abs(shift_deg)
                
                if abs(shift_deg) > 0.1: 
                    if next_i == 0: continue # MASTER ANCHOR LOCK
                        
                    old_yaw, pitch = new_orientations[next_i]
                    correction = shift_deg * LEARNING_RATE
                    correction = np.clip(correction, -MAX_CORRECTION_STEP, MAX_CORRECTION_STEP)
                    new_yaw = old_yaw + correction 
                    
                    base_yaw = self.FACTORY_ORIENTATIONS[next_i][0]
                    new_yaw = np.clip(new_yaw, base_yaw - 4.5, base_yaw + 4.5)
                    
                    new_orientations[next_i] = (new_yaw, pitch)
                    total_corrections += 1

            if max_error_deg <= ERROR_THRESHOLD_DEG or (total_corrections == 0 and max_error_deg > ERROR_THRESHOLD_DEG):
                break

            if iteration < MAX_ITERATIONS - 1:
                self.LENS_ORIENTATIONS = new_orientations
                self._init_geometry_and_weights()

        self.final_errors = [round(float(max_error_deg), 3)] * 6

    def check_drift_and_update(self, frames):
        pixels_per_deg = self.W / 360.0
        TARGET_MAX_ERROR = 0.40   
        MAX_ITERATIONS = 8        
        
        best_orientations = list(self.LENS_ORIENTATIONS)
        best_error = float('inf')
        
        for iteration in range(MAX_ITERATIONS):
            max_drift = 0.0
            new_orientations = list(self.LENS_ORIENTATIONS)
            warped = [cv2.remap(frames[i], self.maps_x[i], self.maps_y[i], cv2.INTER_LINEAR) for i in range(6)]
            
            for i in range(6):
                next_i = (i + 1) % 6
                mask_curr = cv2.cvtColor(warped[i], cv2.COLOR_BGR2GRAY) > 10
                mask_next = cv2.cvtColor(warped[next_i], cv2.COLOR_BGR2GRAY) > 10
                overlap = mask_curr & mask_next
                
                if np.sum(overlap) == 0: continue
                y, x = np.where(overlap)
                min_x, max_x = np.min(x), np.max(x)
                
                padding = 40 
                search_min_x = max(0, min_x - padding)
                search_max_x = min(self.W, max_x + padding)
                
                strip_curr = warped[i][:, search_min_x:search_max_x]
                strip_next = warped[next_i][:, search_min_x:search_max_x]
                
                shift_px = self.find_shift_with_sift(strip_curr, strip_next)
                if shift_px is None: continue
                    
                shift_deg = shift_px / pixels_per_deg
                if abs(shift_deg) > 3.5: continue
                if abs(shift_deg) > max_drift: max_drift = abs(shift_deg)
                
                if abs(shift_deg) > 0.05:
                    if next_i == 0: continue # MASTER ANCHOR
                        
                    old_yaw, pitch = new_orientations[next_i]
                    correction = np.clip(shift_deg * 0.40, -0.6, 0.6)
                    new_yaw = old_yaw + correction
                    
                    base_yaw = self.FACTORY_ORIENTATIONS[next_i][0]
                    new_yaw = np.clip(new_yaw, base_yaw - 4.5, base_yaw + 4.5)
                    
                    new_orientations[next_i] = (new_yaw, pitch)

            if max_drift < best_error:
                best_error = max_drift
                best_orientations = list(self.LENS_ORIENTATIONS)
            elif max_drift > best_error + 0.05:
                self.LENS_ORIENTATIONS = best_orientations
                self._init_geometry_and_weights()
                break

            self.LENS_ORIENTATIONS = new_orientations
            self._init_geometry_and_weights()
            if max_drift <= TARGET_MAX_ERROR: break

    def stitch_video(self, input_paths, output_path, resize_to=(1920, 960), run_sift=False):
        caps = []
        for p in input_paths:
            caps.append(cv2.VideoCapture(p))
            
        if any(not c.isOpened() for c in caps): 
            return False
            
        if run_sift:
            init_frames = []
            for cap in caps:
                ret, frame = cap.read()
                init_frames.append(frame)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
                
            self.calibrate_angles_with_sift(init_frames)
        else:
            print("\n   [INFO] Bypassing SIFT. Using cached geometry for speed.")
            
        fps = caps[0].get(cv2.CAP_PROP_FPS) or 29.97
        
        crop_h = self.H - self.CROP_TOP - self.CROP_BOTTOM
        out_w, out_h = resize_to if resize_to else (self.W, crop_h)

        ffmpeg_cmd = _build_ffmpeg_cmd(output_path, out_w, out_h, fps)
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        _nvenc_in_use = 'h264_nvenc' in ffmpeg_cmd
        
        RESET_FRAME_INTERVAL = int(7 * 60 * fps) 
        
        frame_idx = 0
        try:
            while True:
                frames = []
                for cap in caps:
                    ret, frame = cap.read()
                    if not ret: 
                        frames = None
                        break
                    frames.append(frame)
                
                if frames is None: break
                
                if frame_idx > 0 and frame_idx % RESET_FRAME_INTERVAL == 0:
                    print(f"\n\n[🚨 HARD RESET] 7 Minutes Reached (Frame {frame_idx}). Wiping out accumulative drift...")
                    print("[INFO] Re-running legacy angle calibration pipeline from scratch...")
                    
                    self.calibrate_angles_with_sift(frames)
                    
                    print("[SUCCESS] New baseline geometry locked in. Resuming stitching seamlessly.\n")
                
                elif frame_idx > 0 and frame_idx % 150 == 0:
                    print(f"\n   [Frame {frame_idx}] Pausing to check for thermal drift...")
                    self.check_drift_and_update(frames)
                    print(f"   [Frame {frame_idx}] Check complete. Resuming video processing...")
                
                warpeds = [_gpu_remap(frames[i], self.maps_x[i], self.maps_y[i],
                           gpu_maps=self.gpu_maps[i] if self.gpu_maps else None)
                           for i in range(6)]
                
                final_pano = _blend_frames(warpeds, self.weights)
                final_pano = final_pano[self.CROP_TOP : self.H - self.CROP_BOTTOM, :]

                if resize_to:
                    final_pano = cv2.resize(final_pano, resize_to, interpolation=cv2.INTER_AREA)

                if _nvenc_in_use and ffmpeg_proc.poll() is not None:
                    print("[WARNING] NVENC failed mid-encode. Falling back to CPU...")
                    cpu_cmd = [
                        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                        '-pix_fmt', 'bgr24', '-s', f'{out_w}x{out_h}', '-r', str(fps),
                        '-i', 'pipe:', '-c:v', 'libx264', '-preset', 'medium',
                        '-crf', '23', '-pix_fmt', 'yuv420p', output_path
                    ]
                    ffmpeg_proc = subprocess.Popen(cpu_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    _nvenc_in_use = False

                try:
                    ffmpeg_proc.stdin.write(final_pano.tobytes())
                except OSError:
                    pass
                
                frame_idx += 1
                if frame_idx % 10 == 0: print(f"   Processed {frame_idx} frames...", end='\r')
                    
        except KeyboardInterrupt: pass
        finally:
            for cap in caps: cap.release()
            try: ffmpeg_proc.stdin.close()
            except Exception: pass
            ffmpeg_proc.wait()
            print("\nDone!")
            
        return True


# ==========================================
# 5. VIDEO AI CLASSES
# ==========================================

class ClipVideoTrimmer:
    @staticmethod
    def trim(input_path, output_path, start_sec, duration_sec):
        cmd = [
            "ffmpeg", 
            "-ss", str(start_sec),
            "-i", input_path,
            "-t", str(duration_sec),
            "-c", "copy",
            "-map", "0",
            "-copy_unknown",
            "-y",
            output_path
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError:
            return False


class ClipVideoUndistorter:
    @staticmethod
    def process(input_path, output_path, K_orig, D, xi, orig_size, fov_scale=0.4,
                pbar=None, lens_label="", frame_queue=None):
        """Undistort a single lens video. If pbar (tqdm) is provided, updates it
        instead of creating its own. Thread-safe when each thread has its own lens."""
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if vid_w == 0 or vid_h == 0:
            return None

        w = vid_h
        h = vid_w

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
        xi_vec = np.array([xi], dtype=np.float64)

        # Precompute undistortion maps once — reuse every frame with INTER_CUBIC
        # (cv2.omnidir.undistortImage uses fixed bilinear; remap lets us use cubic)
        map1, map2 = cv2.omnidir.initUndistortRectifyMap(
            K_scaled, D, xi_vec, np.eye(3), K_new,
            (w, h), cv2.CV_32FC1, cv2.omnidir.RECTIFY_PERSPECTIVE
        )

        def _start_ffmpeg(cmd):
            return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        def _write_frame(proc, data):
            """Write one frame; return False if the process has already died."""
            if proc.poll() is not None:
                return False
            try:
                proc.stdin.write(data)
                return True
            except OSError:
                return False

        ffmpeg_proc = None
        _nvenc_in_use = False
        if output_path:
            ffmpeg_cmd = _build_ffmpeg_cmd(output_path, w, h, fps)
            ffmpeg_proc = _start_ffmpeg(ffmpeg_cmd)
            _nvenc_in_use = 'h264_nvenc' in ffmpeg_cmd

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        # Always create per-lens bar; also update shared outer pbar if provided
        own_pbar = None
        if total_frames > 0:
            own_pbar = tqdm(total=total_frames, desc=f"  Undistorting {lens_label}",
                           unit="frame", dynamic_ncols=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            undistorted = cv2.remap(frame_rotated, map1, map2, cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
            if ffmpeg_proc:
                ok = _write_frame(ffmpeg_proc, undistorted.tobytes())
                if not ok and _nvenc_in_use:
                    # NVENC failed at runtime — restart with libx264
                    if own_pbar: own_pbar.write(f"[WARNING] NVENC failed for {lens_label}. Falling back to libx264...")
                    try:
                        ffmpeg_proc.stdin.close()
                    except Exception:
                        pass
                    ffmpeg_proc.wait()
                    cpu_cmd = [
                        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                        '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(fps),
                        '-i', 'pipe:', '-c:v', 'libx264', '-preset', 'medium',
                        '-crf', '23', '-pix_fmt', 'yuv420p', output_path
                    ]
                    ffmpeg_proc = _start_ffmpeg(cpu_cmd)
                    _nvenc_in_use = False
                    _write_frame(ffmpeg_proc, undistorted.tobytes())

            if frame_queue is not None:
                frame_queue.put(undistorted.copy())

            if pbar: pbar.update(1)
            if own_pbar: own_pbar.update(1)

        if own_pbar: own_pbar.close()
        if frame_queue is not None:
            frame_queue.put(None)  # end-of-stream sentinel for pipeline mode
        cap.release()
        if ffmpeg_proc:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
        return K_new.tolist(), [w, h]


class ClipVehicleDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.colors = {
            'auto': (0, 165, 255), 
            'bus': (255, 0, 0), 
            'car': (0, 255, 0), 
            'motorbike': (255, 255, 0), 
            'truck': (0, 0, 255)
        }
        self.excluded_classes = ['tractor', 'rickshaw', 'e-rickshaw', 'cart', 'person', 'cycle']
        self.CONF_FAR = 0.10
        self.CONF_CLOSE = 0.18 # Match relaxed distortion threshold from image class

    def setup_resolution(self, w, h):
        ref_w = 3328
        ref_h = 1664
        scale_x = w / ref_w
        scale_y = h / ref_h
        
        self.SKY_Y_LIMIT = int((ref_h * 0.10) * scale_y)
        self.VEHICLE_Y_LIMIT = int(1150 * scale_y)
        self.VIP_Y_LIMIT = int(950 * scale_y)
        
        poly = [
            [0, 1021], [78, 1088], [341, 1165], [419, 1094], 
            [612, 1121], [679, 1206], [875, 1223], [921, 1290], 
            [1083, 1229], [1616, 1215], [1930, 1306], [2064, 1113], 
            [2206, 1065], [2340, 1111], [2807, 1021], [3267, 1009], 
            [3287, 1013], [3328, 1664], [0, 1664]
        ]
        self.EGO_POLYGON = np.array([[int(x * scale_x), int(y * scale_y)] for x, y in poly], np.int32)

    def get_dynamic_threshold(self, bbox_bottom_y, label):
        if label in ['bus', 'truck']: 
            if bbox_bottom_y > self.VIP_Y_LIMIT:
                return 0.15
            else:
                return 0.35  
                
        y_clamped = max(self.SKY_Y_LIMIT, min(bbox_bottom_y, self.VEHICLE_Y_LIMIT))
        if self.VEHICLE_Y_LIMIT == self.SKY_Y_LIMIT:
            ratio = 1.0
        else:
            ratio = ((y_clamped - self.SKY_Y_LIMIT) / (self.VEHICLE_Y_LIMIT - self.SKY_Y_LIMIT)) ** 2 
            
        base_thresh = self.CONF_FAR + (self.CONF_CLOSE - self.CONF_FAR) * ratio
        
        if label == 'auto': return max(base_thresh, 0.60) 
        if label == 'motorbike': return max(base_thresh, 0.20)
        return base_thresh

    def process_video(self, video_path, output_path, json_path, meta, save_media=True):
        start_time = time.perf_counter() 
        
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        self.setup_resolution(w, h)
        
        temp_id_hits = {}
        frame_cache = {}      
        frame_idx = 0
        
        # PASS 1: Tracking and Filtering
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break
                
            # --- UPDATED TRACKER LOGIC ---
            # Run tracker at a strong scale to capture small details
            results = self.model.track(
                frame, 
                persist=True, 
                verbose=False, 
                conf=0.10, 
                iou=0.45, 
                imgsz=1920, 
                tracker=r"C:\viswak_MUMMAS_360degcamera\Insta360ImageAnalysis\FINAL_custom_track.yaml"
            )[0]
            # -----------------------------
            
            temp_dets = []
            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                ids = results.boxes.id.int().cpu().tolist()
                clss = results.boxes.cls.int().cpu().tolist()
                confs = results.boxes.conf.cpu().tolist()
                
                for box, t_id, cls_idx, conf in zip(boxes, ids, clss, confs):
                    if box[3] < self.SKY_Y_LIMIT:
                        continue
                    
                    check_y = int(box[3] - ((box[3] - box[1]) * 0.1))
                    bottom_center = (int((box[0] + box[2]) / 2), check_y)
                    if cv2.pointPolygonTest(self.EGO_POLYGON, bottom_center, False) >= 0: 
                        continue
                        
                    raw_label = self.model.names[cls_idx]
                    if raw_label in self.excluded_classes: 
                        continue
                        
                    req_conf = self.get_dynamic_threshold(box[3], raw_label)
                    
                    if raw_label != 'auto':
                        if temp_id_hits.get(t_id, 0) > 3:
                            req_conf = req_conf * 0.80
                            
                    if conf < req_conf: 
                        continue
                        
                    temp_dets.append({
                        'box': box, 
                        'id': t_id, 
                        'conf': round(conf, 3), 
                        'lbl': raw_label
                    })

            valid_dets = []
            for i in range(len(temp_dets)):
                is_swallowed = False
                det1 = temp_dets[i]
                
                for j in range(len(temp_dets)):
                    if i == j: 
                        continue
                    det2 = temp_dets[j]
                    
                    if det2['lbl'] in ['bus', 'truck'] and det1['lbl'] not in ['bus', 'truck']:
                        x_left = max(det1['box'][0], det2['box'][0])
                        y_top = max(det1['box'][1], det2['box'][1])
                        x_right = min(det1['box'][2], det2['box'][2])
                        y_bottom = min(det1['box'][3], det2['box'][3])
                        
                        if x_right > x_left and y_bottom > y_top:
                            inter_area = (x_right - x_left) * (y_bottom - y_top)
                            box1_area = (det1['box'][2] - det1['box'][0]) * (det1['box'][3] - det1['box'][1])
                            
                            if (inter_area / box1_area) > 0.95 and det1['box'][3] <= det2['box'][3]: 
                                is_swallowed = True
                                break
                                
                if not is_swallowed: 
                    valid_dets.append(det1)
                    temp_id_hits[det1['id']] = temp_id_hits.get(det1['id'], 0) + 1
                    
            frame_cache[frame_idx] = valid_dets
            frame_idx += 1
            
        cap.release()

        # Temporal Smoothing
        history_map = defaultdict(list)
        conf_map = defaultdict(list)
        
        for f in frame_cache:
            for det in frame_cache[f]:
                history_map[det['id']].append(det['lbl'])
                conf_map[det['id']].append(det['conf'])

        master_id_map = {}
        for t_id, history in history_map.items():
            counts = Counter(history)
            final_label = counts.most_common(1)[0][0]
            
            if final_label in ['motorbike', 'auto'] and counts.get('car', 0) / len(history) > 0.20: 
                final_label = 'car'
                
            avg_conf = sum(conf_map[t_id]) / len(conf_map[t_id])
            master_id_map[t_id] = {'label': final_label, 'avg_conf': round(avg_conf, 3)}

        # PASS 2: Render
        cap = cv2.VideoCapture(video_path)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)) if save_media else None
        live_counts = {name: 0 for name in ['car', 'auto', 'bus', 'truck', 'motorbike']}
        counted_ids = set()
        
        full_metadata = {
            "source_file": video_path, 
            "summary_counts": {}, 
            "detections": [], 
            "inherited_meta": meta
        }

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break
                
            frame_log = {"frame": frame_idx, "objects": []}
            current_dets = frame_cache.get(frame_idx, [])
            
            for det in current_dets:
                if det['id'] in master_id_map:
                    t_id = det['id']
                    label = master_id_map[det['id']]['label']
                    box = det['box']
                    avg_c = master_id_map[det['id']]['avg_conf']
                    
                    if t_id not in counted_ids: 
                        live_counts[label] += 1
                        counted_ids.add(t_id)
                        
                    frame_log["objects"].append({
                        "id": t_id, 
                        "label": label, 
                        "bbox": box.tolist(), 
                        "avg_conf": avg_c
                    })
                    
                    if save_media:
                        color = self.colors.get(label, (255, 255, 255))
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 3) 
            
            if save_media:
                cv2.polylines(frame, [self.EGO_POLYGON], isClosed=True, color=(0, 0, 255), thickness=2)
                cv2.rectangle(frame, (10, 10), (250, 230), (0,0,0), -1) 
                
                for i, (v_type, count) in enumerate(live_counts.items()): 
                    cv2.putText(frame, f"{v_type.upper()}: {count}", (20, 80 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                out.write(frame)

            full_metadata["detections"].append(frame_log)
            frame_idx += 1
            
        cap.release()
        if out: out.release()
        
        full_metadata["summary_counts"] = live_counts
        full_metadata["processing_time_sec"] = round(time.perf_counter() - start_time, 4)
        
        if save_media:
            with open(json_path, "w") as f: 
                json.dump(full_metadata, f, indent=4)
            
        return live_counts


class ClipPedestrianDetector:
    def __init__(self, model_path):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        print("Loading SAHI Model...")
        self.model = AutoDetectionModel.from_pretrained(
            model_type='ultralytics', 
            model_path=model_path,
            confidence_threshold=0.15,
            image_size=832, 
            device=self.device,
        )
        
        # High-Resolution Physics Rules (3328x1664)
        self.TARGET_FPS = 30                 
        self.MAX_SPEED_PX_PER_SEC = 130      
        self.MIN_MOVEMENT_PX = 20            
        self.EGO_STOPPED_THRESHOLD_PX = 3.5  

        # Biological Constraints
        self.MAX_PED_WIDTH_PX = 450          
        self.MAX_PED_HEIGHT_PX = 900         
        self.MIN_ASPECT_RATIO = 1.1          

        self.BASE_EGO_POLYGON = np.array([
            [0, 1021], [78, 1088], [341, 1165], [419, 1094], 
            [612, 1121], [679, 1206], [875, 1223], [921, 1290], 
            [1083, 1229], [1616, 1215], [1930, 1306], [2064, 1113], 
            [2206, 1065], [2340, 1111], [2807, 1021], [3267, 1009], 
            [3287, 1013], 
            [3328, 1664], 
            [0, 1664]     
        ], np.int32)

    def process_video(self, video_path, vehicle_json_path, output_path, json_path, meta, save_media=True):
        start_time = time.perf_counter() 
        
        # Note: vehicle_json_path is preserved in signature for compatibility, 
        # but vehicles are tracked live via SAHI Dual-Detection in this pipeline.
        
        print(f"Initializing DeepSORT ReID Tracker for {video_path}...")
        tracker = DeepSort(
            max_age=self.TARGET_FPS * 5, 
            n_init=3, 
            nms_max_overlap=1.0, 
            max_cosine_distance=0.2,
            embedder="mobilenet",
            half=True if self.device == "cuda:0" else False
        )
        
        box_ann = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
        lbl_ann = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
        
        track_history = defaultdict(list)
        track_total_age = defaultdict(int)    
        track_stopped_age = defaultdict(int)  

        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps == 0: original_fps = self.TARGET_FPS
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Dynamically scale polygon
        scale_x = width / 3328.0
        scale_y = height / 1664.0
        scaled_ego_polygon = (self.BASE_EGO_POLYGON * [scale_x, scale_y]).astype(np.int32)
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.TARGET_FPS, (width, height)) if save_media else None
        
        json_data = []
        frame_stride = original_fps / self.TARGET_FPS
        original_frame_count = 0
        target_frame_count = 0
        
        seen_ped_ids = set()
        prev_gray = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if original_frame_count < target_frame_count * frame_stride:
                original_frame_count += 1
                continue
            original_frame_count += 1
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            camera_dx, camera_dy = 0.0, 0.0
            ego_movement = 0.0

            if prev_gray is not None:
                prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
                if prev_pts is not None:
                    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
                    if status is not None:
                        mask = status.flatten() == 1
                        if np.sum(mask) > 0:
                            good_prev = prev_pts[mask].reshape(-1, 2)
                            good_curr = curr_pts[mask].reshape(-1, 2)
                            dxs = good_curr[:, 0] - good_prev[:, 0]
                            dys = good_curr[:, 1] - good_prev[:, 1]
                            camera_dx = float(np.median(dxs))
                            camera_dy = float(np.median(dys))
                            ego_movement = np.sqrt(camera_dx**2 + camera_dy**2)

            prev_gray = gray.copy()
            is_vehicle_stopped = ego_movement < self.EGO_STOPPED_THRESHOLD_PX

            for t_id in list(track_history.keys()):
                for i in range(len(track_history[t_id])):
                    old_cx, old_cy = track_history[t_id][i]
                    track_history[t_id][i] = (old_cx + camera_dx, old_cy + camera_dy)

            # --- SAHI INFERENCE ---
            sahi_frame = frame.copy()
            cv2.fillPoly(sahi_frame, [scaled_ego_polygon], (0, 0, 0))
            
            result = get_sliced_prediction(
                sahi_frame,
                self.model,
                slice_height=832,          
                slice_width=832,           
                overlap_height_ratio=0.15,  
                overlap_width_ratio=0.25,        
                postprocess_type="NMS",          
                postprocess_match_threshold=0.40, 
                verbose=0
            )
            
            xyxy, confidence, class_id, current_vehicle_bboxes = [], [], [], []
            
            for obj in result.object_prediction_list:
                bbox = [obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy]
                if obj.category.id == 0: 
                    xyxy.append(bbox)
                    confidence.append(obj.score.value)
                    class_id.append(obj.category.id)
                elif obj.category.id in [2, 3, 5, 7]:
                    current_vehicle_bboxes.append((bbox, obj.category.id))

            if len(xyxy) > 0:
                people = sv.Detections(
                    xyxy=np.array(xyxy),
                    confidence=np.array(confidence),
                    class_id=np.array(class_id)
                )
            else:
                people = sv.Detections.empty()

            # --- SPATIAL & BIOLOGICAL FILTER ---
            pedestrian_mask = np.ones(len(people), dtype=bool)

            for i, p_box in enumerate(people.xyxy):
                box_width = p_box[2] - p_box[0]
                box_height = p_box[3] - p_box[1]
                p_area = box_width * box_height
                px_center = (p_box[0] + p_box[2]) / 2
                py_bottom = p_box[3]
                
                aspect_ratio = box_height / box_width if box_width > 0 else 0
                    
                if box_width > self.MAX_PED_WIDTH_PX or box_height > self.MAX_PED_HEIGHT_PX or aspect_ratio < self.MIN_ASPECT_RATIO:
                    pedestrian_mask[i] = False
                    continue 

                for v_data in current_vehicle_bboxes:
                    v_box, v_class = v_data
                    
                    if v_class == 3:
                        pad_x, pad_y = 100, 80
                    else:            
                        pad_x, pad_y = 15, 15
                    
                    v_xA, v_yA = v_box[0] - pad_x, v_box[1] - pad_y
                    v_xB, v_yB = v_box[2] + pad_x, v_box[3] + pad_y
                    
                    xA, yA = max(p_box[0], v_xA), max(p_box[1], v_yA)
                    xB, yB = min(p_box[2], v_xB), min(p_box[3], v_yB)
                    interArea = max(0, xB - xA) * max(0, yB - yA)
                    
                    if p_area > 0:
                        overlap_ratio = interArea / p_area
                        feet_inside = (v_xA <= px_center <= v_xB) and (v_yA <= py_bottom <= v_yB)
                        
                        if overlap_ratio > 0.15 or feet_inside:
                            pedestrian_mask[i] = False
                            break

            spatial_filtered_people = people[pedestrian_mask]

            # --- DEEPSORT TRACKER ---
            deepsort_bbs = []
            for i in range(len(spatial_filtered_people)):
                bbox = spatial_filtered_people.xyxy[i]
                conf = spatial_filtered_people.confidence[i]
                cls = spatial_filtered_people.class_id[i]
                
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                deepsort_bbs.append(([x1, y1, w, h], conf, cls))

            ds_tracks = tracker.update_tracks(deepsort_bbs, frame=frame)
            
            ds_xyxy, ds_tracker_id, ds_conf, ds_class_id = [], [], [], []

            for track in ds_tracks:
                if not track.is_confirmed() or track.time_since_update > 0:
                    continue
                    
                ltrb = track.to_ltrb()
                ds_xyxy.append([ltrb[0], ltrb[1], ltrb[2], ltrb[3]])
                ds_tracker_id.append(int(track.track_id))
                ds_conf.append(track.get_det_conf() if track.get_det_conf() is not None else 1.0)
                ds_class_id.append(0)

            if len(ds_xyxy) > 0:
                tracked = sv.Detections(
                    xyxy=np.array(ds_xyxy),
                    confidence=np.array(ds_conf),
                    tracker_id=np.array(ds_tracker_id),
                    class_id=np.array(ds_class_id)
                )
            else:
                tracked = sv.Detections.empty()

            # --- KINEMATIC FILTER ---
            current_speeds = {}
            if tracked.tracker_id is not None and len(tracked.tracker_id) > 0:
                speed_mask = []
                
                for bbox, track_id in zip(tracked.xyxy, tracked.tracker_id):
                    cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
                    track_history[track_id].append((cx, cy))
                    
                    if len(track_history[track_id]) > self.TARGET_FPS:
                        track_history[track_id].pop(0)
                    
                    is_pedestrian = True 
                    
                    if len(track_history[track_id]) >= 3:
                        old_cx, old_cy = track_history[track_id][0]
                        dist = np.sqrt((cx - old_cx)**2 + (cy - old_cy)**2)
                        time_elapsed = (len(track_history[track_id]) - 1) / self.TARGET_FPS
                        speed_px_per_sec = dist / time_elapsed
                        current_speeds[track_id] = speed_px_per_sec
                        
                        if speed_px_per_sec > self.MAX_SPEED_PX_PER_SEC:
                            is_pedestrian = False
                            
                        if len(track_history[track_id]) == self.TARGET_FPS:
                            box_width = bbox[2] - bbox[0]
                            dynamic_min_movement = max(5.0, box_width * 0.15) 
                            if dist < dynamic_min_movement:
                                is_pedestrian = False 
                    
                    speed_mask.append(is_pedestrian)
                    
                tracked = tracked[np.array(speed_mask, dtype=bool)]

            # --- VISIBILITY FILTER ---
            valid_mask = []
            if tracked.tracker_id is not None:
                for t_id in tracked.tracker_id:
                    track_total_age[t_id] += 1
                    if is_vehicle_stopped:
                        track_stopped_age[t_id] += 1
                    
                    age = track_total_age[t_id]
                    
                    if age >= 3:
                        if age > 150 and track_stopped_age[t_id] < 10:
                            valid_mask.append(False)
                        else:
                            valid_mask.append(True)
                    else:
                        valid_mask.append(False)
                
                tracked = tracked[np.array(valid_mask, dtype=bool)]

            if tracked.tracker_id is not None:
                for t_id in tracked.tracker_id:
                    seen_ped_ids.add(int(t_id))

            # --- ANNOTATION & EXPORT ---
            if save_media:
                labels = []
                if tracked.tracker_id is not None:
                    for t_id in tracked.tracker_id:
                        speed = current_speeds.get(t_id, 0)
                        labels.append(f"#{t_id} | {speed:.0f}px/s")
                
                ann = frame.copy()
                ann = box_ann.annotate(scene=ann, detections=tracked)
                ann = lbl_ann.annotate(scene=ann, detections=tracked, labels=labels)

                cv2.putText(ann, f"Total Pedestrians: {len(seen_ped_ids)}", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4, cv2.LINE_AA)
                out.write(ann)

            f_entries = []
            if tracked.tracker_id is not None:
                for b, i, c in zip(tracked.xyxy, tracked.tracker_id, tracked.confidence):
                    f_entries.append({
                        "id": int(i),
                        "conf": float(round(c, 4)),
                        "speed_px_s": float(round(current_speeds.get(i, 0.0), 2)),
                        "bbox": b.tolist()
                    })

            json_data.append({
                "frame": target_frame_count,
                "timestamp": round(target_frame_count / self.TARGET_FPS, 3),
                "detections": f_entries
            })
            
            target_frame_count += 1 

        cap.release()
        if out: out.release()
        
        final_output_dict = {
            "inherited_meta": meta, 
            "total_pedestrians": len(seen_ped_ids), 
            "frames": json_data,
            "processing_time_sec": round(time.perf_counter() - start_time, 4) 
        }
        
        if save_media:
            with open(json_path, 'w') as f: 
                json.dump(final_output_dict, f, indent=4)
            
        return len(seen_ped_ids)


# ==========================================
# 5.6 STATIC IMAGE AI CLASSES (IN-MEMORY OPTIMIZED)
# ==========================================

class PanoramicImageDetector:
    def __init__(self, model_path):
        print(f"[INFO] Loading Model: {model_path}...")
        self.model = YOLO(model_path)
        self.colors = {
            'auto': (0, 165, 255), 'bus': (255, 0, 0), 'car': (0, 255, 0),
            'motorbike': (255, 255, 0), 'truck': (0, 0, 255)
        }
        self.excluded_classes = ['tractor', 'rickshaw', 'e-rickshaw', 'cart', 'person', 'cycle']
        
        # --- FIXED: SKY LIMIT (Top 10% of 1664 is ~166) ---
        self.SKY_Y_LIMIT = int(1664 * 0.10)  
        self.VEHICLE_Y_LIMIT = 1150 
        
        self.EGO_POLYGON = np.array([
            [0, 1021], [78, 1088], [341, 1165], [419, 1094], 
            [612, 1121], [679, 1206], [875, 1223], [921, 1290], 
            [1083, 1229], [1616, 1215], [1930, 1306], [2064, 1113], 
            [2206, 1065], [2340, 1111], [2807, 1021], [3267, 1009], 
            [3287, 1013], 
            [3328, 1664], 
            [0, 1664]     
        ], np.int32)        
        
        self.CONF_FAR = 0.10        
        self.CONF_CLOSE = 0.18    

    def get_dynamic_threshold(self, bbox_bottom_y, label):
        if label in ['bus', 'truck']:
            if bbox_bottom_y > 950:
                return 0.15  
            else:
                return 0.40  

        y_clamped = max(self.SKY_Y_LIMIT, min(bbox_bottom_y, self.VEHICLE_Y_LIMIT))
        
        if self.VEHICLE_Y_LIMIT == self.SKY_Y_LIMIT:
            ratio = 1.0
        else:
            linear_ratio = (y_clamped - self.SKY_Y_LIMIT) / (self.VEHICLE_Y_LIMIT - self.SKY_Y_LIMIT)
            ratio = linear_ratio ** 2 
            
        base_thresh = self.CONF_FAR + (self.CONF_CLOSE - self.CONF_FAR) * ratio
        
        # --- REVERTED: Original Auto & Bike Penalties ---
        if label == 'auto': return max(base_thresh, 0.60) 
        if label == 'motorbike': return max(base_thresh, 0.20)
        return base_thresh

    def process_single_image(self, image_path, output_path, meta=None, frame_arr=None, save_media=True):
        start_time = time.perf_counter() 
        
        if frame_arr is not None:
            frame = frame_arr.copy()
        else:
            if not os.path.exists(image_path): 
                UI.error(f"Image not found: {image_path}")
                return {}

            frame = cv2.imread(image_path)
            if frame is None:
                UI.error(f"Could not read image: {image_path}")
                return {}

        h, w = frame.shape[:2]
        
        full_metadata = {
            "source_file": image_path,
            "resolution": f"{w}x{h}",
            "summary_counts": {},
            "objects": [],
            "inherited_meta": meta if meta else {}
        }
        
        all_boxes = []
        all_confs = []
        all_clss = []
        
        inference_scales = [2560, 3360]
        
        for size in inference_scales:
            results = self.model.predict(frame, 
                                         verbose=False, 
                                         conf=0.10,  
                                         iou=0.75, 
                                         imgsz=size)[0]
            
            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                clss = results.boxes.cls.int().cpu().tolist()
                confs = results.boxes.conf.cpu().tolist()

                for box, cls_idx, conf in zip(boxes, clss, confs):
                    if box[3] < self.SKY_Y_LIMIT: continue 
                    
                    bottom_center = (int((box[0] + box[2]) / 2), int(box[3]))
                    if cv2.pointPolygonTest(self.EGO_POLYGON, bottom_center, False) >= 0:
                        continue
                    
                    raw_label = self.model.names[cls_idx]
                    if raw_label in self.excluded_classes: continue

                    req_conf = self.get_dynamic_threshold(box[3], raw_label)
                    if conf < req_conf: continue
                    
                    all_boxes.append(box.tolist())
                    all_confs.append(conf)
                    all_clss.append(raw_label)

        live_counts = {name: 0 for name in ['car', 'auto', 'bus', 'truck', 'motorbike']}
        
        # --- PRE-NMS: MODIFIED SWALLOWED BOX TRAP ---
        valid_indices = []
        for i in range(len(all_boxes)):
            box1 = all_boxes[i]
            label1 = all_clss[i]
            
            is_swallowed = False
            for j in range(len(all_boxes)):
                if i == j: continue
                box2 = all_boxes[j]
                label2 = all_clss[j]
                
                if label2 in ['bus', 'truck'] and label1 not in ['bus', 'truck']:
                    x_left = max(box1[0], box2[0])
                    y_top = max(box1[1], box2[1])
                    x_right = min(box1[2], box2[2])
                    y_bottom = min(box1[3], box2[3])
                    
                    if x_right > x_left and y_bottom > y_top:
                        intersection_area = (x_right - x_left) * (y_bottom - y_top)
                        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                        
                        ioa = intersection_area / box1_area
                        
                        # Trigger only if strictly swallowed (>95%) AND physically overlapping height-wise
                        if ioa > 0.95 and box1[3] <= box2[3]:
                            is_swallowed = True
                            break
            
            if not is_swallowed:
                valid_indices.append(i)

        filtered_boxes = [all_boxes[i] for i in valid_indices]
        filtered_confs = [all_confs[i] for i in valid_indices]
        filtered_clss = [all_clss[i] for i in valid_indices]

        # --- FIXED: CLASS-AWARE NMS ---
        indices_to_keep = []
        if len(filtered_boxes) > 0:
            cv_boxes = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in filtered_boxes]
            
            # Process NMS per object class to prevent Cars deleting Bikes, etc.
            unique_classes = set(filtered_clss)
            for u_cls in unique_classes:
                cls_indices = [idx for idx, c in enumerate(filtered_clss) if c == u_cls]
                cls_boxes = [cv_boxes[idx] for idx in cls_indices]
                cls_confs = [filtered_confs[idx] for idx in cls_indices]
                
                # NMS threshold raised to 0.65 (boxes must be 65% overlapping to be deleted)
                nms_res = cv2.dnn.NMSBoxes(cls_boxes, cls_confs, score_threshold=0.0, nms_threshold=0.65)
                
                if len(nms_res) > 0:
                    nms_res = nms_res.flatten() if hasattr(nms_res, 'flatten') else nms_res
                    for i in nms_res:
                        indices_to_keep.append(cls_indices[i])
            
            # Draw kept boxes
            for i in indices_to_keep:
                box = filtered_boxes[i]
                conf = filtered_confs[i]
                raw_label = filtered_clss[i]
                
                if raw_label in live_counts: 
                    live_counts[raw_label] += 1
                
                full_metadata["objects"].append({
                    "label": raw_label, 
                    "bbox": box,
                    "confidence": round(conf, 3)
                })

                if save_media:
                    color = self.colors.get(raw_label, (255, 255, 255))
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 3) 
                    label_text = f"{raw_label} {conf:.2f}"
                    cv2.putText(frame, label_text, (box[0], box[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
        # Draw the custom Ego-Vehicle exclusion zone
        if save_media:
            cv2.polylines(frame, [self.EGO_POLYGON], isClosed=True, color=(0, 0, 255), thickness=2)

            # --- DRAW COUNTER OVERLAY ---
            cv2.rectangle(frame, (10, 10), (250, 230), (0,0,0), -1) 
            cv2.putText(frame, "VEHICLE COUNT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            for i, (v_type, count) in enumerate(live_counts.items()):
                text = f"{v_type.upper()}: {count}"
                cv2.putText(frame, text, (20, 80 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imwrite(output_path, frame)
    
        json_output = output_path.replace(".jpg", ".json").replace(".png", ".json")
        full_metadata["summary_counts"] = live_counts
        full_metadata["processing_time_sec"] = round(time.perf_counter() - start_time, 4)
        
        if save_media:
            with open(json_output, "w") as f:
                json.dump(full_metadata, f, indent=4)
            
        return live_counts

import os
import cv2
import json
import time
import torch
import numpy as np
import supervision as sv
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

class PanoramicPedestrianDetector:
    def __init__(self, model_path):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        print("Loading SAHI Model for Image Detection...")
        self.model = AutoDetectionModel.from_pretrained(
            model_type='ultralytics', 
            model_path=model_path,
            confidence_threshold=0.15,
            image_size=832, 
            device=self.device,
        )
        
        # --- BIOLOGICAL GEOMETRY LIMITS ---
        self.MAX_PED_WIDTH_PX = 450          
        self.MAX_PED_HEIGHT_PX = 900         
        self.MIN_ASPECT_RATIO = 1.1          

        # --- CUSTOM EGO-VEHICLE POLYGON ---
        self.BASE_EGO_POLYGON = np.array([
            [0, 1021], [78, 1088], [341, 1165], [419, 1094], 
            [612, 1121], [679, 1206], [875, 1223], [921, 1290], 
            [1083, 1229], [1616, 1215], [1930, 1306], [2064, 1113], 
            [2206, 1065], [2340, 1111], [2807, 1021], [3267, 1009], 
            [3287, 1013], 
            [3328, 1664], 
            [0, 1664]     
        ], np.int32)

    def process_single_image(self, pano_path, veh_json_path, out_img_path, meta, frame_arr=None, save_media=True):
        start_time = time.perf_counter() 
        
        # Note: veh_json_path is preserved in signature for compatibility, 
        # but vehicles are detected live via SAHI Dual-Detection.
        
        if frame_arr is not None:
            img = frame_arr.copy()
        else:
            img = cv2.imread(pano_path)
            if img is None:
                return 0
                
        height, width = img.shape[:2]
        
        # --- Dynamically scale the ego polygon to match the image resolution ---
        scale_x = width / 3328.0
        scale_y = height / 1664.0
        scaled_ego_polygon = (self.BASE_EGO_POLYGON * [scale_x, scale_y]).astype(np.int32)
        
        # =========================================================
        # --- OPTIMIZED SAHI INFERENCE ---
        # =========================================================
        sahi_frame = img.copy()
        cv2.fillPoly(sahi_frame, [scaled_ego_polygon], (0, 0, 0))
        
        result = get_sliced_prediction(
            sahi_frame,
            self.model,
            slice_height=832,          
            slice_width=832,           
            overlap_height_ratio=0.15,  
            overlap_width_ratio=0.25,        
            postprocess_type="NMS",          
            postprocess_match_threshold=0.40, 
            verbose=0
        )
        
        xyxy = []
        confidence = []
        class_id = []
        current_vehicle_bboxes = [] 
        
        for obj in result.object_prediction_list:
            bbox = [obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy]
            
            if obj.category.id == 0: 
                xyxy.append(bbox)
                confidence.append(obj.score.value)
                class_id.append(obj.category.id)
                
            elif obj.category.id in [2, 3, 5, 7]:
                current_vehicle_bboxes.append((bbox, obj.category.id))

        if len(xyxy) > 0:
            people = sv.Detections(
                xyxy=np.array(xyxy),
                confidence=np.array(confidence),
                class_id=np.array(class_id)
            )
        else:
            people = sv.Detections.empty()

        # =========================================================
        # --- BIOLOGICAL & SPATIAL FILTER ---
        # =========================================================
        p_mask = np.ones(len(people), dtype=bool)

        for i, p_box in enumerate(people.xyxy):
            box_width = p_box[2] - p_box[0]
            box_height = p_box[3] - p_box[1]
            p_area = box_width * box_height
            px_center = (p_box[0] + p_box[2]) / 2
            py_bottom = p_box[3]
            
            # --- Size & Aspect Ratio Filter ---
            aspect_ratio = box_height / box_width if box_width > 0 else 0
                
            if box_width > self.MAX_PED_WIDTH_PX or box_height > self.MAX_PED_HEIGHT_PX or aspect_ratio < self.MIN_ASPECT_RATIO:
                p_mask[i] = False
                continue 

            # --- DYNAMIC VEHICLE PADDING ---
            for v_data in current_vehicle_bboxes:
                v_box, v_class = v_data
                
                if v_class == 3: # Motorcycle
                    pad_x = 100
                    pad_y = 80
                else:            # Car, Bus, Truck
                    pad_x = 15
                    pad_y = 15
                
                v_xA = v_box[0] - pad_x
                v_yA = v_box[1] - pad_y
                v_xB = v_box[2] + pad_x
                v_yB = v_box[3] + pad_y
                
                xA, yA = max(p_box[0], v_xA), max(p_box[1], v_yA)
                xB, yB = min(p_box[2], v_xB), min(p_box[3], v_yB)
                interArea = max(0, xB - xA) * max(0, yB - yA)
                
                if p_area > 0:
                    overlap_ratio = interArea / p_area
                    feet_inside = (v_xA <= px_center <= v_xB) and (v_yA <= py_bottom <= v_yB)
                    
                    if overlap_ratio > 0.15 or feet_inside:
                        p_mask[i] = False
                        break

        filtered_people = people[p_mask]
        ped_count = len(filtered_people)
        
        # =========================================================
        # --- ANNOTATION & EXPORT ---
        # =========================================================
        if save_media:
            box_ann = sv.BoxAnnotator()
            img = box_ann.annotate(scene=img, detections=filtered_people)
            
            cv2.putText(img, f"Total Pedestrians: {ped_count}", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4, cv2.LINE_AA)
            cv2.imwrite(out_img_path, img)
            
            out_json_path = out_img_path.replace(".jpg", ".json")
            f_entries = []
            for b, c in zip(filtered_people.xyxy, filtered_people.confidence):
                f_entries.append({
                    "conf": round(float(c), 3),
                    "bbox": b.tolist()
                })
                
            final_output_dict = {
                "inherited_meta": meta,
                "total_pedestrians": ped_count,
                "detections": f_entries,
                "processing_time_sec": round(time.perf_counter() - start_time, 4) 
            }
            with open(out_json_path, 'w') as f:
                json.dump(final_output_dict, f, indent=4)
            
        return ped_count


# ==========================================
# 5.5 STATIC PIPELINE HELPERS (IN-MEMORY)
# ==========================================

def run_extraction(folder, frame_no, out_dir, OMNI_JSON, meta, save_media=True):
    start_time = time.perf_counter() 
    undist_dir = os.path.join(out_dir, "undistorted")
    intr_dir = os.path.join(out_dir, "INTRINSICS")
    os.makedirs(undist_dir, exist_ok=True)
    os.makedirs(intr_dir, exist_ok=True)
    
    K_orig, D, xi, orig_size = load_omni_intrinsics(OMNI_JSON)
    img_paths = []
    extracted_frames = []
    
    for i in range(1, 7):
        print(f"   -> Extracting & Undistorting Lens {i}/6...", end='\r')
        vid_path = os.path.join(folder, f"LENS{i}", f"video_lens{i}.mp4")
        if not os.path.exists(vid_path): 
            UI.error(f"Missing video for Lens {i}: {vid_path}")
            return False, intr_dir, [], [], meta
            
        cap = cv2.VideoCapture(vid_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, meta["playback_time_msec"])
        ret, frame = cap.read()
        cap.release()
        
        if not ret: 
            UI.error(f"Failed to read frame {frame_no} from Lens {i}")
            return False, intr_dir, [], [], meta
            
        undistorted, K_new, w, h = process_and_undistort(frame, K_orig, D, xi, orig_size)
        if undistorted is None: 
            UI.error(f"Undistortion failed for Lens {i}")
            return False, intr_dir, [], [], meta
            
        extracted_frames.append(undistorted)
        out_img_path = os.path.join(undist_dir, f"undistorted_LENS{i}.jpg")
        
        if save_media:
            cv2.imwrite(out_img_path, undistorted, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_paths.append(out_img_path)
        
        # We MUST ALWAYS save the intrinsic JSONs, because the SIFT class reads them from the hard drive!
        intr_dict = {"K": K_new.tolist(), "D": [0]*5, "image_size": [w, h]}
        with open(os.path.join(intr_dir, f"calibration_pinhole_lens{i}.json"), "w") as jf:
            json.dump(intr_dict, jf, indent=2)
            
    print("   -> Extracting & Undistorting Lens 6/6... Done!      ")
            
    meta["extraction_time_sec"] = round(time.perf_counter() - start_time, 4)
    meta["extracted_lenses"] = img_paths
    
    if save_media:
        with open(os.path.join(out_dir, "static_extraction_meta.json"), "w") as f:
            json.dump(meta, f, indent=4)
            
    return True, intr_dir, img_paths, extracted_frames, meta

def run_stitching(output_dir, intr_dir, extracted_frames, meta_info, cached_stitcher=None, run_sift=True, save_media=True):
    start_time = time.perf_counter() 
    pano_path = os.path.join(output_dir, "Final_Static_Stitch.jpg")
    
    stitcher = cached_stitcher if cached_stitcher is not None else OptimizedImageStitcher(intrinsics_dir=intr_dir)
    
    final_pano = stitcher.stitch_frames(extracted_frames, pano_path, run_sift=run_sift, save_media=save_media) 
    

    if final_pano is not None:
        meta_info["stitching_time_sec"] = round(time.perf_counter() - start_time, 4)
        meta_info["lens_zooms"] = stitcher.LENS_ZOOMS
        meta_info["lens_errors_deg"] = [float(e) for e in stitcher.final_errors]
        meta_info["final_orientations"] = [(float(yaw), float(pitch)) for yaw, pitch in stitcher.LENS_ORIENTATIONS]
        
        if save_media:
            try:
                with open(os.path.join(output_dir, "static_stitched_meta.json"), "w") as f: 
                    json.dump(meta_info, f, indent=4)
            except IOError as e:
                UI.warn(f"Could not save stitched_data.json: {e}")
            
        return pano_path, meta_info, stitcher, final_pano
        
    return None, meta_info, stitcher, None


# ==========================================
# 6. PIPELINE EXECUTION LOOPS
# ==========================================

def frames_pipeline(mapped_data, registry, OMNI_JSON, ROOT_OUTPUT, YOLO_VEHICLE_MODEL, YOLO_PEDESTRIAN_MODEL, target_dt, override_meta=None, action_choice=None, batch_mode=False, save_media=True, stitcher_cache=None, date_key=None, veh_detector=None, ped_detector=None):
    folder, frame_no, meta = get_target_frame(mapped_data, registry, ROOT_OUTPUT, target_dt, override_meta=override_meta, is_clip=False)
    if not folder: 
        return None, None # Signal failure back to batch processor

    if not action_choice:
        print("\n--- Image Frames Pipeline Options ---")
        print(" 1) Extract Undistorted Lenses")
        print(" 2) Extract & Stitch Panorama")
        print(" 3) Extract, Stitch, & Detect Vehicles")
        print(" 4) Extract, Stitch, Detect Vehicles & Pedestrians")
        print(" 5) Temporal Burst (Future Vehicle Dynamics)")
        choice = UI.input("Select an option (1-5) [e.g., 2]: ")
    else:
        choice = str(action_choice)
    
    if not batch_mode:
        locked_frame = None
        pipeline_dirs = [d for d in os.listdir(ROOT_OUTPUT) if d.startswith("Pipeline_20")]
        for d in pipeline_dirs:
            try:
                parts = d.split("_F")
                if len(parts) == 2:
                    d_f_no = int(parts[1])
                    if abs(d_f_no - frame_no) <= 25:
                        time_str = parts[0].replace("Pipeline_", "")
                        d_time = datetime.strptime(time_str, "%Y-%m-%d_%H-%M-%S")
                        expected_time = datetime.fromisoformat(meta["start_ist_str"]) + timedelta(seconds=d_f_no / meta["fps"])
                        if abs((d_time - expected_time).total_seconds()) < 2.0:
                            locked_frame = d_f_no
                            break
            except Exception: 
                continue
        
        if locked_frame is not None:
            if locked_frame != frame_no:
                UI.info(f"Existing session detected! Locking to Frame {locked_frame} to prevent duplicates.")
                frame_no = locked_frame
                meta["frame_no"] = frame_no
        else:
            if not meta.get("is_manual", False):
                lens6_path = os.path.join(folder, "LENS6", "video_lens6.mp4")
                if os.path.exists(lens6_path):
                    best_frame = find_sharpest_frame_offset(lens6_path, frame_no)
                    if best_frame != frame_no:
                        UI.success(f"Dynamic Sharpness applied! Shifted Frame {frame_no} -> {best_frame}")
                        frame_no = best_frame
                        meta["frame_no"] = frame_no

    if choice == '5': 
        if not batch_mode:
            try: 
                offset_x = int(UI.input("Enter temporal frame offset (x) (e.g., 15): "))
            except ValueError: 
                UI.error("Invalid integer.")
                return None, None
                
            burst_frames = [frame_no - offset_x, frame_no, frame_no + offset_x]
            burst_results = []
            
            for f in burst_frames:
                if f < 0: 
                    continue
                    
                f_meta = meta.copy()
                f_meta["frame_no"] = f
                f_meta["is_burst_extraction"] = True 
                
                f_safe = get_exact_frame_label(f_meta, f)
                f_dir = os.path.join(ROOT_OUTPUT, f"Pipeline_{f_safe}")
                f_intr = os.path.join(f_dir, "INTRINSICS")
                f_imgs = [os.path.join(f_dir, "undistorted", f"undistorted_LENS{i}.jpg") for i in range(1, 7)]
                f_pano = os.path.join(f_dir, "Final_Static_Stitch.jpg")
                
                extracted_frames_burst = []
                do_extract = True
                if all(os.path.exists(p) for p in f_imgs):
                    if not UI.ask_yes_no(f"Frame {f} lenses exist. Overwrite?"):
                        do_extract = False
                        # BUG FIX: Read images from disk if extraction is skipped
                        extracted_frames_burst = [cv2.imread(p) for p in f_imgs]
                        
                if do_extract:
                    _, f_intr, f_imgs, extracted_frames_burst, f_meta = run_extraction(folder, f, f_dir, OMNI_JSON, f_meta, save_media=save_media)
                    
                do_stitch = True
                if os.path.exists(f_pano):
                    if not UI.ask_yes_no(f"Frame {f} panorama exists. Overwrite?"):
                        do_stitch = False
                        
                if do_stitch:
                    # BUG FIX: Add validation check so it doesn't crash on empty/None lists
                    if extracted_frames_burst and len(extracted_frames_burst) == 6 and not any(img is None for img in extracted_frames_burst):
                        f_pano, f_meta, _, final_pano_burst = run_stitching(f_dir, f_intr, extracted_frames_burst, f_meta, cached_stitcher=None, run_sift=True, save_media=save_media)
                    else:
                        UI.error(f"Cannot stitch frame {f}: Missing/corrupt lenses.")
                        f_pano = None
                    
                if f_pano: 
                    burst_results.append({"frame": f, "output_dir": f_dir, "panorama_path": f_pano})
            
            if burst_results:
                center_label = get_exact_frame_label(meta, frame_no)
                center_time = center_label.split('_F')[0]
                b_dir = os.path.join(ROOT_OUTPUT, f"Temporal_Burst_{center_time}_CenterF{frame_no}")
                os.makedirs(b_dir, exist_ok=True)
                
                with open(os.path.join(b_dir, "temporal_metadata.json"), "w") as jf: 
                    json.dump({"burst_results": burst_results}, jf, indent=4)
                    
                UI.success("Temporal Burst Complete!")
            UI.pause()
        return None, None

    safe_label = get_exact_frame_label(meta, frame_no)
    out_dir = os.path.join(ROOT_OUTPUT, f"Pipeline_{safe_label}")
    intr_dir = os.path.join(out_dir, "INTRINSICS")
    img_paths = [os.path.join(out_dir, "undistorted", f"undistorted_LENS{i}.jpg") for i in range(1, 7)]
    
    pano_path = os.path.join(out_dir, "Final_Static_Stitch.jpg")
    veh_img = os.path.join(out_dir, "Final_Static_Stitch_detected.jpg")
    veh_json = os.path.join(out_dir, "Final_Static_Stitch_detected.json")
    ped_img = os.path.join(out_dir, "Final_Static_Stitch_pedestrians.jpg")
    ped_json = os.path.join(out_dir, "Final_Static_Stitch_pedestrians.json")

    do_ext = True
    do_stitch = False
    do_veh = False
    do_ped = False

    if not batch_mode:
        print(f"\n{UI.CYAN}--- CHECKING EXISTING ARTIFACTS ---{UI.RESET}")
        
        if all(os.path.exists(p) for p in img_paths):
            do_ext = UI.ask_yes_no("Undistorted lenses already exist. Overwrite?")
            
        if choice in ['2', '3', '4']:
            do_stitch = True
            if os.path.exists(pano_path):
                do_stitch = UI.ask_yes_no("Stitched panorama already exists. Overwrite?")
                
        if choice in ['3', '4']:
            do_veh = True
            if os.path.exists(veh_img) and os.path.exists(veh_json):
                do_veh = UI.ask_yes_no("Vehicle detection already exists. Overwrite?")
                
        if choice == '4':
            do_ped = True
            if os.path.exists(ped_img):
                do_ped = UI.ask_yes_no("Pedestrian detection already exists. Overwrite?")
    else:
        intrinsics_exist = all(os.path.exists(os.path.join(intr_dir, f"calibration_pinhole_lens{i}.json")) for i in range(1, 7))
        
        do_ext = not all(os.path.exists(p) for p in img_paths) or not intrinsics_exist or not save_media
        if choice in ['2', '3', '4']: do_stitch = not os.path.exists(pano_path) or not save_media
        if choice in ['3', '4']: do_veh = not (os.path.exists(veh_img) and os.path.exists(veh_json)) or not save_media
        if choice == '4': do_ped = not os.path.exists(ped_img) or not save_media

    if not batch_mode:
        print(f"\n{UI.CYAN}=== PIPELINE EXECUTION ==={UI.RESET}")
    
    extracted_frames = []
    if do_ext: 
        if not batch_mode: UI.info("Step 1: Running Undistortion Extraction...")
        _, intr_dir, img_paths, extracted_frames, meta = run_extraction(folder, frame_no, out_dir, OMNI_JSON, meta, save_media=save_media)
        if not batch_mode and img_paths: UI.success("Extraction Complete.")
    else:
        # BUG FIX: Re-populate extracted_frames by reading images from disk so stitching doesn't crash
        extracted_frames = [cv2.imread(p) for p in img_paths]
        if not batch_mode and img_paths: UI.success("Extraction Complete.")
    
    final_pano = None
    
    # BUG FIX: Validation to prevent IndexError in Stitching if extraction fails or files are corrupt
    valid_extraction = extracted_frames and len(extracted_frames) == 6 and not any(f is None for f in extracted_frames)

    if choice in ['2', '3', '4'] and do_stitch:
        if valid_extraction:
            if not batch_mode: UI.info("Step 2: Running SIFT Stitching...")
            
            # --- NEW CACHING LOGIC ---
            cached_stitcher = None
            run_sift = True
            
            if batch_mode and stitcher_cache is not None and date_key is not None:
                if date_key in stitcher_cache:
                    cached_stitcher = stitcher_cache[date_key]
                    run_sift = False
                    
            pano_path, meta, returned_stitcher, final_pano = run_stitching(out_dir, intr_dir, extracted_frames, meta, cached_stitcher=cached_stitcher, run_sift=run_sift, save_media=save_media)
            
            # Save stitcher to cache for future rows on this same date
            if batch_mode and stitcher_cache is not None and date_key is not None and date_key not in stitcher_cache:
                if returned_stitcher is not None:
                    stitcher_cache[date_key] = returned_stitcher
            
            if not batch_mode and pano_path: UI.success("Stitching Complete.")
        else:
            UI.error("Extraction failed or lenses corrupted. Stitching skipped.")
            pano_path = None
            
    veh_counts = {}
    if choice in ['3', '4']:
        # BUG FIX: Removed "final_pano is not None" condition which skipped detection on existing panos. 
        # Replaced with a safe check ensuring either final_pano OR a valid pano_path exists.
        if do_veh:
            if final_pano is not None or (pano_path and os.path.exists(pano_path)):
                if not batch_mode: UI.info("Step 3: Running YOLO Vehicle Detection...")
                detector = veh_detector if veh_detector else PanoramicImageDetector(YOLO_VEHICLE_MODEL)
                veh_counts = detector.process_single_image(pano_path, veh_img, meta, frame_arr=final_pano, save_media=save_media)
                if not batch_mode and veh_counts: UI.success("Vehicle Detection Complete.")
            else:
                UI.error("Panorama image missing. Vehicle detection skipped.")
        else:
            # FIX: If skipped because artifact exists, read the existing JSON
            if os.path.exists(veh_json):
                try:
                    with open(veh_json, 'r') as f: veh_counts = json.load(f).get("summary_counts", {})
                except Exception: pass
        
    ped_count = 0
    if choice == '4':
        # BUG FIX: Same logic fix for Pedestrian detection as Vehicle detection.
        if do_ped:
            if final_pano is not None or (pano_path and os.path.exists(pano_path)):
                if not batch_mode: UI.info("Step 4: Running YOLO Pedestrian Detection...")
                detector = ped_detector if ped_detector else PanoramicPedestrianDetector(YOLO_PEDESTRIAN_MODEL)
                ped_count = detector.process_single_image(pano_path, veh_json, ped_img, meta, frame_arr=final_pano, save_media=save_media)
                if not batch_mode and ped_count is not None: UI.success("Pedestrian Detection Complete.")
            else:
                UI.error("Panorama image missing. Pedestrian detection skipped.")
        else:
            # FIX: If skipped because artifact exists, read existing JSON
            if os.path.exists(ped_json):
                try:
                    with open(ped_json, 'r') as f: ped_count = json.load(f).get("total_pedestrians", 0)
                except Exception: pass
    
    if not batch_mode:
        UI.success(f"All done! Output at: {out_dir}")
        UI.pause()
        
    # --- EPHEMERAL WORKSPACE CLEANUP ---
    # If the user requested no media, wipe the entire output directory from the disk!
    if not save_media and os.path.exists(out_dir):
        try:
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)
        except Exception:
            pass
            
    return veh_counts, ped_count


# ---------------------------------------------------------------------------
# V4: Checkpoint helpers — save/restore per-lens undistortion progress
# ---------------------------------------------------------------------------
def _load_checkpoint(out_dir):
    """Load checkpoint dict {lens_id_str: output_path} from out_dir/checkpoint.json."""
    path = os.path.join(out_dir, "checkpoint.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def _save_checkpoint(out_dir, done_lenses):
    """Persist {lens_id_str: output_path} to out_dir/checkpoint.json after each lens."""
    path = os.path.join(out_dir, "checkpoint.json")
    with open(path, "w") as f:
        json.dump(done_lenses, f, indent=2)

def _clear_checkpoint(out_dir):
    """Remove checkpoint file after a complete successful run."""
    path = os.path.join(out_dir, "checkpoint.json")
    if os.path.exists(path):
        os.remove(path)
# ---------------------------------------------------------------------------


def _verify_output(output_path, expected_W, expected_H):
    """V4: Run ffprobe after stitching to confirm output integrity."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,codec_name",
             "-of", "json", output_path],
            capture_output=True, text=True, timeout=30
        )
        info = json.loads(result.stdout).get("streams", [{}])[0]
        w     = info.get("width")
        h     = info.get("height")
        codec = info.get("codec_name", "")
        ok    = (w == expected_W and h == expected_H and "h264" in codec)
        status = "OK" if ok else "FAIL"
        print(f"[Verify] {status} — {w}x{h} {codec} — {output_path}")
        return ok
    except Exception as e:
        print(f"[Verify] ERROR — {e}")
        return False


def clips_pipeline(mapped_data, registry, OMNI_JSON, ROOT_OUTPUT, YOLO_VEHICLE_MODEL, YOLO_PEDESTRIAN_MODEL, target_dt, override_meta=None, action_choice=None, batch_mode=False, save_media=True, batch_dur=None):
    folder, frame_no, meta = get_target_frame(mapped_data, registry, ROOT_OUTPUT, target_dt, override_meta=override_meta, is_clip=True, batch_dur=batch_dur)
    
    if not folder: 
        return None, None 

    # BUG FIX: Ensure clip_duration_sec is actually in meta if provided by override_meta
    if override_meta and "clip_duration_sec" in override_meta:
        meta["clip_duration_sec"] = override_meta["clip_duration_sec"]
    
    # Fallback if duration is still missing
    if "clip_duration_sec" not in meta:
        meta["clip_duration_sec"] = 5.0
        
    if not action_choice:
        print("\n--- Video Clips Pipeline Options ---")
        print(" 1) Extract & Undistort Video Lenses")
        print(" 2) Extract & Stitch Panoramic Video")
        print(" 3) Extract, Stitch, & Detect Vehicles in Video")
        print(" 4) Extract, Stitch, Detect Vehicles & Pedestrians in Video")
        choice = UI.input("Select an option (1-4) [e.g., 2]: ")
    else:
        choice = str(action_choice)

    safe_label = get_exact_frame_label(meta, frame_no)
    dur_str = f"{meta.get('clip_duration_sec', 1.0):.1f}".replace(".", "p")
    out_dir = os.path.join(ROOT_OUTPUT, f"ClipPipeline_{safe_label}_D{dur_str}")
    
    raw_dir = os.path.join(out_dir, "RAW_TRIMMED")
    undist_dir = os.path.join(out_dir, "UNDISTORTED")
    intr_dir = os.path.join(out_dir, "INTRINSICS")
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(undist_dir, exist_ok=True)
    os.makedirs(intr_dir, exist_ok=True)
    
    undist_paths = [os.path.join(undist_dir, f"undistorted_lens{i}.mp4") for i in range(1, 7)]
    
    stitched_vid = os.path.join(out_dir, "Final_Stitch.mp4")
    veh_vid = os.path.join(out_dir, "Final_Stitch_vehicles.mp4")
    veh_json = os.path.join(out_dir, "vehicles.json")
    ped_vid = os.path.join(out_dir, "Final_Stitch_pedestrians.mp4")
    ped_json = os.path.join(out_dir, "pedestrians.json")

    do_undistort = True
    do_stitch = False
    do_veh = False
    do_ped = False

    if not batch_mode:
        print(f"\n{UI.CYAN}--- CHECKING EXISTING ARTIFACTS ---{UI.RESET}")
        
        if all(os.path.exists(p) for p in undist_paths):
            do_undistort = UI.ask_yes_no("Undistorted videos already exist for this clip. Overwrite?")
                
        if choice in ['2', '3', '4']:
            do_stitch = True
            if os.path.exists(stitched_vid):
                do_stitch = UI.ask_yes_no("Stitched video already exists for this clip. Overwrite?")

        if choice in ['3', '4']:
            do_veh = True
            if os.path.exists(veh_vid) and os.path.exists(veh_json):
                do_veh = UI.ask_yes_no("Vehicle detection already exists for this clip. Overwrite?")

        if choice == '4':
            do_ped = True
            if os.path.exists(ped_vid) and os.path.exists(ped_json):
                do_ped = UI.ask_yes_no("Pedestrian detection already exists for this clip. Overwrite?")
    else:
        do_undistort = not all(os.path.exists(p) for p in undist_paths) or not save_media
        if choice in ['2', '3', '4']: do_stitch = not os.path.exists(stitched_vid) or not save_media
        if choice in ['3', '4']: do_veh = not (os.path.exists(veh_vid) and os.path.exists(veh_json)) or not save_media
        if choice == '4': do_ped = not (os.path.exists(ped_vid) and os.path.exists(ped_json)) or not save_media

    if not batch_mode:
        print(f"\n{UI.CYAN}=== PIPELINE EXECUTION ==={UI.RESET}")

    # STEP 1: Trim & Undistort
    if do_undistort:
        if not batch_mode: UI.info("Step 1: Trimming and Undistorting Lenses (This may take time)...")
        undistort_start_time = time.perf_counter() 
        
        K_orig, D, xi, orig_size = load_omni_intrinsics(OMNI_JSON)
        play_time_sec = float(meta.get('playback_time_sec', float(meta['frame_no']) / float(meta['fps'])))

        # --- Trim all 6 lenses first (fast, stream-copy) ---
        PARALLEL_NVENC = _cfg("processing", "parallel_lenses", 1)
        trimmed = {}  # lens_index -> raw_vid path
        for i in range(1, 7):
            src_vid = os.path.join(folder, f"LENS{i}", f"video_lens{i}.mp4")
            raw_vid = os.path.join(raw_dir, f"lens{i}.mp4")
            if ClipVideoTrimmer.trim(src_vid, raw_vid, play_time_sec, meta['clip_duration_sec']):
                trimmed[i] = raw_vid

        # --- Get total frames from first trimmed video for progress bar ---
        sample_total = 0
        if trimmed:
            _tcap = cv2.VideoCapture(list(trimmed.values())[0])
            sample_total = int(_tcap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            _tcap.release()
        total_all_frames = sample_total * len(trimmed)

        pbar = tqdm(total=total_all_frames, desc="Step 1: Undistort",
                    unit="frame", dynamic_ncols=True) if total_all_frames > 0 else None

        # --- Worker function for one lens ---
        results_lock = threading.Lock()
        results = {}  # lens_index -> (K_new, dims)

        def _undistort_one_lens(lens_i):
            raw_vid = trimmed[lens_i]
            result = ClipVideoUndistorter.process(
                raw_vid, undist_paths[lens_i - 1],
                K_orig, D, xi, orig_size,
                pbar=pbar, lens_label=f"Lens {lens_i}")
            if result is not None:
                K_new, dims = result
                intr_dict = {"K": K_new, "D": [0]*5, "image_size": dims}
                intr_path = os.path.join(intr_dir, f"calibration_pinhole_lens{lens_i}.json")
                with results_lock:
                    with open(intr_path, "w") as jf:
                        json.dump(intr_dict, jf, indent=2)
                    results[lens_i] = (K_new, dims)

        # --- Process lenses in parallel waves of PARALLEL_NVENC ---
        lens_list = sorted(trimmed.keys())

        # V4: Load checkpoint — skip lenses already undistorted in a previous run
        checkpoint = _load_checkpoint(out_dir)
        lens_list_todo = [li for li in lens_list if str(li) not in checkpoint]
        if len(lens_list_todo) < len(lens_list):
            n_skip = len(lens_list) - len(lens_list_todo)
            print(f"[Checkpoint] Resuming — skipping {n_skip} already-completed lens(es).")

        for wave_start in range(0, len(lens_list_todo), PARALLEL_NVENC):
            wave = lens_list_todo[wave_start:wave_start + PARALLEL_NVENC]
            with ThreadPoolExecutor(max_workers=max(1, len(wave))) as executor:
                futures = {executor.submit(_undistort_one_lens, li): li for li in wave}
                for f in as_completed(futures):
                    li = futures[f]
                    f.result()  # re-raises any exception from the thread
                    checkpoint[str(li)] = undist_paths[li - 1]
                    _save_checkpoint(out_dir, checkpoint)

        _clear_checkpoint(out_dir)

        if pbar: pbar.close()

        meta["undistortion_time_sec"] = round(time.perf_counter() - undistort_start_time, 4)
        with open(os.path.join(out_dir, "clip_extraction_meta.json"), "w") as f: json.dump(meta, f, indent=4)
        if not batch_mode: UI.success("Undistortion Complete.")

    # STEP 2: Stitch
    if choice in ['2', '3', '4'] and do_stitch:
        if not batch_mode: UI.info("Step 2: Running SIFT Video Stitcher...")
        stitch_start_time = time.perf_counter()
        
        stitcher = ClipVideoStitcher(intrinsics_dir=intr_dir)

        # V4: Pipeline mode — undistort and stitch run concurrently via per-lens queues
        import queue as _queue_mod
        Q_SIZE = _cfg("processing", "pipeline_queue_size", 60)
        frame_queues = [_queue_mod.Queue(maxsize=Q_SIZE) for _ in range(6)]

        # Determine fps and total_frames from first undistorted file
        _pipe_fps = 29.97
        _pipe_total = 0
        if os.path.exists(undist_paths[0]):
            _pipe_cap = cv2.VideoCapture(undist_paths[0])
            _pipe_fps = _pipe_cap.get(cv2.CAP_PROP_FPS) or 29.97
            _pipe_total = int(_pipe_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            _pipe_cap.release()
        elif 'trimmed' in dir() and trimmed:
            _pipe_cap = cv2.VideoCapture(list(trimmed.values())[0])
            _pipe_fps = _pipe_cap.get(cv2.CAP_PROP_FPS) or 29.97
            _pipe_total = int(_pipe_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            _pipe_cap.release()

        # Load calibration if not already in scope from STEP 1
        if 'K_orig' not in dir():
            K_orig, D, xi, orig_size = load_omni_intrinsics(OMNI_JSON)

        def _undistort_to_queue(lens_idx):
            """Re-undistort lens (lens_idx+1) feeding frames to queue AND writing to disk."""
            li = lens_idx + 1
            ClipVideoUndistorter.process(
                input_path  = trimmed[li] if 'trimmed' in dir() and li in trimmed else undist_paths[lens_idx],
                output_path = undist_paths[lens_idx],
                K_orig      = K_orig,
                D           = D,
                xi          = xi,
                orig_size   = orig_size,
                fov_scale   = _cfg("processing", "fov_scale", 0.4),
                pbar        = None,
                lens_label  = f"Lens {li}",
                frame_queue = frame_queues[lens_idx],
            )

        with ThreadPoolExecutor(max_workers=6) as _exec:
            _undist_futures = [_exec.submit(_undistort_to_queue, i) for i in range(6)]
            stitch_success = stitcher.stitch_video_queued(
                frame_queues, stitched_vid, _pipe_fps, _pipe_total
            )
            for _fu in _undist_futures:
                _fu.result()

        if stitch_success:
            _verify_output(stitched_vid,
                           _cfg("output", "width", 1920),
                           _cfg("output", "height", 960))
            meta["stitching_time_sec"] = round(time.perf_counter() - stitch_start_time, 4)
            meta["input_videos"] = undist_paths
            meta["lens_zooms"] = stitcher.LENS_ZOOMS
            meta["final_orientations"] = [(float(y), float(p)) for y, p in stitcher.LENS_ORIENTATIONS]
            with open(os.path.join(out_dir, "clip_stitched_meta.json"), "w") as f: json.dump(meta, f, indent=4)
            if not batch_mode: UI.success("Stitching Complete.")
        else:
            UI.error("Stitching Failed.")
            return None, None

    veh_counts = {}
    if choice in ['3', '4']:
        if do_veh:
            if not batch_mode: UI.info("Step 3: Running YOLO Tracking (Vehicles)...")
            veh_counts = ClipVehicleDetector(YOLO_VEHICLE_MODEL).process_video(stitched_vid, veh_vid, veh_json, meta)
            if not batch_mode: UI.success("Vehicle Detection Complete.")
        else:
            if os.path.exists(veh_json):
                try:
                    with open(veh_json, 'r') as f: veh_counts = json.load(f).get("summary_counts", {})
                except Exception: pass

    ped_count = 0
    if choice == '4':
        if do_ped:
            if not batch_mode: UI.info("Step 4: Running YOLO Tracking (Pedestrians)...")
            ped_count = ClipPedestrianDetector(YOLO_PEDESTRIAN_MODEL).process_video(stitched_vid, veh_json, ped_vid, ped_json, meta)
            if not batch_mode: UI.success("Pedestrian Detection Complete.")
        else:
            if os.path.exists(ped_json):
                try:
                    with open(ped_json, 'r') as f: ped_count = json.load(f).get("total_pedestrians", 0)
                except Exception: pass

    if not batch_mode:
        UI.success(f"Video Pipeline Finished! Output at: {out_dir}")
        UI.pause()
        
    if not save_media and os.path.exists(out_dir): shutil.rmtree(out_dir, ignore_errors=True)
    return veh_counts, ped_count


# ==========================================
# 7. BATCH PROCESS HELPER (CRASH PROOF)
# ==========================================

def batch_process_csv(csv_path, ROOT_OUTPUT, HARD_DRIVE_ROOT, YOLO_VEHICLE, YOLO_PED, OMNI_JSON):
    if 'pd' not in globals():
        UI.error("Pandas library is required for Batch Processing. Please pip install pandas.")
        return
    
    try:
        if csv_path.endswith('.csv'): df = pd.read_csv(csv_path)
        else: df = pd.read_excel(csv_path)
    except Exception as e:
        UI.error(f"Failed to read file: {e}")
        return
        
    has_dur = 'duration' in [str(c).lower() for c in df.columns]
    ts_col = next((c for c in df.columns if str(c).lower() == 'timestamp'), None)
    if not ts_col: 
        UI.error("No 'timestamp' column found in the file.")
        return

    dur_col = next((c for c in df.columns if str(c).lower() == 'duration'), None) if has_dur else None

    print("\n--- Batch Pipeline Options ---")
    print(" 1) Extract Undistorted Lenses")
    print(" 2) Extract & Stitch Panorama")
    print(" 3) Extract, Stitch, & Detect Vehicles")
    print(" 4) Extract, Stitch, Detect Vehicles & Pedestrians")
    choice = UI.input("Select an option (1-4): ")
    
    save_media = UI.ask_yes_no("Do you want to save the media (.mp4/.jpg) and .json files to disk?")
    modify_csv = UI.ask_yes_no("Do you want to modify the input file to include detection counts?") if choice in ['3', '4'] else True
        
    if 'available' not in df.columns: df['available'] = ""
    if modify_csv:
        cols_to_add = ['CAR', 'AUTO', 'BUS', 'TRUCK', 'MOTORBIKE']
        if choice == '4': cols_to_add.append('PEDESTRIAN')
        for c in cols_to_add:
            if c not in df.columns: df[c] = ""

    out_name = csv_path.rsplit('.', 1)[0] + '_processed.csv'
    
    def safe_write_row(data_to_write, is_header=False):
        while True:
            try:
                if is_header: data_to_write.to_csv(out_name, index=False)
                else: pd.DataFrame([data_to_write]).to_csv(out_name, mode='a', header=False, index=False)
                break 
            except PermissionError:
                print(f"\n{UI.RED}{UI.BOLD}[BLOCKED] Windows denied access to save the file!{UI.RESET}")
                print(f"{UI.YELLOW}Is '{os.path.basename(out_name)}' open in Excel?{UI.RESET}")
                UI.input("Please close the file in Excel, then press ENTER to resume saving...")

    safe_write_row(df.head(0), is_header=True)
    skip_all_missing = False
    
    cached_registries = {}
    cached_mapped_data = {} 
    daily_stitcher_cache = {} 
    
    # --- IN-MEMORY OPTIMIZATION: LOAD AI ONCE ---
    print(f"\n{UI.CYAN}[INFO] Loading AI Models into RAM (This happens once)...{UI.RESET}")
    veh_detector = PanoramicImageDetector(YOLO_VEHICLE) if choice in ['3', '4'] else None
    ped_detector = PanoramicPedestrianDetector(YOLO_PED) if choice == '4' else None
    
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        ts_str = str(row_dict[ts_col])
        target_dt = parse_target_timestamp(ts_str)
        
        if not target_dt: 
            safe_write_row(row_dict); continue
        
        date_key = target_dt.strftime("%Y-%m-%d")
        
        UI.header(f"Processing Batch {idx+1}/{len(df)}", HARD_DRIVE_ROOT, ROOT_OUTPUT)
        print(f"Target String: {ts_str} -> Model Target: {target_dt}")
        UI.info("Phase 1: Checking Data Availability (IMU & Video Meta)...")
        
        csv_found, json_folder, session_meta = AssetDiscovery.search_for_timestamp(HARD_DRIVE_ROOT, target_dt, verbose=False)
        
        if not csv_found or not json_folder:
            row_dict['available'] = 'no'
            print(f"{UI.YELLOW}Data not found for {ts_str}.{UI.RESET}")
            if not skip_all_missing:
                ans = input(f"{UI.YELLOW}Skip? [yes/no/yes_all]: {UI.RESET}").strip().lower()
                if ans == 'yes_all': skip_all_missing = True
                elif ans != 'yes': break 
            safe_write_row(row_dict); continue
            
        row_dict['available'] = 'yes'
        
        # --- FIX 1: REGISTRY CACHING & MASTER MAPPING ---
        # Only build and parse if we haven't seen this folder/csv combo yet in this batch run
        cache_key = f"{csv_found}_{json_folder}"
        if cache_key not in cached_registries:
            registry = build_session_registry(os.path.dirname(json_folder))
            mapping_output_path = os.path.join(ROOT_OUTPUT, "Master_Mapping.csv")
            mapped_data = process_imu_gps_file(csv_found, registry, mapping_output_path)
            
            cached_registries[cache_key] = registry
            cached_mapped_data[cache_key] = mapped_data
        else:
            registry = cached_registries[cache_key]
            mapped_data = cached_mapped_data[cache_key]
        
        if not mapped_data: 
            row_dict['available'] = 'no'
            safe_write_row(row_dict); continue
        
        dur_val = float(row_dict[dur_col]) if dur_col else None
        
        if has_dur:
            res = clips_pipeline(mapped_data, registry, OMNI_JSON, ROOT_OUTPUT, YOLO_VEHICLE, YOLO_PED, target_dt, action_choice=choice, batch_mode=True, save_media=save_media, batch_dur=dur_val)
        else:
            res = frames_pipeline(mapped_data, registry, OMNI_JSON, ROOT_OUTPUT, YOLO_VEHICLE, YOLO_PED, target_dt, action_choice=choice, batch_mode=True, save_media=save_media, stitcher_cache=daily_stitcher_cache, date_key=date_key, veh_detector=veh_detector, ped_detector=ped_detector)
            
        if res == (None, None):
            row_dict['available'] = 'no'
            print(f"{UI.YELLOW}Pipeline aborted for {ts_str} (e.g., >3.0s gap or corrupted media). Flipped to 'no'.{UI.RESET}")
        else:
            veh_counts, ped_count = res
            
            # Make sure we actually have data to write
            if choice in ['3', '4'] and modify_csv:
                # Even if the count is empty (no vehicles detected), we write 0s to prevent NaNs
                if not veh_counts: veh_counts = {'car': 0, 'auto': 0, 'bus': 0, 'truck': 0, 'motorbike': 0}
                
                for k, v in veh_counts.items():
                    if k.upper() in row_dict:
                        row_dict[k.upper()] = v
                if choice == '4':
                    row_dict['PEDESTRIAN'] = ped_count if ped_count is not None else 0
                
        safe_write_row(row_dict)
        
    UI.success(f"Batch complete. File safely appended line-by-line to: {out_name}")
    UI.pause()

# ==========================================
# 8. MAIN HUB LOOP
# ==========================================

def main():
    OMNI_JSON = r"C:\viswak_MUMMAS_360degcamera\Insta360ImageAnalysis\INTRINSICS\calibration_omni.json"
    YOLO_VEHICLE_MODEL = r"C:\viswak_MUMMAS_360degcamera\Insta360ImageAnalysis\best.pt"
    YOLO_PEDESTRIAN_MODEL = r"C:\viswak_MUMMAS_360degcamera\Insta360ImageAnalysis\yolo11m.pt"
    
    ROOT_OUTPUT = UI.input_dir("Enter Master Output Directory (e.g., C:\\Results): ", create_if_missing=True)
    HARD_DRIVE_ROOT = UI.input_dir("Enter the Root Directory of the Hard Drive (e.g., D:\\): ") 

    while True:
        UI.header("360 CAMERA DATA RETRIEVAL HUB", HARD_DRIVE_ROOT, ROOT_OUTPUT)
        print(" 1) Batch Process via CSV/Excel")
        print(" 2) Single Process: By Timestamp (IST/UTC/Unix)")
        print(" 3) Single Process: By Folder Path & Time/Frame")
        print(f" {UI.RED}4) Quit{UI.RESET}\n")
        
        choice = UI.input("Select an option (1-4) [e.g., 2]: ")

        if choice == '1':
            csv_in = UI.input_file("Enter path to Batch CSV/Excel file: ")
            batch_process_csv(csv_in, ROOT_OUTPUT, HARD_DRIVE_ROOT, YOLO_VEHICLE_MODEL, YOLO_PEDESTRIAN_MODEL, OMNI_JSON)
            continue

        # Variables for Unified Processing (Options 2 & 3)
        folder_path = None
        target_dt = None
        registry = {}
        mapped_data = []
        fps = 29.97 # Standard Default
        f_no = 0
        start_ist_str = datetime.now().isoformat()

        if choice == '2':
            ts_str = UI.input("Enter Target Timestamp (IST, UTC, or Unix): ")
            target_dt = parse_target_timestamp(ts_str)
            if not target_dt:
                UI.error("Could not parse timestamp."); UI.pause(); continue
                
            csv_path, json_folder, _ = AssetDiscovery.search_for_timestamp(HARD_DRIVE_ROOT, target_dt)
            if not csv_path or not json_folder:
                UI.pause(); continue
                
            folder_path = json_folder
            registry = build_session_registry(os.path.dirname(json_folder))
            mapping_output_path = os.path.join(ROOT_OUTPUT, "Master_Mapping.csv")
            mapped_data = process_imu_gps_file(csv_path, registry, mapping_output_path)
            
            if not mapped_data:
                UI.error("Could not parse the IMU CSV successfully."); UI.pause(); continue
            
            # Find closest frame for Option 2
            f_no = mapped_data[0].get('frame_no', 0) if mapped_data else 0

        elif choice == '3':
            folder_path = UI.input_dir("Enter exact path to the target Session Folder: ")
            target_dt = None # No specific target timestamp for manual mode

            # Determine Start Point
            tf_choice = UI.input("Enter 'f' for Frame number, or 's' for Playback Seconds [f/s]: ").lower()
            if tf_choice == 'f':
                f_no = int(UI.input("Enter Frame Number: "))
            else:
                f_no = int(float(UI.input("Enter Playback Seconds: ")) * fps)

        elif choice == '4':
            UI.clear(); print(f"{UI.CYAN}Exiting. Goodbye!{UI.RESET}"); break
        else:
            continue

        # --- UNIFIED PROCESSOR FOR OPTIONS 2 & 3 ---

        # 1. Probe Session Metadata and Video Duration
        meta_path = os.path.join(folder_path, "session_meta.json")
        lens1_path = os.path.join(folder_path, "LENS1", "video_lens1.mp4")
        
        # Get exact FPS and Total Frames from file
        total_frames = 9000
        if os.path.exists(lens1_path):
            cap = cv2.VideoCapture(lens1_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        # Extract Start Time for metadata
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                m = json.load(f)
                if m.get("created_utc"):
                    try:
                        utc_dt = datetime.strptime(m["created_utc"], "%Y-%m-%dT%H:%M:%S.%fZ")
                    except ValueError:
                        try:
                            utc_dt = datetime.strptime(m["created_utc"], "%Y-%m-%dT%H:%M:%SZ")
                        except ValueError:
                            utc_dt = datetime.strptime(m["created_utc"].split('.')[0], "%Y-%m-%dT%H:%M:%S")
                    start_ist_str = (utc_dt.replace(tzinfo=timezone.utc) + timedelta(hours=5, minutes=30)).replace(tzinfo=None).isoformat()

        # 2. Decision: Image or Video
        p_choice = UI.input("Process as: 1) Image Frame  2) Video Clip : ")
        
        clip_dur = 0.0
        if p_choice == '2':
            # Calculate remaining time in video
            max_available_dur = max(0, (total_frames - f_no) / fps)
            
            while True:
                clip_dur = float(UI.input(f"Enter Clip Duration in seconds (Max available: {max_available_dur:.2f}s): "))
                if clip_dur > max_available_dur:
                    UI.warn(f"Requested {clip_dur}s exceeds file capacity.")
                    confirm = UI.input(f"Cap to {max_available_dur:.2f}s and continue? [y/n]: ").lower()
                    if confirm == 'y':
                        clip_dur = max_available_dur
                        break
                    else:
                        UI.info("Please enter a valid duration.")
                else:
                    break

        # 3. Final Metadata Construction
        override_meta = {
            "folder_path": folder_path,
            "frame_no": f_no,
            "fps": fps,
            "start_ist_str": start_ist_str,
            "source_folder": os.path.basename(folder_path),
            "ist_time": "Manual_Input",
            "is_manual": True,
            "playback_time_sec": f"{f_no / fps:.3f}",
            "clip_duration_sec": clip_dur,
            "imu_data": None,
            "utc_time": None,
            "unix_time": 0
        }
        
        # 4. Routing to Pipelines
        if p_choice == '1':
            frames_pipeline(mapped_data, registry, OMNI_JSON, ROOT_OUTPUT, YOLO_VEHICLE_MODEL, YOLO_PEDESTRIAN_MODEL, target_dt, override_meta=override_meta)
        else:
            clips_pipeline(mapped_data, registry, OMNI_JSON, ROOT_OUTPUT, YOLO_VEHICLE_MODEL, YOLO_PEDESTRIAN_MODEL, target_dt, override_meta=override_meta)

if __name__ == "__main__":
    main()