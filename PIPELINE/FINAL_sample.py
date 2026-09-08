import os
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
import re
import concurrent.futures
import queue
import threading
from multiprocessing import Manager
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
        """Forces the user to enter a valid directory path before proceeding."""
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
        """Forces the user to enter a valid file path before proceeding."""
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
        """Asks a yes/no question but defaults to YES if the user does not respond within 'timeout' seconds."""
        print(f"{UI.YELLOW}{UI.BOLD}? {msg} [y/n] (Auto-Yes in {timeout}s): {UI.RESET}", end="", flush=True)
        
        start_time = time.time()
        ans = ""
        
        if os.name == 'nt':
            # Windows non-blocking input
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
            # Unix/Mac non-blocking input
            while True:
                i, o, e = select.select([sys.stdin], [], [], 0.1)
                if i:
                    ans = sys.stdin.readline().strip()
                    break
                if time.time() - start_time > timeout:
                    print(f"\n{UI.MAGENTA}Timeout reached. Auto-selecting: YES{UI.RESET}")
                    return True

        ans = ans.strip().lower()
        if ans in ['n', 'no']: 
            return False
        
        # Default to Yes for 'y', 'yes', or any unrecognized input
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
            if w and h: 
                orig_size = [float(w), float(h)]
        elif isinstance(raw_size, (list, tuple)) and len(raw_size) >= 2:
            orig_size = [float(raw_size[0]), float(raw_size[1])]
            
    return K, D, xi, orig_size

def get_intrinsics_for_lens(lens_num, omni_json_path, is_optimized):
    """Dynamically loads per-lens intrinsics based on the date-routing logic."""
    base_dir = os.path.dirname(omni_json_path)
    json_filename = os.path.basename(omni_json_path)
    
    if is_optimized:
        json_num = lens_num
        
        # --- FALLBACK LOGIC FOR MISSING CALIBRATIONS ---
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
            
    # Legacy execution (or fallback if optimized files are missing)
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
        undistorted = cv2.omnidir.undistortImage(
            frame_rotated, K_scaled, D, xi_vec,
            cv2.omnidir.RECTIFY_PERSPECTIVE,
            Knew=K_new,
            new_size=(w, h)
        )
        return undistorted, K_new, w, h
    except cv2.error as e:
        return None, None, None, None

def parse_robust_datetime(ts_str):
    """Robustly parses timestamps across ISO formats, Unix timestamps, and date strings."""
    if not ts_str:
        return None
    if isinstance(ts_str, (int, float)):
        try:
            return datetime.fromtimestamp(ts_str, timezone.utc)
        except Exception:
            return None
    s = str(ts_str).strip()
    try:
        val = float(s)
        if 1000000000 <= val <= 2500000000:
            return datetime.fromtimestamp(val, timezone.utc)
    except ValueError:
        pass
    clean = s.replace("Z", "+00:00").replace("z", "+00:00")
    try:
        dt = datetime.fromisoformat(clean)
        return dt
    except Exception:
        pass
    clean = s.replace("T", " ").replace("Z", "").replace("z", "").strip()
    for fmt in [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S.%f",
        "%d-%m-%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S.%f",
        "%d/%m/%Y %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y%m%d_%H%M%S",
    ]:
        try:
            return datetime.strptime(clean, fmt)
        except ValueError:
            pass
    return None

def parse_utc_to_ist(utc_str):
    """Converts UTC / ISO timestamp string to IST datetime object."""
    if not utc_str:
        return None
    dt = parse_robust_datetime(utc_str)
    if not dt:
        return None
    if dt.tzinfo is not None:
        utc_dt = dt.astimezone(timezone.utc)
        ist_dt = utc_dt + timedelta(hours=5, minutes=30)
        return ist_dt.replace(tzinfo=None)
    else:
        return dt + timedelta(hours=5, minutes=30)

def parse_timestamp_from_run_name(folder_name):
    """Extracts timestamp from session folder name (e.g. run_20260131_121529_5450)."""
    m = re.search(r'run_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', str(folder_name))
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                            int(m.group(4)), int(m.group(5)), int(m.group(6)))
        except Exception:
            pass
    return None

def parse_target_timestamp(ts_str):
    """Dynamically detects and parses IST, UTC, and Unix timestamp strings to standard IST."""
    if not ts_str:
        return None
    ts_str = str(ts_str).strip()
    
    # Try Unix float
    try:
        unix_val = float(ts_str)
        if 1000000000 <= unix_val <= 2500000000:
            utc_dt = datetime.fromtimestamp(unix_val, timezone.utc)
            ist_dt = utc_dt + timedelta(hours=5, minutes=30)
            return ist_dt.replace(tzinfo=None)
    except ValueError:
        pass

    # Try ISO with timezone
    clean_iso = ts_str.replace("Z", "+00:00").replace("z", "+00:00")
    try:
        dt = datetime.fromisoformat(clean_iso)
        if dt.tzinfo is not None:
            utc_dt = dt.astimezone(timezone.utc)
            return (utc_dt + timedelta(hours=5, minutes=30)).replace(tzinfo=None)
    except Exception:
        pass

    # Try standard string formats
    clean_str = ts_str.replace("T", " ").replace("Z", "").replace("z", "").strip()
    for fmt in [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%d-%m-%Y %H:%M:%S.%f",
        "%d-%m-%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S.%f",
        "%d/%m/%Y %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y%m%d_%H%M%S"
    ]:
        try:
            return datetime.strptime(clean_str, fmt)
        except ValueError:
            pass

    return None

# ==========================================
# CLASS & LCZ CATEGORIZATION HELPERS
# ==========================================

def sanitize_folder_name(name):
    """
    Sanitizes class name for safe directory naming across Windows and POSIX filesystems.
    Strips illegal characters (< > : " / \\ | ? *) while preserving spaces, hyphens, and parentheses.
    """
    if not name:
        return "Unclassified"
    clean = re.sub(r'[<>:"/\\|?*]', '_', str(name)).strip()
    return clean if clean else "Unclassified"

def extract_class_name(row_dict):
    """
    Extracts class name (such as LCZ Class) from a row dictionary or mapping.
    Detects headers: 'LCZ Class', 'lcz_class', 'lcz', 'class', 'category',
    'class_name', 'classname', 'label', 'type'.
    """
    if not isinstance(row_dict, dict):
        return None
    for k, v in row_dict.items():
        k_clean = str(k).strip().lower()
        if k_clean in ['lcz class', 'lcz_class', 'lcz', 'class', 'category', 'class_name', 'classname', 'label', 'type']:
            val = str(v).strip()
            if val and val.lower() not in ['nan', 'none', '']:
                return val
    return None

def auto_detect_class_from_csvs(target_dt, candidate_csvs=None):
    """
    Attempts to auto-detect LCZ / class name for a given datetime from known CSV spreadsheets.
    """
    if candidate_csvs is None:
        candidate_csvs = [
            r"D:\MUMMAS\output_converted_timestamp_60sec.csv",
            r"D:\MUMMAS\test_camera_timestamps.csv",
            r"D:\MUMMAS\output_converted_timestamp_60sec_processed.csv",
            r"D:\MUMMAS\test_camera_timestamps_processed.csv"
        ]
    
    if not target_dt or 'pd' not in globals():
        return None

    target_str = target_dt.strftime("%Y-%m-%d %H:%M:%S")

    for c_path in candidate_csvs:
        if not c_path or not os.path.exists(c_path):
            continue
        try:
            df = pd.read_csv(c_path)
            cls_col = next((c for c in df.columns if str(c).strip().lower() in ['lcz class', 'lcz_class', 'lcz', 'class', 'category']), None)
            ts_col = next((c for c in df.columns if str(c).strip().lower() in ['timestamp', 'target_time', 'datetime', 'time']), None)
            if not cls_col or not ts_col:
                continue

            match = df[df[ts_col].astype(str).str.strip() == target_str]
            if not match.empty:
                val = match.iloc[0][cls_col]
                if pd.notna(val) and str(val).strip():
                    return str(val).strip()

            for _, r in df.iterrows():
                try:
                    r_dt = parse_target_timestamp(str(r[ts_col]))
                    if r_dt and abs((r_dt - target_dt).total_seconds()) <= 2.0:
                        val = r[cls_col]
                        if pd.notna(val) and str(val).strip():
                            return str(val).strip()
                except Exception:
                    pass
        except Exception:
            continue

    return None

def sync_to_class_folder(src_pipeline_dir, root_output, class_name):
    """
    Categorizes stitched panoramic and undistorted images from src_pipeline_dir into:
    root_output / Class Names / <Sanitized_Class_Name> / Panoramic / <Pipeline_Folder_Name> / Final_Static_Stitch.jpg
    root_output / Class Names / <Sanitized_Class_Name> / Undistorted / <Pipeline_Folder_Name> / undistorted_LENS*.jpg

    Strictly retains ONLY image files inside Panoramic and Undistorted folders.
    Excludes all JSON metadata, detection annotations, and INTRINSICS calibration files.
    Retains the exact same image and file names without any renaming.
    Uses hardlinks on NTFS when possible (0 disk space overhead) with copy fallback.
    """
    if not class_name or not os.path.exists(src_pipeline_dir):
        return None

    clean_class = sanitize_folder_name(class_name)
    folder_basename = os.path.basename(os.path.normpath(src_pipeline_dir))
    class_root = os.path.join(root_output, "Class Names", clean_class)
    
    pano_dir = os.path.join(class_root, "Panoramic", folder_basename)
    undist_dir = os.path.join(class_root, "Undistorted", folder_basename)

    # Stitched panorama file targets (raw stitched panorama only)
    pano_candidates = [
        "Final_Static_Stitch.jpg",
        "Final_Stitch.mp4",
        "Final_Static_Stitch.mp4"
    ]

    def _transfer_file(src, dst):
        if not os.path.exists(src):
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            if os.path.getsize(src) == os.path.getsize(dst):
                return
            try:
                os.remove(dst)
            except Exception:
                pass
        try:
            os.link(src, dst)
        except Exception:
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                UI.warn(f"Failed to copy {os.path.basename(src)} to class directory: {e}")

    # 1. Sync Panoramic stitched image(s)
    for p_name in pano_candidates:
        src_pano = os.path.join(src_pipeline_dir, p_name)
        if os.path.exists(src_pano):
            _transfer_file(src_pano, os.path.join(pano_dir, p_name))

    # 2. Sync Undistorted lens images (undistorted_LENS1..6.jpg or .mp4)
    src_undist_folder = os.path.join(src_pipeline_dir, "undistorted")
    if os.path.exists(src_undist_folder):
        for f in os.listdir(src_undist_folder):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4')):
                _transfer_file(os.path.join(src_undist_folder, f), os.path.join(undist_dir, f))
    else:
        for i in range(1, 7):
            for ext in ['.jpg', '.mp4', '.png']:
                u_name = f"undistorted_LENS{i}{ext}"
                src_u = os.path.join(src_pipeline_dir, u_name)
                if os.path.exists(src_u):
                    _transfer_file(src_u, os.path.join(undist_dir, u_name))

    return class_root

def categorize_existing_results(root_output, csv_path=None, verbose=True):
    """
    Retroactively scans root_output for unclassified 'Pipeline_*' directories,
    maps their timestamps to LCZ classes using reference CSV spreadsheets,
    and synchronizes them into categorized 'Class Names / <Class_Name> / Panoramic'
    and 'Class Names / <Class_Name> / Undistorted' folders with images only.
    """
    if not os.path.exists(root_output):
        if verbose:
            UI.error(f"Output directory does not exist: {root_output}")
        return {}

    candidate_csvs = []
    if csv_path and os.path.exists(csv_path):
        candidate_csvs.append(csv_path)
    candidate_csvs.extend([
        r"D:\MUMMAS\output_converted_timestamp_60sec.csv",
        r"D:\MUMMAS\test_camera_timestamps.csv",
        r"D:\MUMMAS\output_converted_timestamp_60sec_processed.csv",
        r"D:\MUMMAS\test_camera_timestamps_processed.csv"
    ])

    # Build timestamp -> class lookup map
    ts_map = {}
    if 'pd' in globals():
        for cp in candidate_csvs:
            if not cp or not os.path.exists(cp):
                continue
            try:
                df = pd.read_csv(cp)
                cls_col = next((c for c in df.columns if str(c).strip().lower() in ['lcz class', 'lcz_class', 'lcz', 'class', 'category']), None)
                ts_col = next((c for c in df.columns if str(c).strip().lower() in ['timestamp', 'target_time', 'datetime', 'time']), None)
                if cls_col and ts_col:
                    for _, row in df.iterrows():
                        t_val = str(row[ts_col]).strip()
                        c_val = str(row[cls_col]).strip()
                        if t_val and c_val and c_val.lower() not in ['nan', 'none', '']:
                            if t_val not in ts_map:
                                ts_map[t_val] = c_val
            except Exception:
                continue

    pipeline_dirs = [d for d in os.listdir(root_output) if d.startswith("Pipeline_") and os.path.isdir(os.path.join(root_output, d))]
    if verbose:
        print(f"\n{UI.CYAN}[INFO] Found {len(pipeline_dirs)} session folders to categorize in {root_output}...{UI.RESET}")

    categorized_counts = {}
    for d in pipeline_dirs:
        dir_path = os.path.join(root_output, d)
        matched_class = None

        # 1. Check existing static_stitched_meta.json
        meta_file = os.path.join(dir_path, "static_stitched_meta.json")
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r', encoding='utf-8') as mf:
                    meta_data = json.load(mf)
                    matched_class = meta_data.get("lcz_class") or meta_data.get("class_name")
                    if not matched_class and meta_data.get("ist_time"):
                        t_ist = meta_data.get("ist_time")
                        matched_class = ts_map.get(t_ist)
            except Exception:
                pass

        # 2. Match from directory timestamp
        if not matched_class:
            try:
                time_str = d.split('_F')[0].replace("Pipeline_", "")
                dt = datetime.strptime(time_str, "%Y-%m-%d_%H-%M-%S")
                formatted_ts = dt.strftime("%Y-%m-%d %H:%M:%S")
                matched_class = ts_map.get(formatted_ts)
                if not matched_class:
                    for t_key, c_val in ts_map.items():
                        try:
                            key_dt = parse_target_timestamp(t_key)
                            if key_dt and abs((key_dt - dt).total_seconds()) <= 2.0:
                                matched_class = c_val
                                break
                        except Exception:
                            pass
            except Exception:
                pass

        if matched_class:
            sync_to_class_folder(dir_path, root_output, matched_class)
            categorized_counts[matched_class] = categorized_counts.get(matched_class, 0) + 1
            # Update meta with class if missing
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, 'r', encoding='utf-8') as mf:
                        m_curr = json.load(mf)
                    if 'lcz_class' not in m_curr:
                        m_curr['lcz_class'] = matched_class
                        m_curr['class_name'] = matched_class
                        with open(meta_file, 'w', encoding='utf-8') as mf:
                            json.dump(m_curr, mf, indent=4)
                except Exception:
                    pass

    # Clean up old legacy top-level class folders directly in root_output if present
    known_legacy_classes = [
        "Compact low-rise (LCZ 3)", "Open low-rise (LCZ 6)", "Large low-rise (LCZ 8)",
        "Sparsely built (LCZ 9)", "Low plants (LCZ 14)", "TEST"
    ]
    for leg in known_legacy_classes:
        leg_dir = os.path.join(root_output, sanitize_folder_name(leg))
        if os.path.isdir(leg_dir):
            try:
                shutil.rmtree(leg_dir, ignore_errors=True)
            except Exception:
                pass

    if verbose:
        UI.success(f"Categorization Complete! Organized {sum(categorized_counts.values())}/{len(pipeline_dirs)} folders into 'Class Names':")
        for cls_name, cnt in sorted(categorized_counts.items()):
            print(f"  📁 Class Names/{cls_name}: {cnt} session(s) [Panoramic/ & Undistorted/]")

    return categorized_counts

class GPSResolver:
    """
    Indexes GPS coordinates from IMU/GPS CSV logs and provides
    fast spatial nearest-neighbor lookups to map (lat, lon) -> timestamp.
    """
    _index_cache = {}

    @staticmethod
    def haversine_distance_m(lat1, lon1, lat2, lon2):
        """Calculates great-circle distance between two points in meters."""
        R = 6371000.0 # Earth radius in meters
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return float(R * c)

    @classmethod
    def find_imu_csv_files(cls, base_dirs):
        if isinstance(base_dirs, str):
            base_dirs = [base_dirs]
        
        found_files = []
        valid_kw = {'imu', 'imu-gps', 'imugps', 'imu_gps', 'gps'}

        for b_dir in base_dirs:
            if not b_dir or not os.path.exists(b_dir):
                continue
            for root, dirs, files in os.walk(b_dir):
                root_parts = [p.lower() for p in os.path.normpath(root).split(os.sep)]
                is_imu_folder = any(kw in root_parts for kw in valid_kw)
                
                for f in files:
                    if not f.lower().endswith('.csv') or f.startswith('._') or f.startswith('~'):
                        continue
                    f_lower = f.lower()
                    if is_imu_folder or 'imu' in f_lower or 'gps' in f_lower:
                        found_files.append(os.path.join(root, f))
                        
        return list(set(found_files))

    @classmethod
    def build_index(cls, search_dirs=None, verbose=True):
        if search_dirs is None:
            dirs_to_check = []
        elif isinstance(search_dirs, str):
            dirs_to_check = [search_dirs]
        else:
            dirs_to_check = list(search_dirs)

        dirs_to_check = [os.path.normpath(d) for d in dirs_to_check if d and os.path.exists(d)]
        if not dirs_to_check:
            default_mummas = r"D:\MUMMAS\MUMMAS DATA COLLECTION"
            if os.path.exists(default_mummas):
                dirs_to_check.append(os.path.normpath(default_mummas))
            if os.path.exists(HARD_DRIVE_ROOT) and os.path.normpath(HARD_DRIVE_ROOT) not in dirs_to_check:
                dirs_to_check.append(os.path.normpath(HARD_DRIVE_ROOT))

        cache_key = tuple(sorted(dirs_to_check))
        if cache_key in cls._index_cache:
            return cls._index_cache[cache_key]

        csv_files = cls.find_imu_csv_files(dirs_to_check)
        if not csv_files:
            if verbose:
                UI.warn("No IMU GPS CSV files found in search directories.")
            return None

        if verbose:
            print(f"{UI.CYAN}[INFO] Indexing GPS coordinates from {len(csv_files)} IMU CSV file(s)...{UI.RESET}")

        all_coords = []
        all_timestamps = []
        all_sources = []

        for c_file in csv_files:
            try:
                # 1. Detect headers from file
                lat_col, lon_col, time_col, time_mode = None, None, None, 'local'
                with open(c_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line_s = line.strip()
                        if not line_s or line_s.startswith('#'):
                            continue
                        headers = [h.strip() for h in line_s.split(',')]
                        for c in headers:
                            clow = c.lower().strip()
                            if clow in ['gnss_latitude', 'filter_lla_lat', 'latitude', 'lat', 'y'] and not lat_col:
                                lat_col = c
                            elif clow in ['gnss_longitude', 'filter_lla_lon', 'longitude', 'lon', 'long', 'x'] and not lon_col:
                                lon_col = c
                            elif not time_col:
                                if any(k in clow for k in ['timestamp_local', 'local_timestamp', 'timestamp_ist', 'ist_timestamp']):
                                    time_col = c; time_mode = 'local'
                                elif any(k in clow for k in ['timestamp_utc', 'utc_timestamp', 'timestamp_gmt']):
                                    time_col = c; time_mode = 'utc'
                                elif any(k in clow for k in ['t_unix', 'unix_timestamp', 'epoch', 'unix_time']):
                                    time_col = c; time_mode = 'unix'
                                elif clow in ['timestamp', 'time']:
                                    time_col = c; time_mode = 'local'
                        break

                if not lat_col or not lon_col or not time_col:
                    continue

                # 2. Fast pandas ingestion if available
                used_pandas = False
                if 'pd' in globals():
                    try:
                        df = pd.read_csv(c_file, comment='#', usecols=[lat_col, lon_col, time_col], on_bad_lines='skip', low_memory=False)
                        df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
                        df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
                        df = df.dropna(subset=[lat_col, lon_col, time_col])
                        df = df[(df[lat_col] != 0.0) | (df[lon_col] != 0.0)]
                        
                        # Subsample high-frequency data down to ~5-10Hz to keep spatial index fast and lean
                        if len(df) > 5000:
                            df = df.iloc[::10]

                        lats = df[lat_col].to_numpy()
                        lons = df[lon_col].to_numpy()
                        raw_times = df[time_col].astype(str).tolist()

                        for lat_v, lon_v, r_time in zip(lats, lons, raw_times):
                            r_time = r_time.strip()
                            if time_mode == 'unix':
                                try:
                                    u_val = float(r_time)
                                    u_dt = datetime.fromtimestamp(u_val, timezone.utc)
                                    ist_str = (u_dt + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
                                except ValueError:
                                    continue
                            elif time_mode == 'utc' or r_time.endswith('Z') or r_time.endswith('z'):
                                try:
                                    clean_utc = r_time.replace("T", " ").replace("Z", "").replace("z", "").split('.')[0]
                                    u_dt = datetime.strptime(clean_utc, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                                    ist_str = (u_dt + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
                                except ValueError:
                                    continue
                            else:
                                ist_str = r_time.replace("T", " ").split('.')[0]

                            all_coords.append((float(lat_v), float(lon_v)))
                            all_timestamps.append(ist_str)
                            all_sources.append(c_file)
                        
                        used_pandas = True
                    except Exception:
                        used_pandas = False

                # 3. Fallback to standard csv DictReader
                if not used_pandas:
                    def skip_comments(f_obj):
                        for line in f_obj:
                            if line.strip() and not line.strip().startswith('#'):
                                yield line

                    with open(c_file, 'r', encoding='utf-8', errors='ignore') as f:
                        reader = csv.DictReader(skip_comments(f))
                        for row in reader:
                            raw_lat = row.get(lat_col, '').strip()
                            raw_lon = row.get(lon_col, '').strip()
                            raw_time = row.get(time_col, '').strip()

                            if not raw_lat or not raw_lon or not raw_time:
                                continue

                            try:
                                lat_val = float(raw_lat)
                                lon_val = float(raw_lon)
                                if lat_val == 0.0 and lon_val == 0.0:
                                    continue
                            except ValueError:
                                continue

                            if time_mode == 'unix':
                                try:
                                    u_val = float(raw_time)
                                    u_dt = datetime.fromtimestamp(u_val, timezone.utc)
                                    ist_str = (u_dt + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
                                except ValueError:
                                    continue
                            elif time_mode == 'utc' or raw_time.endswith('Z') or raw_time.endswith('z'):
                                try:
                                    clean_utc = raw_time.replace("T", " ").replace("Z", "").replace("z", "").split('.')[0]
                                    u_dt = datetime.strptime(clean_utc, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                                    ist_str = (u_dt + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
                                except ValueError:
                                    continue
                            else:
                                ist_str = raw_time.replace("T", " ").split('.')[0]

                            all_coords.append((lat_val, lon_val))
                            all_timestamps.append(ist_str)
                            all_sources.append(c_file)

            except Exception:
                continue

        if not all_coords:
            if verbose:
                UI.warn("No valid GPS fixes extracted from IMU files.")
            return None

        coords_arr = np.array(all_coords, dtype=np.float64)
        mean_lat = np.mean(coords_arr[:, 0])
        cos_lat = np.cos(np.radians(mean_lat))
        metric_proj = np.column_stack([coords_arr[:, 0] * 111320.0, coords_arr[:, 1] * 111320.0 * cos_lat])
        
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(metric_proj)
        except ImportError:
            tree = None

        index_data = {
            "coords": coords_arr,
            "metric_proj": metric_proj,
            "mean_lat": mean_lat,
            "cos_lat": cos_lat,
            "timestamps": all_timestamps,
            "sources": all_sources,
            "tree": tree
        }
        cls._index_cache[cache_key] = index_data
        if verbose:
            print(f"{UI.GREEN}[SUCCESS] Indexed {len(coords_arr)} GPS points across {len(csv_files)} IMU log(s).{UI.RESET}")
        return index_data

    @classmethod
    def lookup_gps(cls, search_dirs, target_lat, target_lon, max_dist_meters=200.0, verbose=True):
        idx = cls.build_index(search_dirs, verbose=verbose)
        if not idx:
            return None

        tree = idx["tree"]
        cos_lat = idx["cos_lat"]
        
        target_metric = np.array([target_lat * 111320.0, target_lon * 111320.0 * cos_lat])
        
        if tree is not None:
            dist_approx, nearest_i = tree.query(target_metric)
        else:
            diffs = idx["metric_proj"] - target_metric
            dists_sq = np.sum(diffs**2, axis=1)
            nearest_i = np.argmin(dists_sq)

        matched_lat = idx["coords"][nearest_i, 0]
        matched_lon = idx["coords"][nearest_i, 1]
        exact_dist_m = cls.haversine_distance_m(target_lat, target_lon, matched_lat, matched_lon)

        result = {
            "matched_timestamp": idx["timestamps"][nearest_i],
            "matched_lat": matched_lat,
            "matched_lon": matched_lon,
            "distance_meters": exact_dist_m,
            "source_file": idx["sources"][nearest_i],
            "is_within_tolerance": exact_dist_m <= max_dist_meters
        }
        return result

class AssetDiscovery:
    # 1. Cache for daily metadata.json (New Availability Logic)
    _metadata_cache = {}
    _date_to_dirs = defaultdict(set)
    
    # 2. Cache for the locations and time-boundaries of CSV/JSON files (Original Logic)
    _scan_cache = {
        "dirs_scanned": set(),
        "csv_bounds": [],   
        "json_bounds": []   
    }

    @classmethod
    def _parse_and_cache_session_meta(cls, j_path):
        folder = os.path.normpath(os.path.dirname(j_path))
        if any(jb['folder'] == folder for jb in cls._scan_cache["json_bounds"]):
            return
        try:
            with open(j_path, 'r', encoding='utf-8', errors='ignore') as f:
                meta = json.load(f)
            start_ist = None
            for k in ["created_utc", "start_time_utc", "start_utc", "start_time", "created_at", "timestamp"]:
                val = meta.get(k)
                if val:
                    start_ist = parse_utc_to_ist(val)
                    if start_ist:
                        break
            if not start_ist:
                start_ist = parse_timestamp_from_run_name(os.path.basename(folder))
            if start_ist:
                cls._scan_cache["json_bounds"].append({'folder': folder, 'start': start_ist, 'meta': meta})
        except Exception:
            pass

    @classmethod
    def _parse_and_cache_csv(cls, c_path):
        c_path = os.path.normpath(c_path)
        if any(cb['path'] == c_path for cb in cls._scan_cache["csv_bounds"]):
            return
        f_lower = os.path.basename(c_path).lower()
        if any(k in f_lower for k in ["processed", "batch", "mock", "master", "timestamp_metadata", "output_converted"]):
            return
        try:
            first_t = None
            last_t = None
            time_col_idx = -1
            with open(c_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.strip() and not line.strip().startswith('#'):
                        header = next(csv.reader([line.strip()]))
                        for i, col in enumerate(header):
                            c_name = col.lower().strip()
                            if 'timestamp_local' in c_name or 'local_timestamp' in c_name or c_name == 'timestamp':
                                time_col_idx = i
                                break
                        break
                if time_col_idx == -1:
                    return
                for line in f:
                    if line.strip() and not line.strip().startswith('#'):
                        cols = next(csv.reader([line.strip()]))
                        if len(cols) > time_col_idx:
                            first_t = cols[time_col_idx].strip().replace("T", " ").split('.')[0]
                        break
            if not first_t:
                return
            with open(c_path, 'rb') as f:
                f.seek(0, 2)
                file_size = f.tell()
                chunk_size = min(file_size, 4096)
                f.seek(file_size - chunk_size)
                last_chunk = f.read().decode('utf-8', errors='ignore').splitlines()
                for line in reversed(last_chunk):
                    if line.strip() and not line.strip().startswith('#'):
                        cols = next(csv.reader([line.strip()]))
                        if len(cols) > time_col_idx:
                            last_t = cols[time_col_idx].strip().replace("T", " ").split('.')[0]
                            break
            if first_t and last_t:
                st = datetime.strptime(first_t, "%Y-%m-%d %H:%M:%S")
                ed = datetime.strptime(last_t, "%Y-%m-%d %H:%M:%S")
                if st > ed:
                    st, ed = ed, st
                cls._scan_cache["csv_bounds"].append({'path': c_path, 'start': st, 'end': ed})
        except Exception:
            pass

    @classmethod
    def load_metadata_from_drive(cls, root_drive):
        """Indexes daily metadata.json files and pre-indexes any found session_meta and CSV files."""
        if not os.path.exists(root_drive):
            return
        print("Indexing daily metadata.json files...")
        skip_dirs = {'$recycle.bin', 'system volume information', '.git', '.venv', '__pycache__', 'test_output', 'results'}
        for root, dirs, files in os.walk(root_drive):
            dirs[:] = [d for d in dirs if d.lower() not in skip_dirs and not d.lower().startswith('lens')]
            for f in files:
                f_lower = f.lower()
                if f_lower == "metadata.json":
                    try:
                        with open(os.path.join(root, f), 'r', encoding='utf-8', errors='ignore') as jf:
                            meta = json.load(jf)
                            raw_date = str(meta.get('date', ''))
                            norm_date = raw_date.replace("-", "").replace("/", "").replace("_", "")
                            if norm_date:
                                cls._metadata_cache[norm_date] = meta
                                cls._date_to_dirs[norm_date].add(root)
                    except Exception as e:
                        print(f"Error loading metadata in {root}: {e}")
                elif f_lower in ["session_meta.json", "meta.json", "session_metadata.json"]:
                    cls._parse_and_cache_session_meta(os.path.join(root, f))
                elif f_lower.endswith('.csv'):
                    cls._parse_and_cache_csv(os.path.join(root, f))
        print(f"Indexed {len(cls._metadata_cache)} daily metadata files, {len(cls._scan_cache['json_bounds'])} session_meta files.")

    @classmethod
    def find_candidate_dirs(cls, root_drive, target_dt=None):
        """Finds all potential candidate directories containing session_meta.json and sensor data."""
        candidate_dirs = set()
        if not os.path.exists(root_drive):
            return list(candidate_dirs)
        candidate_dirs.add(root_drive)
        date_patterns = set()
        if target_dt:
            date_patterns.add(target_dt.strftime("%d%m%Y"))
            date_patterns.add(target_dt.strftime("%m%d%Y"))
            date_patterns.add(target_dt.strftime("%Y%m%d"))
            date_patterns.add(target_dt.strftime("%Y-%m-%d"))
            date_patterns.add(target_dt.strftime("%d-%m-%Y"))
            for dp in date_patterns:
                norm = dp.replace("-", "").replace("/", "").replace("_", "")
                if norm in cls._date_to_dirs:
                    for d in cls._date_to_dirs[norm]:
                        candidate_dirs.add(d)
        for sub in ["All date all other sensors", "All other All dates", "MUMMAS DATA COLLECTION", "MUMMAS"]:
            p = os.path.join(root_drive, sub)
            if os.path.isdir(p):
                candidate_dirs.add(p)
        for dp in date_patterns:
            p = os.path.join(root_drive, dp)
            if os.path.isdir(p):
                candidate_dirs.add(p)
            for sub in ["MUMMAS", "MUMMAS DATA COLLECTION", os.path.join("MUMMAS", "MUMMAS DATA COLLECTION")]:
                p_sub = os.path.join(root_drive, sub, dp)
                if os.path.isdir(p_sub):
                    candidate_dirs.add(p_sub)
        try:
            for item in os.listdir(root_drive):
                item_path = os.path.join(root_drive, item)
                if os.path.isdir(item_path):
                    item_lower = item.lower()
                    if item_lower in {'$recycle.bin', 'system volume information', '.git', '.venv', '__pycache__'}:
                        continue
                    if item in date_patterns or any(dp in item for dp in date_patterns):
                        candidate_dirs.add(item_path)
                    try:
                        for sub_item in os.listdir(item_path):
                            sub_path = os.path.join(item_path, sub_item)
                            if os.path.isdir(sub_path):
                                if sub_item in date_patterns or any(dp in sub_item for dp in date_patterns):
                                    candidate_dirs.add(sub_path)
                                elif sub_item.lower() in ["camera", "imu gps", "imu", "all date all other sensors"]:
                                    candidate_dirs.add(sub_path)
                    except Exception:
                        pass
        except Exception:
            pass
        return [os.path.normpath(d) for d in candidate_dirs if os.path.exists(d)]

    @classmethod
    def pre_scan_directories(cls, search_dirs):
        """Scans directories to build an O(1) searchable cache of CSV and JSON metadata."""
        dirs_to_scan = [d for d in search_dirs if d not in cls._scan_cache["dirs_scanned"]]
        if not dirs_to_scan:
            return
        skip_dirs = {'$recycle.bin', 'system volume information', '.git', '.venv', '__pycache__', 'test_output', 'results'}
        for s_dir in dirs_to_scan:
            if not os.path.exists(s_dir):
                cls._scan_cache["dirs_scanned"].add(s_dir)
                continue
            for root, dirs, files in os.walk(s_dir):
                dirs[:] = [d for d in dirs if d.lower() not in skip_dirs and not d.lower().startswith('lens')]
                for f in files:
                    f_lower = f.lower()
                    if f_lower in ["session_meta.json", "meta.json", "session_metadata.json"]:
                        cls._parse_and_cache_session_meta(os.path.join(root, f))
                    elif f_lower.endswith('.csv'):
                        cls._parse_and_cache_csv(os.path.join(root, f))
            cls._scan_cache["dirs_scanned"].add(s_dir)

    @classmethod
    def search_for_timestamp(cls, root_drive, target_dt, verbose=True):
        """
        Unified search that locates session_meta.json, camera frames, and IMU-GPS CSV,
        with multi-tier discovery and flexible date/path matching.
        """
        if not cls._metadata_cache and os.path.exists(root_drive):
            cls.load_metadata_from_drive(root_drive)
        candidate_dirs = cls.find_candidate_dirs(root_drive, target_dt)
        cls.pre_scan_directories(candidate_dirs)

        # --- PART 1: Strict 'availability' & frame_count check from daily metadata.json ---
        date_keys = [
            target_dt.strftime('%d%m%Y'),
            target_dt.strftime('%Y%m%d'),
            target_dt.strftime('%m%d%Y'),
            target_dt.strftime('%Y-%m-%d').replace('-', '')
        ]
        daily_meta = None
        for dk in date_keys:
            if dk in cls._metadata_cache:
                daily_meta = cls._metadata_cache[dk]
                break

        valid_run_id = None
        metadata_imu_path = None

        if daily_meta:
            target_naive = target_dt.replace(tzinfo=None)
            found_valid_run = False
            metadata_fail_reason = "timestamp outside of run windows"
            for trip_key, runs in daily_meta.get('trips', {}).items():
                for run in runs:
                    try:
                        start_ist = datetime.strptime(run['start_time_ist'], '%Y-%m-%d %H:%M:%S')
                        end_ist = datetime.strptime(run['end_time_ist'], '%Y-%m-%d %H:%M:%S')
                        if start_ist <= target_naive <= end_ist:
                            camera_data = run.get('camera', {})
                            if camera_data.get('available') != "available":
                                return None, None, None, f"camera {camera_data.get('available', 'N/A')}"
                            for i in range(1, 7):
                                raw_count = camera_data.get(f"lens{i}_frame_count", 0)
                                try: count = int(raw_count)
                                except (ValueError, TypeError): count = 0
                                if count <= 0:
                                    return None, None, None, f"lens{i} has {count} frames"
                            found_valid_run = True
                            valid_run_id = run.get('run_id')
                            metadata_imu_path = run.get('imu-gps')
                            break
                    except Exception:
                        continue
                if found_valid_run:
                    break
            if not found_valid_run:
                return None, None, None, metadata_fail_reason

        # --- PART 2: Physical file location mapping ---
        best_json_folder = None
        best_meta = None
        best_csv = None

        if valid_run_id:
            for jb in cls._scan_cache["json_bounds"]:
                if valid_run_id in jb['folder'] or os.path.basename(jb['folder']) == valid_run_id:
                    best_json_folder = jb['folder']
                    best_meta = jb['meta']
                    break

        if not best_json_folder:
            min_json_diff = float('inf')
            same_day = [jb for jb in cls._scan_cache["json_bounds"] if jb['start'].date() == target_dt.date()]
            pool = same_day if same_day else cls._scan_cache["json_bounds"]
            for jb in pool:
                diff = abs((jb['start'] - target_dt).total_seconds())
                if diff < min_json_diff:
                    min_json_diff = diff
                    best_json_folder = jb['folder']
                    best_meta = jb['meta']

        if not best_json_folder:
            return None, None, None, "Missing session_meta.json"

        # Resolve IMU-GPS CSV
        if metadata_imu_path:
            if os.path.exists(metadata_imu_path):
                best_csv = metadata_imu_path
            else:
                fname = os.path.basename(metadata_imu_path)
                for c_dir in candidate_dirs:
                    for sub in ["IMU GPS", "IMU", "imu", "imu_gps", ""]:
                        test_p = os.path.join(c_dir, sub, fname) if sub else os.path.join(c_dir, fname)
                        if os.path.exists(test_p):
                            best_csv = os.path.normpath(test_p)
                            break
                    if best_csv: break

        if not best_csv:
            min_csv_diff = float('inf')
            for cb in cls._scan_cache["csv_bounds"]:
                if cb['start'] <= target_dt <= cb['end']:
                    diff = 0
                else:
                    diff = min(abs((cb['start'] - target_dt).total_seconds()), abs((cb['end'] - target_dt).total_seconds()))
                if diff < min_csv_diff and diff <= 60 * 60:
                    min_csv_diff = diff
                    best_csv = cb['path']

        if not best_csv:
            curr = best_json_folder
            for _ in range(4):
                curr = os.path.dirname(curr)
                if not curr: break
                for imu_sub in ["IMU GPS", "IMU", "imu", "imu_gps"]:
                    imu_dir = os.path.join(curr, imu_sub)
                    if os.path.isdir(imu_dir):
                        for f in os.listdir(imu_dir):
                            if f.lower().endswith('.csv'):
                                best_csv = os.path.join(imu_dir, f)
                                break
                    if best_csv: break
                if best_csv: break

        if not best_csv:
            return None, None, None, "Missing IMU/GPS CSV"

        # Check physical video lenses
        missing_lenses = []
        for i in range(1, 7):
            lens_dir = os.path.join(best_json_folder, f"LENS{i}")
            if not os.path.isdir(lens_dir):
                lens_dir = os.path.join(best_json_folder, f"lens{i}")
            has_vid = False
            if os.path.isdir(lens_dir):
                for vf in os.listdir(lens_dir):
                    if vf.lower().endswith('.mp4'):
                        has_vid = True
                        break
            if not has_vid:
                missing_lenses.append(i)

        if missing_lenses:
            return None, None, None, f"Missing Lenses: {missing_lenses}"

        return best_csv, best_json_folder, best_meta, "OK"
    
def build_session_registry(base_root_dir):
    registry = []
    if not os.path.exists(base_root_dir):
        return registry
        
    for root_path, dirs, files in os.walk(base_root_dir):
        dirs[:] = [d for d in dirs if not d.lower().startswith('lens')]
        for f in files:
            if f.lower() in ["session_meta.json", "meta.json", "session_metadata.json"]:
                try:
                    with open(os.path.join(root_path, f), 'r', encoding='utf-8', errors='ignore') as jf: 
                        meta = json.load(jf)
                        
                    start_ist = None
                    for k in ["created_utc", "start_time_utc", "start_utc", "start_time", "created_at", "timestamp"]:
                        val = meta.get(k)
                        if val:
                            start_ist = parse_utc_to_ist(val)
                            if start_ist:
                                break
                    if not start_ist:
                        start_ist = parse_timestamp_from_run_name(os.path.basename(root_path))
                    if not start_ist:
                        continue
                    
                    vid_path = None
                    for l1_name in ["LENS1", "lens1"]:
                        l1_dir = os.path.join(root_path, l1_name)
                        if os.path.isdir(l1_dir):
                            for vf in os.listdir(l1_dir):
                                if vf.lower().endswith('.mp4'):
                                    vid_path = os.path.join(l1_dir, vf)
                                    break
                        if vid_path: break
                    
                    fps = 29.97 # Standard Default
                    frames = 9000
                    if vid_path and os.path.exists(vid_path):
                        cap = cv2.VideoCapture(vid_path)
                        if cap.isOpened():
                            fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
                            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 9000
                            cap.release()
                        
                    if frames == 0: 
                        continue 
                    
                    registry.append({
                        "folder_path": os.path.normpath(root_path), 
                        "folder_name": os.path.basename(root_path),
                        "start_ist": start_ist, 
                        "end_ist": start_ist + timedelta(seconds=frames/fps), 
                        "fps": fps, 
                        "total_frames": frames
                    })
                except Exception as e: 
                    UI.warn(f"Error reading registry for {root_path}: {e}")
                break
                
    return registry


def process_imu_gps_file(csv_path, registry, master_mapping_path, lcz_class=None):
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
                
            # Identify time column and its timezone/epoch nature
            time_col = None
            time_mode = 'local' # 'local', 'utc', or 'unix'

            for c in reader.fieldnames:
                clow = c.lower() if c else ""
                if 'timestamp_local' in clow or 'local_timestamp' in clow or 'timestamp_ist' in clow or 'ist_timestamp' in clow:
                    time_col = c
                    time_mode = 'local'
                    break
                elif 'timestamp_utc' in clow or 'utc_timestamp' in clow or 'timestamp_gmt' in clow:
                    time_col = c
                    time_mode = 'utc'
                    break
                elif 't_unix' in clow or 'unix_timestamp' in clow or 'timestamp_unix' in clow or 'unix_time' in clow or clow == 'epoch':
                    time_col = c
                    time_mode = 'unix'
                    break
                elif clow in ['timestamp', 'time']:
                    time_col = c
                    time_mode = 'local'
                    break

            if not time_col: 
                UI.error(f"Timestamp column missing. Found: {reader.fieldnames}")
                return None
            
            seen_in_this_file = set()
            for row in reader:
                time_str = row[time_col].strip()
                if not time_str: 
                    continue
                
                ist_dt = None
                utc_dt = None

                # Determine if numeric unix timestamp
                is_unix = time_mode == 'unix'
                if not is_unix:
                    try:
                        u_val = float(time_str)
                        if 1000000000 <= u_val <= 2500000000:
                            is_unix = True
                    except ValueError:
                        pass

                if is_unix:
                    try:
                        unix_val = float(time_str)
                        utc_dt = datetime.fromtimestamp(unix_val, timezone.utc)
                        ist_dt = (utc_dt + timedelta(hours=5, minutes=30)).replace(tzinfo=None)
                    except ValueError:
                        continue
                elif time_mode == 'utc' or time_str.endswith('Z') or time_str.endswith('z'):
                    try:
                        clean_utc = time_str.replace("T", " ").replace("Z", "").replace("z", "").split('.')[0]
                        utc_dt = datetime.strptime(clean_utc, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                        ist_dt = (utc_dt + timedelta(hours=5, minutes=30)).replace(tzinfo=None)
                    except ValueError:
                        continue
                else: # Default: Local IST
                    try:
                        clean_time_str = time_str.replace("T", " ").split('.')[0]
                        ist_dt = datetime.strptime(clean_time_str, "%Y-%m-%d %H:%M:%S")
                        utc_dt = (ist_dt - timedelta(hours=5, minutes=30)).replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue

                clean_time_str = ist_dt.strftime("%Y-%m-%d %H:%M:%S")
                if clean_time_str in seen_in_this_file:
                    continue
                    
                # --- STRICT SESSION MATCHING ---
                best_session = None
                min_s_diff = float('inf')
                
                for s in registry:
                    # 1. Exact mathematical match inside the video boundaries
                    if s["start_ist"] <= ist_dt <= s["end_ist"]:
                        best_session = s
                        break
                    
                    # 2. Closest match if slightly outside (e.g. video started 2 secs late)
                    diff = abs((s["start_ist"] - ist_dt).total_seconds())
                    if diff < min_s_diff and diff <= 15 * 60:
                        min_s_diff = diff
                        best_session = s
                        
                if not best_session: 
                    continue
                # -------------------------------
                    
                elapsed_sec = (ist_dt - best_session["start_ist"]).total_seconds()
                if elapsed_sec < 0: elapsed_sec = 0.0
                
                mapped_data.append({
                    "ist_time": clean_time_str, 
                    "utc_time": utc_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "unix_time": f"{utc_dt.timestamp():.3f}", 
                    "source_folder": best_session["folder_name"],
                    "folder_path": best_session["folder_path"], 
                    "playback_time_sec": f"{elapsed_sec:.3f}",
                    "frame_no": int(elapsed_sec * best_session["fps"]),
                    "lcz_class": lcz_class or ""
                })
                seen_in_this_file.add(clean_time_str)
                
        if not mapped_data:
            UI.warn("CSV was read, but no timestamps matched the available video clips.")
            return None
            
        os.makedirs(os.path.dirname(os.path.abspath(master_mapping_path)), exist_ok=True)
        file_exists = os.path.exists(master_mapping_path)
        existing_keys = set()
        
        if file_exists:
            try:
                with open(master_mapping_path, 'r', encoding='utf-8') as f:
                    for r in csv.DictReader(f):
                        if 'ist_time' in r:
                            existing_keys.add((r['ist_time'], r.get('source_folder', '')))
            except Exception: pass
            
        new_rows = [r for r in mapped_data if (r['ist_time'], r.get('source_folder', '')) not in existing_keys]
        
        if new_rows:
            has_lcz_in_file = False
            if file_exists:
                try:
                    with open(master_mapping_path, 'r', encoding='utf-8') as f:
                        hdr = f.readline()
                        has_lcz_in_file = 'lcz_class' in hdr
                except Exception:
                    pass
            fieldnames = ["ist_time", "utc_time", "unix_time", "source_folder", "folder_path", "playback_time_sec", "frame_no"]
            if not file_exists or has_lcz_in_file:
                fieldnames.append("lcz_class")

            with open(master_mapping_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
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
    """Scouts a video sequentially to find the sharpest frame deterministically."""
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
    
    # Seek exactly ONCE to the base frame to ensure deterministic decoding
    cap.set(cv2.CAP_PROP_POS_FRAMES, base_frame)
    
    current_frame = base_frame
    
    # Read sequentially through the offsets
    for _ in range(max_offset + 1):
        ret, frame = cap.read()
        if not ret: 
            break
            
        # Only calculate sharpness if the current sequential frame matches one of our targets
        offset_val = current_frame - base_frame
        if offset_val in offsets:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Rounding to 3 decimals eliminates microscopic floating-point inconsistencies
            sharpness = round(sharpness, 3)
            
            if sharpness > max_sharpness: 
                max_sharpness = sharpness
                best_frame = current_frame
                
        current_frame += 1
                
    cap.release()
    return best_frame


def get_exact_frame_label(meta, f_num):
    """Dynamically calculates the exact local timestamp of any frame based on the video's start time."""
    if "start_ist_str" in meta and "fps" in meta:
        start_ist = datetime.fromisoformat(meta["start_ist_str"])
        exact_time = start_ist + timedelta(seconds=f_num / meta["fps"])
        return f"{exact_time.strftime('%Y-%m-%d_%H-%M-%S')}_F{f_num}"
    else:
        # Fallback safety if metadata is missing
        safe_time = meta.get("ist_time", "Unknown").replace(":", "-").replace(" ", "_")
        return f"{safe_time}_F{f_num}"

def get_target_frame(mapped_data, registry, root_output, target_dt, override_meta=None, is_clip=False, batch_dur=None):
    if override_meta is not None:
        return override_meta["folder_path"], int(override_meta["frame_no"]), override_meta
    
    if batch_dur is None:
        print(f"\n{UI.CYAN}--- Target Mapping ---{UI.RESET}")
    
    match = None
    min_diff = float('inf')
    
    # ----------------------------------------------------
    # FIX: Isolate the TIME to ignore the DD/MM bug offset
    # ----------------------------------------------------
    for r in mapped_data:
        try:
            r_dt = datetime.strptime(r["ist_time"], "%Y-%m-%d %H:%M:%S")
            # Force the parsed date to perfectly match our requested target date so only the TIME difference is measured
            r_dt_healed = r_dt.replace(year=target_dt.year, month=target_dt.month, day=target_dt.day)
            
            diff = abs((r_dt_healed - target_dt).total_seconds())
            if diff < min_diff:
                min_diff = diff
                match = r
        except Exception:
            continue
            
    if min_diff > 3.0 or match is None:
        if batch_dur is None: UI.error(f"Exact or close timestamp not found. Min time gap: {min_diff}s")
        return None, None, None

    if min_diff > 0 and batch_dur is None: 
        UI.warn(f"Exact second not found. Snapping to nearest sensor reading ({match['ist_time']}) off by {min_diff:.1f}s")
        
    if batch_dur is None:
        UI.info(f"Target locked! Source: {match['source_folder']} | Base Frame: {match['frame_no']}")
    
    session = next((s for s in registry if s["folder_name"] == match["source_folder"]), None)
    if not session:
        if batch_dur is None: UI.error("Session data missing from registry.")
        return None, None, None
        
    meta = dict(match)
    meta["start_ist_str"] = session["start_ist"].isoformat()
    meta["fps"] = session["fps"]
    total_frames = session["total_frames"]
    
    if "playback_time_sec" not in meta and "fps" in meta:
        meta["playback_time_sec"] = f"{int(meta['frame_no']) / meta['fps']:.3f}"
        
    folder_path = match["folder_path"]
    frame_no = int(match["frame_no"])

    if is_clip:
        if batch_dur is not None:
            max_available_duration = (total_frames - frame_no) / session["fps"]
            meta["clip_duration_sec"] = min(float(batch_dur), max_available_duration)
        else:
            while True:
                try:
                    dur = float(UI.input("\nEnter duration of clip in seconds (e.g., 5.0): "))
                    max_available_duration = (total_frames - frame_no) / session["fps"]
                    
                    if dur > max_available_duration:
                        print(f"{UI.YELLOW}Warning: Only {max_available_duration:.1f} seconds available from this timestamp to the end of the video.{UI.RESET}")
                        if UI.ask_yes_no(f"Do you want to clip it using the available {max_available_duration:.1f}s duration?"):
                            meta["clip_duration_sec"] = max_available_duration
                            break
                        else:
                            continue 
                    else:
                        meta["clip_duration_sec"] = dur
                        break
                except ValueError:
                    UI.error("Invalid duration. Please enter a number.")

    return folder_path, frame_no, meta


# ==========================================
# 4A. STATIC IMAGE STITCHER CLASSES (Date Routed)
# ==========================================

import os
import json
import numpy as np
import cv2
import glob
import time

# ==========================================
# 1. CORE STITCHER CLASS
# ==========================================
class OptimizedImageStitcher:
    def __init__(self, intrinsics_dir):
        self.intrinsics_dir = intrinsics_dir
        self.W = 1920
        self.H = 960
        self.CROP_TOP = 220      # Scaled down from 331
        self.CROP_BOTTOM = 306   # Scaled down from 530
        
        # --- BASELINE ZOOMS & ORIENTATIONS ---
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
        self.base_Ks = [] 
        self.Ks = []      
        self.img_dims = []
        self.final_errors = [0.0] * 6 

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

    def find_shift_with_sift(self, img1, img2):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
            return None
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
                
        if len(good) < 4:
            return None
            
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
                if shift_px is None:
                    continue
                    
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

        print("\n[INFO] Blending images into final panorama (Optimized Geometry)...")
        total_img = np.zeros((self.H, self.W, 3), dtype=np.float32)
        total_weight = np.zeros((self.H, self.W), dtype=np.float32)
        
        for i in range(6):
            warped = cv2.remap(frames[i], self.maps_x[i], self.maps_y[i], cv2.INTER_LINEAR)
            w = self.weights[i]
            w_3c = np.dstack([w, w, w])
            total_img += warped.astype(np.float32) * w_3c
            total_weight += w
        
        total_weight[total_weight == 0] = 1.0
        total_weight_3c = np.dstack([total_weight, total_weight, total_weight])
        final_pano = total_img / total_weight_3c
        final_pano = np.clip(final_pano, 0, 255).astype(np.uint8)
        
        # --- APPLY CROP BEFORE RESIZE ---
        final_pano = final_pano[self.CROP_TOP : self.H - self.CROP_BOTTOM, :]
        
        # --- NEW: Apply the final 1920x960 resize ---
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
        self.CROP_TOP = 191      # Scaled down from 331
        self.CROP_BOTTOM = 306   # Scaled down from 530
                
        self.LENS_ZOOMS = [1.078, 1.127, 1.105, 1.044, 1.118, 1.08]
        self.LENS_ORIENTATIONS = [
            (-60, 0), (-120, 0), (180, 0), (120, 0), (60, 0), (0, 0)       
        ]
        
        self.maps_x = []
        self.maps_y = []
        self.weights = []
        self.Ks = []
        self.img_dims = []
        self.final_errors = [0.0] * 6 

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

    def find_shift_with_sift(self, img1, img2):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5: return 0
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
                
        if len(good) < 4: return 0
            
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        
        dx = pts1[:, 0] - pts2[:, 0]
        dy = pts1[:, 1] - pts2[:, 1]
        
        valid_matches = np.abs(dy) < 10 
        if np.sum(valid_matches) < 3: return 0
        
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
        
        print("\n[INFO] Blending images into final panorama...")
        total_img = np.zeros((self.H, self.W, 3), dtype=np.float32)
        total_weight = np.zeros((self.H, self.W), dtype=np.float32)
        
        for i in range(6):
            warped = cv2.remap(frames[i], self.maps_x[i], self.maps_y[i], cv2.INTER_LINEAR)
            w = self.weights[i]
            w_3c = np.dstack([w, w, w])
            total_img += warped.astype(np.float32) * w_3c
            total_weight += w
        
        total_weight[total_weight == 0] = 1.0
        total_weight_3c = np.dstack([total_weight, total_weight, total_weight])
        final_pano = total_img / total_weight_3c
        final_pano = np.clip(final_pano, 0, 255).astype(np.uint8)
        
        # --- APPLY CROP BEFORE RESIZE ---
        final_pano = final_pano[self.CROP_TOP : self.H - self.CROP_BOTTOM, :]
        
        # --- NEW: Apply the final 1920x960 resize ---
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
        
        # --- CONFIGURATION ---
        self.W = 1920
        self.H = 960
        self.CROP_TOP = 220      # Scaled down from 331
        self.CROP_BOTTOM = 306   # Scaled down from 335
        
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
        self.base_Ks = [] 
        self.Ks = []      
        self.img_dims = []
        self.final_errors = [0.0] * 6 

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

    def find_shift_with_sift(self, img1, img2):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
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

    # --- CHANGED: Now takes 'frames' instead of 'caps' ---
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
                    if next_i == 0: continue # MASTER ANCHOR: Never let Lens 1 spin!
                        
                    old_yaw, pitch = new_orientations[next_i]
                    correction = np.clip(shift_deg * 0.40, -0.6, 0.6)
                    new_yaw = old_yaw + correction
                    
                    # HARDWARE CLAMP: Do not allow SIFT to push the lens into a black strip
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

    # --- CHANGED: Extract first frames for init, pass 'frames' into resets ---
    def stitch_video(self, input_paths, output_path, resize_to=(1920, 960), run_sift=False):
        caps = []
        for p in input_paths:
            caps.append(cv2.VideoCapture(p))
            
        if any(not c.isOpened() for c in caps): 
            return False
            
        # Initial calibration at the beginning of the video
        if run_sift:
            init_frames = []
            for cap in caps:
                ret, frame = cap.read()
                init_frames.append(frame)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rewind safe ONLY at the start
                
            self.auto_tune_zooms(init_frames)
            self.calibrate_angles_with_sift(init_frames)
        else:
            print("\n   [INFO] Bypassing SIFT. Using cached geometry for speed.")
            
        fps = caps[0].get(cv2.CAP_PROP_FPS) or 29.97
        
        crop_h = self.H - self.CROP_TOP - self.CROP_BOTTOM
        out_dims = resize_to if resize_to else (self.W, crop_h)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, out_dims)
        
        # 7 minutes * 60 seconds * ~30 frames per second = 12600 frames
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
                
                # --- FIXED: HARD RESET PASSES THE IMAGES, NOT THE VIDEO STREAMS ---
                if frame_idx > 0 and frame_idx % RESET_FRAME_INTERVAL == 0:
                    print(f"\n\n[🚨 HARD RESET] 7 Minutes Reached (Frame {frame_idx}). Wiping out accumulative drift...")
                    print("[INFO] Re-running complete calibration pipeline from scratch...")
                    
                    self.auto_tune_zooms(frames)
                    self.calibrate_angles_with_sift(frames)
                    
                    print("[SUCCESS] New baseline geometry locked in. Resuming stitching seamlessly.\n")
                
                elif frame_idx > 0 and frame_idx % 150 == 0:
                    print(f"\n   [Frame {frame_idx}] Checking thermal drift...")
                    self.check_drift_and_update(frames)
                
                total_img = np.zeros((self.H, self.W, 3), dtype=np.float32)
                total_weight = np.zeros((self.H, self.W), dtype=np.float32)
                
                for i in range(6):
                    warped = cv2.remap(frames[i], self.maps_x[i], self.maps_y[i], cv2.INTER_LINEAR)
                    w = self.weights[i]
                    total_img += warped.astype(np.float32) * np.dstack([w, w, w])
                    total_weight += w
                
                total_weight[total_weight == 0] = 1.0
                final_pano = np.clip(total_img / np.dstack([total_weight]*3), 0, 255).astype(np.uint8)
                
                final_pano = final_pano[self.CROP_TOP : self.H - self.CROP_BOTTOM, :]

                if resize_to:
                    final_pano = cv2.resize(final_pano, resize_to, interpolation=cv2.INTER_AREA)

                out.write(final_pano)
                
                frame_idx += 1
                if frame_idx % 10 == 0: print(f"   Processed {frame_idx} frames...", end='\r')
                    
        except KeyboardInterrupt: pass
        finally:
            for cap in caps: cap.release()
            out.release()
            print("\nDone!")
            
        return True


class LegacyClipVideoStitcher:
    def __init__(self, intrinsics_dir):
        self.intrinsics_dir = intrinsics_dir
        
        # --- CONFIGURATION ---
        self.W = 1920
        self.H = 960
        self.CROP_TOP = 191      # Scaled down from 331
        self.CROP_BOTTOM = 306   # Scaled down from 335
        
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
        self.Ks = []
        self.img_dims = []
        self.final_errors = [0.0] * 6 

        self._load_intrinsics()
        print(f"Initializing Geometry ({self.W}x{self.H})...")
        self._init_geometry_and_weights()
        self.FACTORY_ORIENTATIONS = list(self.LENS_ORIENTATIONS) # <--- ADD THIS ANCHOR

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

    def find_shift_with_sift(self, img1, img2):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5: return 0
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
                
        if len(good) < 4: return 0
            
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        
        dx = pts1[:, 0] - pts2[:, 0]
        dy = pts1[:, 1] - pts2[:, 1]
        
        valid_matches = np.abs(dy) < 10 
        if np.sum(valid_matches) < 3: return 0
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
                    if next_i == 0: continue # MASTER ANCHOR: Never let Lens 1 spin!
                        
                    old_yaw, pitch = new_orientations[next_i]
                    correction = np.clip(shift_deg * 0.40, -0.6, 0.6)
                    new_yaw = old_yaw + correction
                    
                    # HARDWARE CLAMP: Do not allow SIFT to push the lens into a black strip
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

    # --- CHANGED: Extract first frames for init, pass 'frames' into resets ---
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
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rewind safe ONLY at the start
                
            self.calibrate_angles_with_sift(init_frames)
        else:
            print("\n   [INFO] Bypassing SIFT. Using cached geometry for speed.")
            
        fps = caps[0].get(cv2.CAP_PROP_FPS) or 29.97
        
        crop_h = self.H - self.CROP_TOP - self.CROP_BOTTOM
        out_dims = resize_to if resize_to else (self.W, crop_h)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, out_dims)
        
        # 7 minutes * 60 seconds * ~30 frames per second = 12600 frames
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
                
                # --- FIXED: HARD RESET PASSES THE IMAGES, NOT THE VIDEO STREAMS ---
                if frame_idx > 0 and frame_idx % RESET_FRAME_INTERVAL == 0:
                    print(f"\n\n[🚨 HARD RESET] 7 Minutes Reached (Frame {frame_idx}). Wiping out accumulative drift...")
                    print("[INFO] Re-running legacy angle calibration pipeline from scratch...")
                    
                    self.calibrate_angles_with_sift(frames)
                    
                    print("[SUCCESS] New baseline geometry locked in. Resuming stitching seamlessly.\n")
                
                elif frame_idx > 0 and frame_idx % 150 == 0:
                    print(f"\n   [Frame {frame_idx}] Pausing to check for thermal drift...")
                    self.check_drift_and_update(frames)
                    print(f"   [Frame {frame_idx}] Check complete. Resuming video processing...")
                
                total_img = np.zeros((self.H, self.W, 3), dtype=np.float32)
                total_weight = np.zeros((self.H, self.W), dtype=np.float32)
                
                for i in range(6):
                    warped = cv2.remap(frames[i], self.maps_x[i], self.maps_y[i], cv2.INTER_LINEAR)
                    w = self.weights[i]
                    total_img += warped.astype(np.float32) * np.dstack([w, w, w])
                    total_weight += w
                
                total_weight[total_weight == 0] = 1.0
                final_pano = np.clip(total_img / np.dstack([total_weight]*3), 0, 255).astype(np.uint8)
                
                final_pano = final_pano[self.CROP_TOP : self.H - self.CROP_BOTTOM, :]

                if resize_to:
                    final_pano = cv2.resize(final_pano, resize_to, interpolation=cv2.INTER_AREA)

                out.write(final_pano)
                
                frame_idx += 1
                if frame_idx % 10 == 0: print(f"   Processed {frame_idx} frames...", end='\r')
                    
        except KeyboardInterrupt: pass
        finally:
            for cap in caps: cap.release()
            out.release()
            print("\nDone!")
            
        return True

# ==========================================
# 5. VIDEO AI CLASSES 
# ==========================================

class ClipVideoTrimmer:
    @staticmethod
    def trim(input_path, output_path, start_sec, duration_sec, fps=29.97):
        cmd = [
            "ffmpeg", 
            "-ss", str(start_sec),
            "-i", input_path,
            "-t", str(duration_sec),
            # --- FIXED: Added scale=iw/2:ih/2 to shrink resolution by 50% ---
            "-vf", f"fps={fps},scale=iw/2:ih/2",      
            "-c:v", "libx264",        
            # --- FIXED: Raised CRF from 18 to 28 for heavy file size compression ---
            "-crf", "28",             
            "-preset", "ultrafast",   
            "-an",                    
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
    def process(input_path, output_path, K_orig, D, xi, orig_size, fov_scale=0.4):
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

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)) if output_path else None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break
                
            frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            undistorted = cv2.omnidir.undistortImage(
                frame_rotated, K_scaled, D, xi_vec, 
                cv2.omnidir.RECTIFY_PERSPECTIVE, 
                Knew=K_new, 
                new_size=(w, h)
            )
            if out: out.write(undistorted)
            
        cap.release()
        if out: out.release()
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
        ref_w = 1920 # Changed from 3328
        ref_h = 960  # Changed from 998
        
        scale_x = w / ref_w
        scale_y = h / ref_h
        
        self.SKY_Y_LIMIT = int((ref_h * 0.10) * scale_y)
        self.VEHICLE_Y_LIMIT = int(788 * scale_y) # Scaled for 960
        self.VIP_Y_LIMIT = int(595 * scale_y)     # Scaled for 960
        
        base_polygon = np.array([
            [0, 660], [259, 752], [279, 735], [324, 733],
            [329, 755], [395, 747], [432, 865], [495, 824],
            [558, 920], [657, 860], [958, 854], [1142, 936],
            [1213, 736], [1274, 729], [1310, 704], [1377, 727],
            [1633, 657], [1919, 649], [2000, 2000], [-100, 2000]
        ], np.float32)
        
        base_polygon[:, 0] *= scale_x
        base_polygon[:, 1] *= scale_y
        self.EGO_POLYGON = np.int32(base_polygon)

    def get_dynamic_threshold(self, bbox_bottom_y, label):
        if label in ['bus', 'truck']: 
            thresh = 0.15 if bbox_bottom_y > self.VIP_Y_LIMIT else 0.35
        else:
            y_clamped = max(self.SKY_Y_LIMIT, min(bbox_bottom_y, self.VEHICLE_Y_LIMIT))
            if self.VEHICLE_Y_LIMIT == self.SKY_Y_LIMIT:
                ratio = 1.0
            else:
                ratio = ((y_clamped - self.SKY_Y_LIMIT) / (self.VEHICLE_Y_LIMIT - self.SKY_Y_LIMIT)) ** 2 
                
            thresh = self.CONF_FAR + (self.CONF_CLOSE - self.CONF_FAR) * ratio
            
            if label == 'auto': thresh = max(thresh, 0.60) 
            
        # Absolute minimum of 0.27 for ALL vehicles
        return max(thresh, 0.27)

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
            # Resolve tracker configuration
            tracker_file = r"C:\viswak_MUMMAS_360degcamera\Insta360ImageAnalysis\FINAL_custom_track.yaml"
            if not os.path.exists(tracker_file):
                local_tracker = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Vehicle_detection_360", "FINAL_custom_track.yaml")
                tracker_file = local_tracker if os.path.exists(local_tracker) else "bytetrack.yaml"
                
            # Run tracker at a strong scale to capture small details
            results = self.model.track(
                frame, 
                persist=True, 
                verbose=False, 
                conf=0.27, 
                iou=0.45, 
                imgsz=1920, 
                tracker=tracker_file
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
        
        # High-Resolution Physics Rules (3328x998)
        self.TARGET_FPS = 30                 
        self.MAX_SPEED_PX_PER_SEC = 130      
        self.MIN_MOVEMENT_PX = 20            
        self.EGO_STOPPED_THRESHOLD_PX = 3.5  

        # Biological Constraints
        self.MAX_PED_WIDTH_PX = 450          
        self.MAX_PED_HEIGHT_PX = 900         
        self.MIN_ASPECT_RATIO = 1.1          

        self.BASE_EGO_POLYGON = np.array([
    [0, 660], [259, 752], [279, 735], [324, 733],
    [329, 755], [395, 747], [432, 865], [495, 824],
    [558, 920], [657, 860], [958, 854], [1142, 936],
    [1213, 736], [1274, 729], [1310, 704], [1377, 727],
    [1633, 657], [1919, 649], [2000, 2000], [-100, 2000]
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
        scale_x = width / 1920.0  # Changed from 3328.0
        scale_y = height / 960.0  # Changed from 998.0
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
        
        self.SKY_Y_LIMIT = int(960 * 0.05)  # Changed 998 to 960
        self.VEHICLE_Y_LIMIT = 788          # Scaled for 960
        
        self.EGO_POLYGON = np.array([
            [0, 660], [259, 752], [279, 735], [324, 733],
            [329, 755], [395, 747], [432, 865], [495, 824],
            [558, 920], [657, 860], [958, 854], [1142, 936],
            [1213, 736], [1274, 729], [1310, 704], [1377, 727],
            [1633, 657], [1919, 649], [2000, 2000], [-100, 2000]
        ], np.int32)       
        
        self.CONF_FAR = 0.10        
        self.CONF_CLOSE = 0.18    

    def get_dynamic_threshold(self, bbox_bottom_y, label):
        # 1. Calculate class-specific or distance-based thresholds
        if label in ['bus', 'truck']:
            thresh = 0.15 if bbox_bottom_y > 595 else 0.40  
        else:
            y_clamped = max(self.SKY_Y_LIMIT, min(bbox_bottom_y, self.VEHICLE_Y_LIMIT))
            if self.VEHICLE_Y_LIMIT == self.SKY_Y_LIMIT:
                ratio = 1.0
            else:
                linear_ratio = (y_clamped - self.SKY_Y_LIMIT) / (self.VEHICLE_Y_LIMIT - self.SKY_Y_LIMIT)
                ratio = linear_ratio ** 2 
                
            thresh = self.CONF_FAR + (self.CONF_CLOSE - self.CONF_FAR) * ratio
            
            if label == 'auto': thresh = max(thresh, 0.60) 

        # 2. Enforce absolute minimum of 0.27 for all vehicles
        return max(thresh, 0.27)

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
                                         conf=0.27,  
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
                json.dump(full_metadata, f, indent=4, default=lambda o: float(o) if isinstance(o, (np.floating, float)) else int(o) if isinstance(o, (np.integer, int)) else o.tolist() if isinstance(o, np.ndarray) else str(o))
            
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
            image_size=832, # 832 is still perfectly fine for SAHI slicing on a 998 height image
            device=self.device,
        )
        
        # --- BIOLOGICAL GEOMETRY LIMITS ---
        self.MAX_PED_WIDTH_PX = 450          
        self.MAX_PED_HEIGHT_PX = 900         
        self.MIN_ASPECT_RATIO = 1.1          

        # --- CUSTOM EGO-VEHICLE POLYGON ---
        # Shifted up by 331 pixels. Bottom corners clamped to 998.
        self.BASE_EGO_POLYGON = np.array([
    [0, 660], [259, 752], [279, 735], [324, 733],
    [329, 755], [395, 747], [432, 865], [495, 824],
    [558, 920], [657, 860], [958, 854], [1142, 936],
    [1213, 736], [1274, 729], [1310, 704], [1377, 727],
    [1633, 657], [1919, 649], [2000, 2000], [-100, 2000]
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
        scale_x = width / 1920.0 # Changed from 3328.0
        scale_y = height / 960.0 # Changed from 998.0
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
                json.dump(final_output_dict, f, indent=4, default=lambda o: float(o) if isinstance(o, (np.floating, float)) else int(o) if isinstance(o, (np.integer, int)) else o.tolist() if isinstance(o, np.ndarray) else str(o))
            
        return ped_count


# ==========================================
# 5.5 STATIC PIPELINE HELPERS (IN-MEMORY & DISK CACHE)
# ==========================================

class GeometryCache:
    """Handles disk-based caching for SIFT angles to share across Multiprocessing workers."""
    @staticmethod
    def get_cache(cache_dir, target_dt, max_diff_sec=3600):
        if not os.path.exists(cache_dir): return None
        valid_caches = []
        for f in os.listdir(cache_dir):
            if f.endswith('.json') and f.startswith('geom_'):
                try:
                    dt_str = f.replace("geom_", "").replace(".json", "")
                    cache_dt = datetime.strptime(dt_str, "%Y-%m-%d_%H-%M-%S")
                    diff = abs((cache_dt - target_dt).total_seconds())
                    if diff <= max_diff_sec:
                        valid_caches.append((diff, os.path.join(cache_dir, f)))
                except Exception: pass
                
        if valid_caches:
            valid_caches.sort() # Grab the closest one in time
            with open(valid_caches[0][1], 'r') as f:
                return json.load(f)
        return None

    @staticmethod
    def save_cache(cache_dir, target_dt, zooms, orientations):
        try:
            os.makedirs(cache_dir, exist_ok=True)
            dt_str = target_dt.strftime("%Y-%m-%d_%H-%M-%S")
            cache_path = os.path.join(cache_dir, f"geom_{dt_str}.json")

            def _to_builtin(val):
                if isinstance(val, (np.floating, float)):
                    return float(val)
                if isinstance(val, (np.integer, int)):
                    return int(val)
                if isinstance(val, np.ndarray):
                    return val.tolist()
                if isinstance(val, (list, tuple)):
                    return [_to_builtin(x) for x in val]
                if isinstance(val, dict):
                    return {str(k): _to_builtin(v) for k, v in val.items()}
                return val

            data = {
                "timestamp": str(target_dt),
                "zooms": _to_builtin(zooms),
                "orientations": _to_builtin(orientations)
            }
            # Safe write to prevent collision between workers
            tmp_path = cache_path + f".tmp{time.time()}"
            with open(tmp_path, 'w') as f:
                json.dump(data, f, default=lambda o: float(o) if isinstance(o, (np.floating, float)) else int(o) if isinstance(o, (np.integer, int)) else o.tolist() if isinstance(o, np.ndarray) else str(o))
            os.replace(tmp_path, cache_path)
        except Exception as e:
            logger.warning(f"Failed to save geometry cache: {e}")

def run_extraction(folder, frame_no, out_dir, OMNI_JSON, meta, save_media=True, target_dt=None):
    start_time = time.perf_counter() 
    undist_dir = os.path.join(out_dir, "undistorted")
    intr_dir = os.path.join(out_dir, "INTRINSICS")
    os.makedirs(undist_dir, exist_ok=True)
    os.makedirs(intr_dir, exist_ok=True)
    
    img_paths = []
    extracted_frames = []
    
    # --- DATE ROUTING LOGIC ---
    cutoff_date = datetime(2026, 3, 1)
    is_optimized = True # Default fallback
    if target_dt:
        is_optimized = target_dt >= cutoff_date
    else:
        try:
            session_dt = datetime.fromisoformat(meta.get("start_ist_str", ""))
            is_optimized = session_dt >= cutoff_date
        except Exception:
            pass
    
    for i in range(1, 7):
        print(f"   -> Extracting & Undistorting Lens {i}/6...", end='\r')
        vid_path = os.path.join(folder, f"LENS{i}", f"video_lens{i}.mp4")
        if not os.path.exists(vid_path): 
            UI.error(f"Missing video for Lens {i}: {vid_path}")
            return False, intr_dir, [], [], meta
            
        cap = cv2.VideoCapture(vid_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        cap.release()
        
        if not ret: 
            UI.error(f"Failed to read frame {frame_no} from Lens {i}")
            return False, intr_dir, [], [], meta
            
        # Dynamic Intrinsic Loading
        K_orig, D, xi, orig_size, source_json = get_intrinsics_for_lens(i, OMNI_JSON, is_optimized)
            
        undistorted, K_new, w, h = process_and_undistort(frame, K_orig, D, xi, orig_size)
        if undistorted is None: 
            UI.error(f"Undistortion failed for Lens {i}")
            return False, intr_dir, [], [], meta
            
        extracted_frames.append(undistorted)
        out_img_path = os.path.join(undist_dir, f"undistorted_LENS{i}.jpg")
        
        if save_media:
            cv2.imwrite(out_img_path, undistorted) 
        img_paths.append(out_img_path)
        
        # Save pinhole metadata with the correct source tracker
        intr_dict = {
            "K": K_new.tolist(), 
            "D": [0.0, 0.0, 0.0, 0.0, 0.0], 
            "model": "pinhole_from_omni",
            "source_omni_json": source_json,
            "image_size": [w, h]
        }
        with open(os.path.join(intr_dir, f"calibration_pinhole_lens{i}.json"), "w") as jf:
            json.dump(intr_dict, jf, indent=2)
            
    print("   -> Extracting & Undistorting Lens 6/6... Done!      ")
            
    meta["extraction_time_sec"] = round(time.perf_counter() - start_time, 4)
    meta["extracted_lenses"] = img_paths
    
    if save_media:
        with open(os.path.join(out_dir, "static_extraction_meta.json"), "w") as f:
            json.dump(meta, f, indent=4)
            
    return True, intr_dir, img_paths, extracted_frames, meta

def run_stitching(output_dir, intr_dir, extracted_frames, meta_info, cached_stitcher=None, run_sift=True, save_media=True, target_dt=None, preloaded_geom=None):
    start_time = time.perf_counter() 
    pano_path = os.path.join(output_dir, "Final_Static_Stitch.jpg")
    
    if cached_stitcher is not None:
        stitcher = cached_stitcher
    else:
        # --- DATE ROUTING LOGIC ---
        cutoff_date = datetime(2026, 3, 1)
        is_optimized = True # Default fallback
        
        if target_dt:
            is_optimized = target_dt >= cutoff_date
        else:
            try:
                session_dt = datetime.fromisoformat(meta_info.get("start_ist_str", ""))
                is_optimized = session_dt >= cutoff_date
            except Exception:
                pass 
                
        if is_optimized:
            stitcher = OptimizedImageStitcher(intrinsics_dir=intr_dir)
        else:
            stitcher = LegacyImageStitcher(intrinsics_dir=intr_dir)
            
    # --- NEW: APPLY CACHED DISK GEOMETRY ---
    if preloaded_geom:
        stitcher.LENS_ZOOMS = preloaded_geom['zooms']
        stitcher.LENS_ORIENTATIONS = [tuple(x) for x in preloaded_geom['orientations']]
        stitcher._init_geometry_and_weights()
    
    final_pano = stitcher.stitch_frames(extracted_frames, pano_path, run_sift=run_sift, save_media=save_media) 
    
    if final_pano is not None:
        meta_info["stitching_time_sec"] = round(time.perf_counter() - start_time, 4)
        meta_info["lens_zooms"] = stitcher.LENS_ZOOMS
        meta_info["lens_errors_deg"] = [float(e) for e in stitcher.final_errors]
        meta_info["final_orientations"] = [(float(yaw), float(pitch)) for yaw, pitch in stitcher.LENS_ORIENTATIONS]
        
        if save_media:
            try:
                with open(os.path.join(output_dir, "static_stitched_meta.json"), "w") as f: 
                    json.dump(meta_info, f, indent=4, default=lambda o: float(o) if isinstance(o, (np.floating, float)) else int(o) if isinstance(o, (np.integer, int)) else o.tolist() if isinstance(o, np.ndarray) else str(o))
            except Exception as e:
                UI.warn(f"Could not save static_stitched_meta.json: {e}")
            
        return pano_path, meta_info, stitcher, final_pano
        
    return None, meta_info, stitcher, None


# ==========================================
# 6. PIPELINE EXECUTION LOOPS
# ==========================================

def frames_pipeline(mapped_data, registry, OMNI_JSON, ROOT_OUTPUT, YOLO_VEHICLE_MODEL, YOLO_PEDESTRIAN_MODEL, target_dt, override_meta=None, action_choice=None, batch_mode=False, save_media=True, stitcher_cache=None, date_key=None, veh_detector=None, ped_detector=None, class_name=None):
    folder, frame_no, meta = get_target_frame(mapped_data, registry, ROOT_OUTPUT, target_dt, override_meta=override_meta, is_clip=False)
    if not folder: 
        return None, None # Signal failure back to batch processor

    if class_name:
        meta["class_name"] = class_name
        meta["lcz_class"] = class_name
    elif override_meta and (override_meta.get("class_name") or override_meta.get("lcz_class")):
        meta["class_name"] = override_meta.get("class_name") or override_meta.get("lcz_class")
        meta["lcz_class"] = meta["class_name"]

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
                    best_frame = find_sharpest_frame_offset(lens6_path, frame_no, offsets=[0, 10, 20])
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
                    _, f_intr, f_imgs, extracted_frames_burst, f_meta = run_extraction(folder, f, f_dir, OMNI_JSON, f_meta, save_media=save_media, target_dt=target_dt)
                    
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
        _, intr_dir, img_paths, extracted_frames, meta = run_extraction(folder, frame_no, out_dir, OMNI_JSON, meta, save_media=save_media, target_dt=target_dt)
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
            if not batch_mode: UI.info("Step 2: Running Stitching...")
            
            # --- NEW DISK CACHING LOGIC (Multiprocessing Safe) ---
            cache_dir = os.path.join(ROOT_OUTPUT, ".geometry_cache")
            geom_data = None
            
            if target_dt:
                # Check disk for any SIFT calculation done within the last 1 Hour (3600 seconds)
                geom_data = GeometryCache.get_cache(cache_dir, target_dt, max_diff_sec=3600) 
            
            run_sift = True
            if geom_data:
                run_sift = False
                
            pano_path, meta, returned_stitcher, final_pano = run_stitching(
                out_dir, intr_dir, extracted_frames, meta, 
                cached_stitcher=None, run_sift=run_sift, 
                save_media=save_media, target_dt=target_dt, preloaded_geom=geom_data
            )

            # Save the newly calculated SIFT geometry to disk so other CPU cores can use it!
            if run_sift and returned_stitcher is not None and target_dt is not None:
                GeometryCache.save_cache(cache_dir, target_dt, returned_stitcher.LENS_ZOOMS, returned_stitcher.LENS_ORIENTATIONS)
            
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
    
    # Synchronize to categorized class folder if class_name is provided
    eff_class = meta.get("class_name") or meta.get("lcz_class") or class_name
    if save_media and eff_class and os.path.exists(out_dir):
        class_out = sync_to_class_folder(out_dir, ROOT_OUTPUT, eff_class)
        if not batch_mode and class_out:
            UI.info(f"Categorized output saved to: {class_out}")

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


def clips_pipeline(mapped_data, registry, OMNI_JSON, ROOT_OUTPUT, YOLO_VEHICLE_MODEL, YOLO_PEDESTRIAN_MODEL, target_dt, override_meta=None, action_choice=None, batch_mode=False, save_media=True, batch_dur=None, class_name=None):
    folder, frame_no, meta = get_target_frame(mapped_data, registry, ROOT_OUTPUT, target_dt, override_meta=override_meta, is_clip=True, batch_dur=batch_dur)
    
    if not folder: 
        return None, None 

    if class_name:
        meta["class_name"] = class_name
        meta["lcz_class"] = class_name
    elif override_meta and (override_meta.get("class_name") or override_meta.get("lcz_class")):
        meta["class_name"] = override_meta.get("class_name") or override_meta.get("lcz_class")
        meta["lcz_class"] = meta["class_name"] 

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
        
        # --- DATE ROUTING LOGIC ---
        cutoff_date = datetime(2026, 3, 1)
        is_optimized = True # Default fallback
        if target_dt:
            is_optimized = target_dt >= cutoff_date
        else:
            try:
                session_dt = datetime.fromisoformat(meta.get("start_ist_str", ""))
                is_optimized = session_dt >= cutoff_date
            except Exception:
                pass
        
        play_time_sec = float(meta.get('playback_time_sec', float(meta['frame_no']) / float(meta['fps'])))
        
        for i in range(1, 7):
            # Dynamic Intrinsic Loading
            K_orig, D, xi, orig_size, source_json = get_intrinsics_for_lens(i, OMNI_JSON, is_optimized)
            
            src_vid = os.path.join(folder, f"LENS{i}", f"video_lens{i}.mp4")
            raw_vid = os.path.join(raw_dir, f"lens{i}.mp4")
            
            # Pass the exact session FPS so FFmpeg knows how many frames to force
            target_fps = meta.get('fps', 29.97)
            trim_success = ClipVideoTrimmer.trim(src_vid, raw_vid, play_time_sec, meta['clip_duration_sec'], target_fps)
            
            if trim_success:
                K_new, dims = ClipVideoUndistorter.process(raw_vid, undist_paths[i-1], K_orig, D, xi, orig_size)
                
                intr_dict = {
                    "K": K_new, 
                    "D": [0.0, 0.0, 0.0, 0.0, 0.0], 
                    "model": "pinhole_from_omni",
                    "source_omni_json": source_json,
                    "image_size": dims
                }
                with open(os.path.join(intr_dir, f"calibration_pinhole_lens{i}.json"), "w") as jf:
                    json.dump(intr_dict, jf, indent=2)
                    
        meta["undistortion_time_sec"] = round(time.perf_counter() - undistort_start_time, 4)
        with open(os.path.join(out_dir, "clip_extraction_meta.json"), "w") as f: json.dump(meta, f, indent=4)
        if not batch_mode: UI.success("Undistortion Complete.")

    # STEP 2: Stitch
    if choice in ['2', '3', '4'] and do_stitch:
        if not batch_mode: UI.info("Step 2: Running SIFT Video Stitcher...")
        stitch_start_time = time.perf_counter()
        
        # --- DATE ROUTING LOGIC ---
        cutoff_date = datetime(2026, 3, 1)
        is_optimized = True # Default fallback
        
        if target_dt:
            is_optimized = target_dt >= cutoff_date
        else:
            try:
                session_dt = datetime.fromisoformat(meta["start_ist_str"])
                is_optimized = session_dt >= cutoff_date
            except Exception:
                pass # Defaults to optimized if unparseable
                
        if is_optimized:
            if not batch_mode: UI.info("Routing to OPTIMIZED Stitcher (Date >= 01/03/2026)")
            stitcher = OptimizedClipVideoStitcher(intrinsics_dir=intr_dir)
        else:
            if not batch_mode: UI.info("Routing to LEGACY Stitcher (Date < 01/03/2026)")
            stitcher = LegacyClipVideoStitcher(intrinsics_dir=intr_dir)
        # --------------------------
        
        stitch_success = stitcher.stitch_video(undist_paths, stitched_vid)
        
        if stitch_success:
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

    # Synchronize to categorized class folder if class_name is provided
    eff_class = meta.get("class_name") or meta.get("lcz_class") or class_name
    if save_media and eff_class and os.path.exists(out_dir):
        class_out = sync_to_class_folder(out_dir, ROOT_OUTPUT, eff_class)
        if not batch_mode and class_out:
            UI.info(f"Categorized output saved to: {class_out}")

    if not batch_mode:
        UI.success(f"Video Pipeline Finished! Output at: {out_dir}")
        UI.pause()
        
    if not save_media and os.path.exists(out_dir): shutil.rmtree(out_dir, ignore_errors=True)
    return veh_counts, ped_count


# ==========================================
# 7. BATCH PROCESS HELPER (CRASH PROOF)
# ==========================================
def gpu_batch_consumer(img_queue, yolo_model_path, out_csv_name, batch_size=16):
    """Dedicated thread that groups images into batches and feeds the TensorRT Engine."""
    print(f"\n[INFO] Starting Dedicated GPU Batch Consumer (Batch Size: {batch_size})")
    model = YOLO(yolo_model_path, task='detect') 
    
    # --- UPDATED: Ego-Polygon Setup (Using new cropped 3328x998 resolution) ---
    # Shifted by -331 on Y-axis. Bottom corners clamped to 998.
    poly = [
    [0, 660], [259, 752], [279, 735], [324, 733],
    [329, 755], [395, 747], [432, 865], [495, 824],
    [558, 920], [657, 860], [958, 854], [1142, 936],
    [1213, 736], [1274, 729], [1310, 704], [1377, 727],
    [1633, 657], [1919, 649], [1919, 959], [2, 956]
]
    EGO_POLYGON = np.array(poly, np.int32)
    
    # --- UPDATED: Sky Limit ---
    SKY_Y_LIMIT = int(960 * 0.05) # Changed from 998
    
    batch_images = []
    batch_metadata = []
    
    def process_and_save_batch(images, metas):
        if not images: return
        
        # --- TRUE GPU BATCH INFERENCE ---
        # The GPU processes all 32 images simultaneously here
        # --- TRUE GPU BATCH INFERENCE ---
        # Changed imgsz=2560 to imgsz=1920 for massive speed boost
        results = model.predict(images, imgsz=1920, conf=0.27, iou=0.75, verbose=False, batch=len(images))
        
        for i, res in enumerate(results):
            row_dict = metas[i]
            live_counts = {'car': 0, 'auto': 0, 'bus': 0, 'truck': 0, 'motorbike': 0}
            
            if res.boxes is not None:
                boxes = res.boxes.xyxy.cpu().numpy().astype(int)
                clss = res.boxes.cls.int().cpu().tolist()
                
                for box, cls_idx in zip(boxes, clss):
                    if box[3] < SKY_Y_LIMIT: continue
                    bottom_center = (int((box[0] + box[2]) / 2), int(box[3]))
                    if cv2.pointPolygonTest(EGO_POLYGON, bottom_center, False) >= 0: continue
                        
                    raw_label = model.names[cls_idx]
                    if raw_label in live_counts:
                        live_counts[raw_label] += 1
            
            # Apply counts to the CSV row dict
            for k, v in live_counts.items():
                if k.upper() in row_dict:
                    row_dict[k.upper()] = v
                    
            # Save row to disk
            pd.DataFrame([row_dict]).to_csv(out_csv_name, mode='a', header=False, index=False)
            
        print(f"   [GPU CONSUMER] ⚡ Processed & Saved Batch of {len(images)} frames.")

    # Endless loop watching the Queue
    while True:
        item = img_queue.get()
        
        # "DONE" is our kill-switch signal
        if item == "DONE":
            process_and_save_batch(batch_images, batch_metadata) # Flush remaining
            break
            
        idx, row_dict, pano = item
        
        # --- NEW: Catch failures BEFORE they hit the GPU ---
        if pano is None:
            # Write the failed row to the CSV immediately so we don't lose it
            pd.DataFrame([row_dict]).to_csv(out_csv_name, mode='a', header=False, index=False)
            continue # Skip adding this to the YOLO batch
            
        batch_images.append(pano)
        batch_metadata.append(row_dict)
        
        if len(batch_images) == batch_size:
            process_and_save_batch(batch_images, batch_metadata)
            batch_images = []
            batch_metadata = []

def _process_single_csv_row(args):
    """CPU Producer Worker: Extracts, Stitches, and sends to Queue."""
    idx, row_dict, ts_col, ROOT_OUTPUT, HARD_DRIVE_ROOT, OMNI_JSON, target_dt, img_queue = args
    
    # 1. Unpack the 4th variable (err_msg)
    csv_found, json_folder, session_meta, err_msg = AssetDiscovery.search_for_timestamp(HARD_DRIVE_ROOT, target_dt, verbose=False)
    
    if not csv_found or err_msg != "OK":
        row_dict['available'] = 'no'
        row_dict['failure_reason'] = err_msg
        img_queue.put((idx, row_dict, None)) # Push failure to queue
        return
        
    row_dict['available'] = 'yes'
    row_dict['failure_reason'] = ""
    
    registry = build_session_registry(os.path.dirname(json_folder))
    mapping_output_path = os.path.join(ROOT_OUTPUT, "Master_Mapping.csv")
    mapped_data = process_imu_gps_file(csv_found, registry, mapping_output_path)
    
    if not mapped_data: 
        row_dict['available'] = 'no'
        row_dict['failure_reason'] = 'No matching timeframe inside the CSV'
        img_queue.put((idx, row_dict, None))
        return

    folder, frame_no, meta = get_target_frame(mapped_data, registry, ROOT_OUTPUT, target_dt, is_clip=False)
    if not folder:
        row_dict['available'] = 'no'
        row_dict['failure_reason'] = 'Time gap too large or exact frame not found'
        img_queue.put((idx, row_dict, None))
        return

    out_dir = os.path.join(ROOT_OUTPUT, f"Pipeline_TEMP_{idx}")
    
    # 1. CPU Task: Extract
    success, intr_dir, img_paths, extracted_frames, meta = run_extraction(folder, frame_no, out_dir, OMNI_JSON, meta, save_media=False, target_dt=target_dt)
    
    if not success or len(extracted_frames) != 6:
        row_dict['available'] = 'no'
        row_dict['failure_reason'] = 'Frame extraction failed or corrupt lenses'
        img_queue.put((idx, row_dict, None))
        return
        
    # 2. CPU Task: Stitch
    cache_dir = os.path.join(ROOT_OUTPUT, ".geometry_cache")
    geom_data = GeometryCache.get_cache(cache_dir, target_dt, max_diff_sec=3600) 
    
    run_sift = False if geom_data else True
    _, _, returned_stitcher, final_pano = run_stitching(
        out_dir, intr_dir, extracted_frames, meta, 
        cached_stitcher=None, run_sift=run_sift, 
        save_media=False, target_dt=target_dt, preloaded_geom=geom_data
    )

    if run_sift and returned_stitcher is not None:
        GeometryCache.save_cache(cache_dir, target_dt, returned_stitcher.LENS_ZOOMS, returned_stitcher.LENS_ORIENTATIONS)
        
    # 3. PUSH TO CONSUMER QUEUE
    if final_pano is not None:
        img_queue.put((idx, row_dict, final_pano))
    else:
        row_dict['available'] = 'no'
        row_dict['failure_reason'] = 'Stitching failed'
        img_queue.put((idx, row_dict, None))

def batch_process_csv(csv_path, ROOT_OUTPUT, HARD_DRIVE_ROOT, YOLO_VEHICLE, YOLO_PED, OMNI_JSON):
    global parse_target_timestamp 
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
    ts_col = next((c for c in df.columns if str(c).strip().lower() in ['timestamp', 'target_time', 'datetime', 'date_time', 'time', 'ist_time', 'utc_time', 'unix_time']), None)

    lat_col = next((c for c in df.columns if str(c).strip().lower() in ['latitude', 'lat', 'y', 'gnss_latitude', 'filter_lla_lat']), None)
    lon_col = next((c for c in df.columns if str(c).strip().lower() in ['longitude', 'lon', 'long', 'x', 'gnss_longitude', 'filter_lla_lon']), None)

    if not ts_col and lat_col and lon_col:
        print(f"\n{UI.CYAN}[INFO] No 'timestamp' column found, but detected GPS columns ('{lat_col}', '{lon_col}').{UI.RESET}")
        print(f"{UI.CYAN}[INFO] Resolving closest timestamps from IMU GPS data...{UI.RESET}")
        
        matched_ts_list = []
        matched_dist_list = []
        for idx_row, row in df.iterrows():
            try:
                lat_v = float(row[lat_col])
                lon_v = float(row[lon_col])
                res = GPSResolver.lookup_gps(HARD_DRIVE_ROOT, lat_v, lon_v, max_dist_meters=500.0, verbose=False)
                if res:
                    matched_ts_list.append(res["matched_timestamp"])
                    matched_dist_list.append(round(res["distance_meters"], 2))
                else:
                    matched_ts_list.append("")
                    matched_dist_list.append(999999.0)
            except Exception:
                matched_ts_list.append("")
                matched_dist_list.append(999999.0)
                
        df['timestamp'] = matched_ts_list
        df['gps_matched_dist_m'] = matched_dist_list
        ts_col = 'timestamp'
        valid_gps_count = sum(1 for t in matched_ts_list if t)
        print(f"{UI.GREEN}[SUCCESS] Resolved {valid_gps_count}/{len(df)} timestamps from GPS coordinates!{UI.RESET}")

    elif ts_col and lat_col and lon_col:
        missing_mask = df[ts_col].isna() | (df[ts_col].astype(str).str.strip() == '')
        if missing_mask.any():
            print(f"\n{UI.CYAN}[INFO] Backfilling {missing_mask.sum()} missing timestamp(s) from GPS coordinates...{UI.RESET}")
            for idx_row in df[missing_mask].index:
                try:
                    lat_v = float(df.loc[idx_row, lat_col])
                    lon_v = float(df.loc[idx_row, lon_col])
                    res = GPSResolver.lookup_gps(HARD_DRIVE_ROOT, lat_v, lon_v, max_dist_meters=500.0, verbose=False)
                    if res:
                        df.loc[idx_row, ts_col] = res["matched_timestamp"]
                except Exception:
                    pass

    if not ts_col: 
        UI.error("Neither 'timestamp' nor valid GPS ('latitude' & 'longitude') columns found in the file.")
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
    if 'failure_reason' not in df.columns: df['failure_reason'] = "" # NEW COLUMN
    
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

    # --- NEW: FAST BULK AVAILABILITY PRE-CHECK ---
    print(f"\n{UI.CYAN}[INFO] Performing Lightning-Fast Bulk Availability Pre-Check...{UI.RESET}")

    # Note: Uses global robust parse_target_timestamp supporting Unix, UTC ISO, IST, and multiple date formats

    target_dts = []
    for ts_str in df[ts_col]:
        target_dts.append(parse_target_timestamp(str(ts_str)))
        
    # --- NEW: Actually load the daily metadata JSONs into RAM! ---
    AssetDiscovery.load_metadata_from_drive(HARD_DRIVE_ROOT)
    
    # Collect candidate dirs for all batch timestamps
    all_candidates = set()
    for dt in target_dts:
        if dt:
            all_candidates.update(AssetDiscovery.find_candidate_dirs(HARD_DRIVE_ROOT, dt))
    if not all_candidates:
        all_candidates.add(HARD_DRIVE_ROOT)
        
    # Trigger filesystem scan into memory
    AssetDiscovery.pre_scan_directories(all_candidates)
    
    availability_list = []
    reason_list = [] # NEW: Capture the failure reasons
    
    for dt in target_dts:
        if not dt:
            availability_list.append('no')
            reason_list.append('Unparseable Timestamp format')
        else:
            # FIX 1: Unpack 4 variables and capture the error message
            c, j, m, err_msg = AssetDiscovery.search_for_timestamp(HARD_DRIVE_ROOT, dt, verbose=False)
            if c and j and err_msg == "OK":
                availability_list.append('yes')
                reason_list.append('')
            else:
                availability_list.append('no')
                reason_list.append(err_msg)
                
    df['available'] = availability_list
    df['failure_reason'] = reason_list # Save reasons to the DataFrame
    
    total_no = availability_list.count('no')
    total_yes = availability_list.count('yes')
    print(f"\n{UI.GREEN}Pre-Check Complete!{UI.RESET}")
    print(f"  -> {total_yes} Timestamps are AVAILABLE.")
    print(f"  -> {total_no} Timestamps are MISSING data.")
    
    safe_write_row(df.head(0), is_header=True)
    
    if total_yes == 0:
        UI.error("No valid data found for ANY timestamp in the CSV. Empty processed file generated. Exiting.")
        # Immediately dump all "no" rows into the file
        for idx, row in df.iterrows():
            safe_write_row(row.to_dict())
        return
        
    # --- IN-MEMORY OPTIMIZATION: LOAD AI ONCE ---
    print(f"\n{UI.CYAN}[INFO] Loading AI Models into RAM (This happens once)...{UI.RESET}")
    veh_detector = PanoramicImageDetector(YOLO_VEHICLE) if choice in ['3', '4'] else None
    ped_detector = PanoramicPedestrianDetector(YOLO_PED) if choice == '4' else None
    
    cached_registries = {}
    cached_mapped_data = {} 
    daily_stitcher_cache = {} 
    
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        
        # --- IMMEDIATE SKIP FOR 'NO' DATA ---
        if row_dict['available'] == 'no':
            print(f"[{idx+1}/{len(df)}] Skipping missing timestamp: {row_dict[ts_col]} (Reason: {row_dict['failure_reason']})")
            safe_write_row(row_dict)
            continue
            
        ts_str = str(row_dict[ts_col])
        target_dt = parse_target_timestamp(ts_str)
        date_key = target_dt.strftime("%Y-%m-%d")
        
        UI.header(f"Processing Batch {idx+1}/{len(df)}", HARD_DRIVE_ROOT, ROOT_OUTPUT)
        print(f"Target String: {ts_str} -> Model Target: {target_dt}")
        
        # FIX 2: Unpack 4 variables in the main processing loop
        csv_found, json_folder, session_meta, err_msg = AssetDiscovery.search_for_timestamp(HARD_DRIVE_ROOT, target_dt, verbose=False)
        
        row_class = extract_class_name(row_dict)
        # --- REGISTRY CACHING & MASTER MAPPING ---
        cache_key = f"{csv_found}_{json_folder}_{row_class}"
        if cache_key not in cached_registries:
            registry = build_session_registry(os.path.dirname(json_folder))
            mapping_output_path = os.path.join(ROOT_OUTPUT, "Master_Mapping.csv")
            mapped_data = process_imu_gps_file(csv_found, registry, mapping_output_path, lcz_class=row_class)
            
            cached_registries[cache_key] = registry
            cached_mapped_data[cache_key] = mapped_data
        else:
            registry = cached_registries[cache_key]
            mapped_data = cached_mapped_data[cache_key]
        
        if not mapped_data: 
            row_dict['available'] = 'no'
            row_dict['failure_reason'] = 'No matching timeframe inside the CSV'
            safe_write_row(row_dict); continue
        
        dur_val = float(row_dict[dur_col]) if dur_col else None
        
        if has_dur:
            res = clips_pipeline(mapped_data, registry, OMNI_JSON, ROOT_OUTPUT, YOLO_VEHICLE, YOLO_PED, target_dt, action_choice=choice, batch_mode=True, save_media=save_media, batch_dur=dur_val, class_name=row_class)
        else:
            res = frames_pipeline(mapped_data, registry, OMNI_JSON, ROOT_OUTPUT, YOLO_VEHICLE, YOLO_PED, target_dt, action_choice=choice, batch_mode=True, save_media=save_media, stitcher_cache=daily_stitcher_cache, date_key=date_key, veh_detector=veh_detector, ped_detector=ped_detector, class_name=row_class)
            
        if res == (None, None):
            row_dict['available'] = 'no'
            row_dict['failure_reason'] = 'Pipeline aborted (Time gap >3.0s or corrupt media)'
            print(f"{UI.YELLOW}Pipeline aborted for {ts_str}. Flipped to 'no'.{UI.RESET}")
        else:
            veh_counts, ped_count = res
            
            if choice in ['3', '4'] and modify_csv:
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
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    OMNI_JSON = os.path.join(PROJECT_ROOT, "FINAL_Intrinsics", "calibration_omni.json")
    if not os.path.exists(OMNI_JSON):
        OMNI_JSON = r"D:\MUMMAS\360ImageAnalysis_Vishwak\FINAL_Intrinsics\calibration_omni.json"

    YOLO_VEHICLE_MODEL = os.path.join(PROJECT_ROOT, "Vehicle_detection_360", "best.pt")
    if not os.path.exists(YOLO_VEHICLE_MODEL):
        YOLO_VEHICLE_MODEL = r"D:\MUMMAS\360ImageAnalysis_Vishwak\Vehicle_detection_360\best.pt"

    YOLO_PEDESTRIAN_MODEL = r"C:\viswak_MUMMAS_360degcamera\Insta360ImageAnalysis\yolo11m.pt"
    if not os.path.exists(YOLO_PEDESTRIAN_MODEL):
        for alt_path in [
            os.path.join(PROJECT_ROOT, "Human_detection_360", "yolo11m.pt"),
            os.path.join(PROJECT_ROOT, "Human_detection_360", "yolo11n.pt"),
            "yolo11m.pt",
            "yolo11n.pt"
        ]:
            if os.path.exists(alt_path):
                YOLO_PEDESTRIAN_MODEL = alt_path
                break
    
    ROOT_OUTPUT = UI.input_dir("Enter Master Output Directory (e.g., C:\\Results): ", create_if_missing=True)
    HARD_DRIVE_ROOT = UI.input_dir("Enter the Root Directory of the Hard Drive (e.g., D:\\): ") 

    # Prime metadata cache immediately
    AssetDiscovery.load_metadata_from_drive(HARD_DRIVE_ROOT) 

    while True:
        UI.header("360 CAMERA DATA RETRIEVAL HUB", HARD_DRIVE_ROOT, ROOT_OUTPUT)
        print(" 1) Batch Process via CSV/Excel (Supports Timestamps, Lat/Long, & LCZ Classes)")
        print(" 2) Single Process: By Timestamp (IST/UTC/Unix)")
        print(" 3) Single Process: By GPS Coordinates (Latitude, Longitude)")
        print(" 4) Single Process: By Folder Path & Time/Frame")
        print(" 5) Video Compression Tool (4K -> 1080p / 720p / 420p & Master Mapping)")
        print(" 6) Categorize Existing Results in Output Directory by LCZ / Class Name")
        print(f" {UI.RED}7) Quit{UI.RESET}\n")
        
        choice = UI.input("Select an option (1-7) [e.g., 2]: ")

        if choice == '7':
            UI.clear()
            print(f"{UI.CYAN}Exiting. Goodbye!{UI.RESET}")
            break

        if choice == '6':
            print(f"\n{UI.CYAN}--- Categorize Existing Results by LCZ / Class Name ---{UI.RESET}")
            target_dir = UI.input_dir(f"Enter Directory to categorize [default: {ROOT_OUTPUT}]: ", create_if_missing=False)
            if not target_dir:
                target_dir = ROOT_OUTPUT
            ref_csv = UI.input(r"Enter reference CSV with LCZ Class (optional, press Enter to search default MUMMAS CSVs): ").strip()
            categorize_existing_results(target_dir, csv_path=ref_csv if ref_csv else None, verbose=True)
            UI.pause()
            continue

        if choice == '5':
            try:
                from FINAL_4k_to_1080p import run_compression_interactive
                suggested_src = HARD_DRIVE_ROOT if HARD_DRIVE_ROOT else r"D:\MUMMAS\MUMMAS DATA COLLECTION"
                suggested_dst = os.path.normpath(suggested_src) + "-1080p"
                run_compression_interactive(default_src=suggested_src, default_dst=suggested_dst)
            except ImportError:
                import importlib.util
                comp_py = os.path.join(SCRIPT_DIR, "FINAL_4k_to_1080p.py")
                if os.path.exists(comp_py):
                    spec = importlib.util.spec_from_file_location("comp_mod", comp_py)
                    comp_mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(comp_mod)
                    suggested_src = HARD_DRIVE_ROOT if HARD_DRIVE_ROOT else r"D:\MUMMAS\MUMMAS DATA COLLECTION"
                    suggested_dst = os.path.normpath(suggested_src) + "-1080p"
                    comp_mod.run_compression_interactive(default_src=suggested_src, default_dst=suggested_dst)
                else:
                    UI.error("FINAL_4k_to_1080p.py not found in PIPELINE directory.")
            UI.pause()
            continue

        if choice == '1':
            csv_in = UI.input_file("Enter path to Batch CSV/Excel file: ")
            batch_process_csv(csv_in, ROOT_OUTPUT, HARD_DRIVE_ROOT, YOLO_VEHICLE_MODEL, YOLO_PEDESTRIAN_MODEL, OMNI_JSON)
            continue

        # Variables for Unified Processing (Options 2, 3, & 4)
        folder_path = None
        target_dt = None
        registry = []
        mapped_data = []
        fps = 29.97 # Standard Default
        f_no = 0
        start_ist_str = datetime.now().isoformat()

        if choice == '2':
            ts_str = UI.input("Enter Target Timestamp (IST, UTC, or Unix): ")
            target_dt = parse_target_timestamp(ts_str)
            if not target_dt:
                UI.error("Could not parse timestamp. Please use format YYYY-MM-DD HH:MM:SS")
                UI.pause()
                continue
                
            csv_path, json_folder, meta_data, err_msg = AssetDiscovery.search_for_timestamp(HARD_DRIVE_ROOT, target_dt)
            if not csv_path or not json_folder:
                UI.error(f"Required assets not found: {err_msg}")
                UI.pause()
                continue
                
            folder_path = json_folder
            registry = build_session_registry(os.path.dirname(json_folder))
            mapping_output_path = os.path.join(ROOT_OUTPUT, "Master_Mapping.csv")
            mapped_data = process_imu_gps_file(csv_path, registry, mapping_output_path)
            
            if not mapped_data:
                UI.error("Could not parse the IMU CSV successfully.")
                UI.pause()
                continue
            
            f_no = mapped_data[0].get('frame_no', 0) if mapped_data else 0

        elif choice == '3':
            try:
                lat_str = UI.input("Enter Target Latitude (e.g., 12.99150): ")
                target_lat = float(lat_str)
                lon_str = UI.input("Enter Target Longitude (e.g., 80.23370): ")
                target_lon = float(lon_str)
            except ValueError:
                UI.error("Invalid latitude or longitude. Please enter valid numbers.")
                UI.pause()
                continue

            radius_str = UI.input("Enter Max Search Radius in meters [default: 100m]: ").strip()
            try:
                max_radius = float(radius_str) if radius_str else 100.0
            except ValueError:
                max_radius = 100.0

            gps_res = GPSResolver.lookup_gps(HARD_DRIVE_ROOT, target_lat, target_lon, max_dist_meters=max_radius)
            if not gps_res:
                UI.error(f"No IMU GPS data available to match coordinates ({target_lat}, {target_lon}).")
                UI.pause()
                continue

            dist_m = gps_res["distance_meters"]
            ts_str = gps_res["matched_timestamp"]
            matched_lat = gps_res["matched_lat"]
            matched_lon = gps_res["matched_lon"]

            if dist_m > max_radius:
                UI.warn(f"Nearest GPS point is {dist_m:.1f}m away (exceeds {max_radius:.0f}m tolerance).")
                confirm = UI.ask_yes_no(f"Proceed using nearest timestamp {ts_str}?")
                if not confirm:
                    continue
            else:
                UI.success(f"GPS Match Found! Distance: {dist_m:.2f}m")
                print(f"  Target:    ({target_lat:.6f}, {target_lon:.6f})")
                print(f"  Matched:   ({matched_lat:.6f}, {matched_lon:.6f})")
                print(f"  Timestamp: {ts_str} (IST)")
                print(f"  Source:    {os.path.basename(gps_res['source_file'])}")

            target_dt = parse_target_timestamp(ts_str)
            if not target_dt:
                UI.error(f"Could not parse timestamp from GPS record: {ts_str}")
                UI.pause()
                continue

            csv_path, json_folder, meta_data, err_msg = AssetDiscovery.search_for_timestamp(HARD_DRIVE_ROOT, target_dt)
            if not csv_path or not json_folder:
                UI.error(f"Required assets not found for timestamp {ts_str}: {err_msg}")
                UI.pause()
                continue

            folder_path = json_folder
            registry = build_session_registry(os.path.dirname(json_folder))
            mapping_output_path = os.path.join(ROOT_OUTPUT, "Master_Mapping.csv")
            mapped_data = process_imu_gps_file(csv_path, registry, mapping_output_path)

            if not mapped_data:
                UI.error("Could not parse the IMU CSV successfully.")
                UI.pause()
                continue

            f_no = mapped_data[0].get('frame_no', 0) if mapped_data else 0

        elif choice == '4':
            folder_path = UI.input_dir("Enter exact path to the target Session Folder: ")
            target_dt = None # No specific target timestamp for manual mode

            tf_choice = UI.input("Enter 'f' for Frame number, or 's' for Playback Seconds [f/s]: ").lower()
            if tf_choice == 'f':
                try:
                    f_no = int(UI.input("Enter Frame Number: "))
                except ValueError:
                    UI.error("Invalid frame number.")
                    UI.pause()
                    continue
            else:
                try:
                    f_no = int(float(UI.input("Enter Playback Seconds: ")) * fps)
                except ValueError:
                    UI.error("Invalid playback seconds.")
                    UI.pause()
                    continue
            
            override_meta_tmp = {}
            mapping_output_path = os.path.join(ROOT_OUTPUT, "Master_Mapping.csv")
            mapped_data = process_manual_frame(folder_path, f_no, fps, override_meta_tmp, mapping_output_path)
            if not mapped_data:
                UI.error("Manual processing failed.")
                UI.pause()
                continue

        # Dynamic Extracted Lenses / FPS Detection
        lens1_path = None
        meta_path = os.path.join(folder_path, "session_meta.json") if folder_path else None
        for l1_cand in ["LENS1", "lens1"]:
            l1_dir = os.path.join(folder_path, l1_cand) if folder_path else None
            if l1_dir and os.path.exists(l1_dir):
                for vf in os.listdir(l1_dir):
                    if vf.lower().endswith('.mp4'):
                        lens1_path = os.path.join(l1_dir, vf)
                        break
            if lens1_path: break
        
        total_frames = 9000
        if lens1_path and os.path.exists(lens1_path):
            cap = cv2.VideoCapture(lens1_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 9000
            cap.release()

        if meta_path and os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8', errors='ignore') as f:
                    m = json.load(f)
                    val = m.get("created_utc") or m.get("start_time_utc") or m.get("start_utc") or m.get("start_time") or m.get("created_at") or m.get("timestamp")
                    start_dt = parse_utc_to_ist(val) if val else parse_timestamp_from_run_name(os.path.basename(folder_path))
                    if start_dt:
                        start_ist_str = start_dt.isoformat()
            except Exception as e:
                UI.warn(f"Failed to read session_meta.json cleanly: {e}")

        # 2. Decision: Image or Video
        p_choice = UI.input("Process as: 1) Image Frame  2) Video Clip : ")
        
        clip_dur = 0.0
        if p_choice == '2':
            max_available_dur = max(0, (total_frames - f_no) / fps)
            while True:
                try:
                    clip_dur = float(UI.input(f"Enter Clip Duration in seconds (Max available: {max_available_dur:.2f}s): "))
                    if clip_dur > max_available_dur:
                        UI.warn(f"Requested {clip_dur}s exceeds file capacity.")
                        confirm = UI.input(f"Cap to {max_available_dur:.2f}s and continue? [y/n]: ").lower()
                        if confirm == 'y':
                            clip_dur = max_available_dur
                            break
                    else:
                        break
                except ValueError:
                    UI.error("Please enter a valid number.")

        # ==========================================
        # 3 & 4. ROUTING TO PIPELINES WITH LCZ/CLASS
        # ==========================================
        single_class = auto_detect_class_from_csvs(target_dt) if target_dt else None
        
        if choice in ['2', '3']:
            # Option 2 (By Timestamp) & Option 3 (By GPS): Let the pipeline map the timestamp naturally. 
            if p_choice == '1':
                frames_pipeline(mapped_data, registry, OMNI_JSON, ROOT_OUTPUT, YOLO_VEHICLE_MODEL, YOLO_PEDESTRIAN_MODEL, target_dt, class_name=single_class)
            else:
                clips_pipeline(mapped_data, registry, OMNI_JSON, ROOT_OUTPUT, YOLO_VEHICLE_MODEL, YOLO_PEDESTRIAN_MODEL, target_dt, batch_dur=clip_dur, class_name=single_class)
                
        elif choice == '4':
            # Option 4 (Manual Input): Force the pipeline to use the exact folder and frame.
            elapsed_sec = f_no / fps if fps > 0 else 0.0
            opt3_ist = "Manual_Input"
            opt3_utc = None
            opt3_unix = "0.000"

            if start_ist_str:
                try:
                    start_dt = datetime.fromisoformat(start_ist_str)
                    exact_ist_dt = start_dt + timedelta(seconds=elapsed_sec)
                    opt3_ist = exact_ist_dt.strftime("%Y-%m-%d %H:%M:%S")
                    opt3_utc_dt = (exact_ist_dt - timedelta(hours=5, minutes=30)).replace(tzinfo=timezone.utc)
                    opt3_utc = opt3_utc_dt.strftime("%Y-%m-%d %H:%M:%S")
                    opt3_unix = f"{opt3_utc_dt.timestamp():.3f}"
                    if not single_class:
                        single_class = auto_detect_class_from_csvs(exact_ist_dt)
                except Exception:
                    pass

            override_meta = {
                "folder_path": folder_path,
                "frame_no": f_no,
                "fps": fps,
                "start_ist_str": start_ist_str,
                "source_folder": os.path.basename(folder_path),
                "ist_time": opt3_ist,
                "is_manual": True,
                "playback_time_sec": f"{elapsed_sec:.3f}",
                "clip_duration_sec": clip_dur,
                "imu_data": None,
                "utc_time": opt3_utc,
                "unix_time": opt3_unix,
                "class_name": single_class,
                "lcz_class": single_class
            }

            # Record Option 4 manual extraction into Master_Mapping.csv
            mapping_output_path = os.path.join(ROOT_OUTPUT, "Master_Mapping.csv")
            try:
                os.makedirs(ROOT_OUTPUT, exist_ok=True)
                file_exists = os.path.exists(mapping_output_path)
                has_lcz_in_file = False
                if file_exists:
                    try:
                        with open(mapping_output_path, 'r', encoding='utf-8') as f:
                            hdr = f.readline()
                            has_lcz_in_file = 'lcz_class' in hdr
                    except Exception: pass

                fieldnames = ["ist_time", "utc_time", "unix_time", "source_folder", "folder_path", "playback_time_sec", "frame_no"]
                if not file_exists or has_lcz_in_file:
                    fieldnames.append("lcz_class")

                with open(mapping_output_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow({
                        "ist_time": opt3_ist,
                        "utc_time": opt3_utc if opt3_utc else "N/A",
                        "unix_time": opt3_unix,
                        "source_folder": os.path.basename(folder_path),
                        "folder_path": folder_path,
                        "playback_time_sec": f"{elapsed_sec:.3f}",
                        "frame_no": f_no,
                        "lcz_class": single_class or ""
                    })
            except Exception as e:
                UI.warn(f"Could not append manual frame to Master_Mapping.csv: {e}")
            
            if p_choice == '1':
                frames_pipeline(mapped_data, registry, OMNI_JSON, ROOT_OUTPUT, YOLO_VEHICLE_MODEL, YOLO_PEDESTRIAN_MODEL, target_dt, override_meta=override_meta, class_name=single_class)
            else:
                clips_pipeline(mapped_data, registry, OMNI_JSON, ROOT_OUTPUT, YOLO_VEHICLE_MODEL, YOLO_PEDESTRIAN_MODEL, target_dt, override_meta=override_meta, class_name=single_class)

if __name__ == "__main__":
    # Add this line explicitly for Windows PyTorch multiprocessing
    import multiprocessing
    multiprocessing.freeze_support() 
    
    main()