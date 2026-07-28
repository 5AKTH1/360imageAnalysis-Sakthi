import os
import csv
import json
import time
from datetime import datetime, timedelta
import sys

try:
    import pandas as pd
    import cv2 
except ImportError:
    print("[ERROR] Missing libraries. Run: pip install pandas opencv-python")
    sys.exit(1)

# ==========================================
# 1. UI HELPERS (Fixed AttributeError)
# ==========================================
if os.name == 'nt': os.system('') 

class UI:
    CYAN, GREEN, YELLOW, RED, MAGENTA, BOLD, RESET = '\033[96m', '\033[92m', '\033[93m', '\033[91m', '\033[95m', '\033[1m', '\033[0m'
    
    @staticmethod
    def header(title):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"{UI.MAGENTA}{UI.BOLD}{'='*65}{UI.RESET}")
        print(f"{UI.CYAN}{UI.BOLD}{title.center(65)}{UI.RESET}")
        print(f"{UI.MAGENTA}{UI.BOLD}{'='*65}{UI.RESET}")

    @staticmethod
    def info(msg): print(f"{UI.CYAN}[INFO]{UI.RESET} {msg}")
    @staticmethod
    def success(msg): print(f"{UI.GREEN}[SUCCESS]{UI.RESET} {msg}")
    @staticmethod
    def warn(msg): print(f"{UI.YELLOW}[WARN]{UI.RESET} {msg}")
    @staticmethod
    def error(msg): print(f"{UI.RED}[ERROR]{UI.RESET} {msg}")

# ==========================================
# 2. CORE SCANNER LOGIC
# ==========================================

class DeepScanner:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.valid_intervals = [] 

    def _get_video_duration(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): return 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            return duration
        except: return 0

    def _verify_and_get_vid_range(self, session_folder, json_path):
        try:
            # Check all 6 lenses
            for i in range(1, 7):
                if not os.path.exists(os.path.join(session_folder, f"LENS{i}", f"video_lens{i}.mp4")):
                    return None

            with open(json_path, 'r') as f:
                meta = json.load(f)
                utc_str = meta.get("created_utc", "").replace('Z', '').split('.')[0]
                start_ist = datetime.fromisoformat(utc_str) + timedelta(hours=5, minutes=30)

            lens1_path = os.path.join(session_folder, "LENS1", "video_lens1.mp4")
            duration = self._get_video_duration(lens1_path)
            
            if duration == 0: return None
            return (start_ist, start_ist + timedelta(seconds=duration))
        except: return None

    def _get_imu_range(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line for line in f if line.strip() and not line.strip().startswith('#')]
                if len(lines) < 2: return None
                reader = csv.DictReader(lines)
                # Clean column names for matching
                reader.fieldnames = [fn.strip().lower() for fn in reader.fieldnames]
                time_col = next((c for c in reader.fieldnames if 'timestamp_local' in c), None)
                if not time_col: return None
                
                start_dt = datetime.strptime(next(reader)[time_col].split('.')[0], "%Y-%m-%d %H:%M:%S")
                
                # Fast seek last line
                last_line = lines[-1].split(',')
                # find index of time_col
                idx = reader.fieldnames.index(time_col)
                end_dt = datetime.strptime(last_line[idx].split('.')[0], "%Y-%m-%d %H:%M:%S")
                return (start_dt, end_dt)
        except: return None

    def build_index(self):
        imu_ranges = []
        vid_ranges = []
        
        UI.info("Building global data index. This probes video headers for accuracy...")
        for root, _, files in os.walk(self.root_dir):
            for f in files:
                full_path = os.path.join(root, f)
                low_f = f.lower()
                
                if low_f.endswith('.csv') and ('imu' in low_f or 'imu-gps' in root.lower()):
                    r = self._get_imu_range(full_path)
                    if r: imu_ranges.append(r)
                
                if low_f == "session_meta.json":
                    r = self._verify_and_get_vid_range(root, full_path)
                    if r: vid_ranges.append(r)

        imu_timeline = self._merge(imu_ranges)
        vid_timeline = self._merge(vid_ranges)
        self.valid_intervals = self._intersect(imu_timeline, vid_timeline)
        UI.success(f"Index Built! Found {len(self.valid_intervals)} verified data segments.")

    def _merge(self, intervals):
        if not intervals: return []
        intervals.sort()
        merged = [intervals[0]]
        for curr in intervals[1:]:
            prev = merged[-1]
            if curr[0] <= prev[1] + timedelta(seconds=2):
                merged[-1] = (prev[0], max(prev[1], curr[1]))
            else:
                merged.append(curr)
        return merged

    def _intersect(self, list1, list2):
        i = j = 0
        res = []
        while i < len(list1) and j < len(list2):
            s = max(list1[i][0], list2[j][0])
            e = min(list1[i][1], list2[j][1])
            if s < e: res.append((s, e))
            if list1[i][1] < list2[j][1]: i += 1
            else: j += 1
        return res

    def get_status(self, req_s, req_e):
        overlaps = []
        for s, e in self.valid_intervals:
            intersect_s = max(s, req_s)
            intersect_e = min(e, req_e)
            if intersect_s < intersect_e:
                overlaps.append((intersect_s, intersect_e))
        
        if not overlaps: return "no"
        total_sec = sum((m[1]-m[0]).total_seconds() for m in overlaps)
        req_sec = (req_e - req_s).total_seconds()
        
        if total_sec >= req_sec - 5: return "yes"
        parts = "".join([f"({m[0].strftime('%H:%M:%S')} - {m[1].strftime('%H:%M:%S')})" for m in overlaps])
        return f"yes{parts}"

# ==========================================
# 3. MAIN INTERFACE
# ==========================================

def main():
    UI.header("ULTIMATE 360 DATA VALIDATOR")
    csv_file = input(f"{UI.BOLD}Enter Input CSV Path: {UI.RESET}").strip()
    drive = input(f"{UI.BOLD}Enter Drive Root: {UI.RESET}").strip()
    
    scanner = DeepScanner(drive)
    scanner.build_index()
    
    try:
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()
    except Exception as e:
        UI.error(f"Error reading CSV: {e}"); return

    results = []
    for idx, row in df.iterrows():
        try:
            d = str(row['Date']).strip().replace('/', '-')
            ts_s = datetime.strptime(f"{d} {str(row['Start_Time']).strip()}", "%d-%m-%Y %H:%M:%S")
            ts_e = datetime.strptime(f"{d} {str(row['End_Time']).strip()}", "%d-%m-%Y %H:%M:%S")
            
            status = scanner.get_status(ts_s, ts_e)
            results.append(status)
            print(f"[{idx+1}] Processing: {d} | Result: {status}")
        except:
            results.append("Format Error")

    df['available'] = results
    out = csv_file.replace(".csv", "_validated.csv")
    df.to_csv(out, index=False)
    UI.success(f"Validation complete! File saved as: {out}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{UI.YELLOW}Process stopped by user.{UI.RESET}")