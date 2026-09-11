#!/usr/bin/env python3
"""
Timestamp_mapping.py
====================
Maps real-world campaign annotation CSV files containing missing timestamps and
runtime folder names (e.g. Jan Feb Campaign Image Extraction) into pipeline-ready
generic input CSV files for 360 AI panoramic extraction and detection.

Key Capabilities:
1. Playback Time Parsing: Converts MM.SS, MM:SS, and Excel-truncated decimal formats
   into precise seconds offsets.
2. Direct Run Name Parsing: Extracts exact start timestamps from `run_YYYYMMDD_HHMMSS`
   and computes the target frame timestamp.
3. Dataset Drive Discovery: Scans local or external hard drives (e.g., D:\MUMMAS DATA COLLECTION)
   to resolve shorthand run identifiers like `TRIP1R1`, `TRIP2R1`, `R1`, `R2` to exact
   session folders and start times.
4. Robust Column Shifting: Transparently handles annotator column shifts (such as `lens1` in LONG).
5. Generic Pipeline Format: Outputs `MAPPED_<original_filename>.csv` with `category` and
   `timestamp` as primary columns, ready for direct feeding into `batch_process_csv`.

"""

import os
import sys
import re
import csv
import json
import argparse
from datetime import datetime, date, timedelta
from collections import defaultdict
from typing import Optional, Tuple, Dict, Any, List

# --- Terminal UI Styling ---
class UI:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @staticmethod
    def header(title: str):
        print(f"\n{UI.CYAN}{UI.BOLD}{'=' * 75}{UI.RESET}")
        print(f"{UI.CYAN}{UI.BOLD} {title}{UI.RESET}")
        print(f"{UI.CYAN}{UI.BOLD}{'=' * 75}{UI.RESET}\n")

    @staticmethod
    def info(msg: str):
        print(f"{UI.CYAN}[INFO]{UI.RESET} {msg}")

    @staticmethod
    def success(msg: str):
        print(f"{UI.GREEN}{UI.BOLD}[SUCCESS]{UI.RESET} {msg}")

    @staticmethod
    def warn(msg: str):
        print(f"{UI.YELLOW}[WARN]{UI.RESET} {msg}")

    @staticmethod
    def error(msg: str):
        print(f"{UI.RED}{UI.BOLD}[ERROR]{UI.RESET} {msg}")

    @staticmethod
    def input(msg: str) -> str:
        return input(f"{UI.BOLD}{msg}{UI.RESET}").strip()

    @staticmethod
    def input_file(msg: str) -> str:
        """Prompts the user until a valid existing file path is entered."""
        while True:
            raw = UI.input(msg).strip('"').strip("'")
            if not raw:
                UI.warn("Input cannot be empty. Please enter a file path.")
                continue
            if raw.lower() in ['q', 'quit', 'exit']:
                print("\nExiting.")
                sys.exit(0)
            if os.path.isfile(raw):
                return os.path.abspath(raw)
            UI.error(f"File not found: '{raw}'. Please enter a valid file path (or 'q' to quit).")

    @staticmethod
    def input_dir(msg: str, default_dir: Optional[str] = None) -> str:
        """Prompts the user for a directory path, with optional default."""
        prompt = f"{msg} (Default: {default_dir}): " if default_dir else f"{msg}: "
        while True:
            raw = UI.input(prompt).strip('"').strip("'")
            if not raw and default_dir:
                return os.path.abspath(default_dir)
            if raw.lower() in ['q', 'quit', 'exit']:
                print("\nExiting.")
                sys.exit(0)
            if os.path.isdir(raw):
                return os.path.abspath(raw)
            UI.error(f"Directory not found: '{raw}'. Please enter a valid directory path (or 'q' to quit).")


# ==============================================================================
# 1. PARSING UTILITIES
# ==============================================================================

def parse_playback_time(pt_str: Any) -> Optional[int]:
    """
    Parses video playback time into total seconds.
    Handles:
      - '2.58'   -> 2 min 58 sec = 178 sec
      - '119.52' -> 119 min 52 sec = 7192 sec
      - '03:28'  -> 3 min 28 sec = 208 sec
      - '00:00'  -> 0 sec
      - '19.1'   -> 19 min 10 sec (Excel auto-stripped trailing zero from 19.10)
      - '12.5'   -> 12 min 50 sec
    """
    if pt_str is None:
        return None
    s = str(pt_str).strip()
    if not s or s.lower() in ['nan', 'none', '']:
        return None

    # Format MM:SS or HH:MM:SS
    if ':' in s:
        parts = s.split(':')
        try:
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        except ValueError:
            return None

    # Format MM.SS or M.SS (with possible trailing 0 stripped by Excel)
    if '.' in s:
        parts = s.split('.')
        if len(parts) == 2:
            try:
                m = int(parts[0])
                sec_part = parts[1].strip()
                # If single digit (e.g. '1' from 19.1), it represents 10s of seconds
                if len(sec_part) == 1:
                    sec = int(sec_part + '0')
                else:
                    sec = int(sec_part[:2])
                return m * 60 + sec
            except ValueError:
                return None

    # Integer seconds
    try:
        val = float(s)
        return int(val)
    except ValueError:
        return None


def parse_date_string(date_str: Any) -> Optional[date]:
    """
    Normalizes various date formats into a standard date object.
    Supports: DD/MM/YYYY, DD-MM-YYYY, DD-MM-YY, 02-2026 (Feb 2, 2026).
    """
    if date_str is None:
        return None
    s = str(date_str).strip()
    if not s or s.lower() in ['nan', 'none', '']:
        return None

    # Special case from campaign sheet: '02-2026' represents '02-02-2026'
    if re.match(r'^0?2-2026$', s):
        return date(2026, 2, 2)

    for fmt in [
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d/%m/%y",
        "%d-%m-%y",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d%m%Y"
    ]:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def parse_timestamp_from_run_name(run_name: str) -> Optional[datetime]:
    """
    Extracts start datetime from session folder name (e.g. run_20260131_121529_5450 or run_20260117_103923).
    """
    if not run_name:
        return None
    m = re.search(r'run_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', str(run_name).strip())
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                            int(m.group(4)), int(m.group(5)), int(m.group(6)))
        except Exception:
            pass
    return None


# ==============================================================================
# 2. DATASET SCANNER & RUN RESOLVER
# ==============================================================================

class DatasetScanner:
    """
    Scans data directory / hard drive roots to map date and shorthand labels
    (e.g. TRIP1R1, R1, R2) to exact runtime folders and starting timestamps.
    """
    def __init__(self, search_dirs: Optional[List[str]] = None):
        self.search_dirs = [os.path.normpath(d) for d in (search_dirs or []) if d and os.path.exists(d)]
        # Key: (norm_date_str, shorthand_key) -> {'run_name': str, 'start_dt': datetime, 'end_dt': datetime, 'folder_path': str}
        self.run_index: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # Key: (norm_date_str, trip_str, run_idx_str) -> run dict
        self.trip_run_index: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        # Key: norm_date_str -> list of sorted run dicts
        self.date_runs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        # Key: run_name -> run dict
        self.name_to_run: Dict[str, Dict[str, Any]] = {}

        self.index_drives()

    def add_directory(self, directory: str):
        if directory and os.path.exists(directory):
            norm = os.path.normpath(directory)
            if norm not in self.search_dirs:
                self.search_dirs.append(norm)
                self._scan_root(norm)

    def index_drives(self):
        for root in self.search_dirs:
            self._scan_root(root)

    def _normalize_date_str(self, date_val: Any) -> str:
        """Converts date or string to DDMMYYYY format."""
        if isinstance(date_val, date):
            return date_val.strftime("%d%m%Y")
        s = str(date_val).strip()
        d_obj = parse_date_string(s)
        if d_obj:
            return d_obj.strftime("%d%m%Y")
        return s.replace("-", "").replace("/", "").replace("_", "")

    def _scan_root(self, root_dir: str):
        """Recursively scans root_dir for date folders and indexes runs."""
        skip_dirs = {'$recycle.bin', 'system volume information', '.git', '.venv', '__pycache__', 'test_output', 'results'}
        
        # 1. Check if root_dir contains cumulative_metadata.json
        cum_meta_path = os.path.join(root_dir, "cumulative_metadata.json")
        if os.path.exists(cum_meta_path):
            self._index_cumulative_metadata(cum_meta_path)

        try:
            entries = os.listdir(root_dir)
        except Exception:
            return

        for item in entries:
            item_path = os.path.join(root_dir, item)
            if not os.path.isdir(item_path) or item.lower() in skip_dirs:
                continue

            # Check if directory name is a date folder (e.g. 31012026 or 20260131)
            norm_date = None
            if len(item) == 8 and item.isdigit():
                # DDMMYYYY or YYYYMMDD
                if int(item[:2]) <= 31 and int(item[2:4]) <= 12 and int(item[4:]) >= 2020:
                    norm_date = item  # DDMMYYYY
                elif int(item[:4]) >= 2020 and int(item[4:6]) <= 12 and int(item[6:]) <= 31:
                    norm_date = f"{item[6:]}{item[4:6]}{item[:4]}"
            else:
                d_obj = parse_date_string(item)
                if d_obj:
                    norm_date = d_obj.strftime("%d%m%Y")

            if norm_date:
                self._index_date_folder(item_path, norm_date)

    def _index_cumulative_metadata(self, cum_meta_path: str):
        """Indexes runs directly from a root cumulative_metadata.json file."""
        try:
            with open(cum_meta_path, 'r', encoding='utf-8', errors='ignore') as cmf:
                data = json.load(cmf)
            if not isinstance(data, dict):
                return
            for raw_date_key, date_info in data.items():
                if not isinstance(date_info, dict):
                    continue
                d_obj = parse_date_string(raw_date_key)
                norm_date = d_obj.strftime("%d%m%Y") if d_obj else raw_date_key.replace("-", "").replace("/", "").replace("_", "")
                
                trips = date_info.get("trips", {})
                runs_found = []
                for trip_id, run_list in trips.items():
                    for r_idx, r_item in enumerate(run_list, 1):
                        r_name = r_item.get("run_id")
                        st_str = r_item.get("start_time_ist")
                        et_str = r_item.get("end_time_ist")
                        start_dt = None
                        end_dt = None
                        if st_str and st_str != "N/A":
                            try:
                                start_dt = datetime.strptime(st_str, "%Y-%m-%d %H:%M:%S")
                            except ValueError:
                                pass
                        if et_str and et_str != "N/A":
                            try:
                                end_dt = datetime.strptime(et_str, "%Y-%m-%d %H:%M:%S")
                            except ValueError:
                                pass
                        if not start_dt and r_name:
                            start_dt = parse_timestamp_from_run_name(r_name)
                        if r_name:
                            runs_found.append({
                                "run_name": r_name,
                                "start_dt": start_dt,
                                "end_dt": end_dt,
                                "folder_path": os.path.dirname(cum_meta_path),
                                "trip": str(trip_id),
                                "run_idx": r_idx,
                                "norm_date": norm_date
                            })
                if runs_found:
                    self._register_runs(norm_date, runs_found)
        except Exception:
            pass

    def _register_runs(self, norm_date: str, runs_found: List[Dict[str, Any]]):
        """Registers a list of runs into lookup indexes."""
        def get_sort_key(r):
            dt = r.get("start_dt")
            return (dt if dt else datetime.min, r.get("run_name", ""))

        runs_found.sort(key=get_sort_key)

        # Re-assign sequential run_idx per trip
        trip_counters = defaultdict(int)
        for r in runs_found:
            t = str(r.get("trip", "1"))
            trip_counters[t] += 1
            r["run_idx"] = trip_counters[t]

        # Store indexed runs
        for r in runs_found:
            r_name = r["run_name"]
            trip_str = str(r["trip"])
            idx_num = r["run_idx"]

            self.name_to_run[r_name] = r
            if r not in self.date_runs[norm_date]:
                self.date_runs[norm_date].append(r)

            # Register direct (norm_date, trip, run) lookup
            self.trip_run_index[(norm_date, trip_str, str(idx_num))] = r
            if trip_str == "1":
                self.trip_run_index[(norm_date, "", str(idx_num))] = r

            # Shorthand keys:
            keys = [
                f"TRIP{trip_str}R{idx_num}",
                f"TRIP{trip_str}_R{idx_num}",
                f"TRIP{trip_str}RUN{idx_num}",
                f"TRIP{trip_str}_RUN{idx_num}",
                f"TRIP{trip_str} RUN{idx_num}",
                f"T{trip_str}R{idx_num}",
                f"T{trip_str}_R{idx_num}"
            ]
            if trip_str == "1":
                keys.extend([
                    f"R{idx_num}",
                    f"RUN{idx_num}",
                    f"RUN {idx_num}",
                    f"TRIP R{idx_num}",
                    f"TRIPR{idx_num}",
                    f"TRIPRUN{idx_num}"
                ])
            for k in keys:
                norm_k = re.sub(r'[\s_\-]+', '', k).upper()
                self.run_index[(norm_date, norm_k)] = r

        # Also register global chronological run index for R<N> fallback
        for g_idx, r in enumerate(runs_found, 1):
            k_g = f"R{g_idx}"
            if (norm_date, k_g) not in self.run_index:
                self.run_index[(norm_date, k_g)] = r
                self.run_index[(norm_date, f"RUN{g_idx}")] = r

    def _index_date_folder(self, folder_path: str, norm_date: str):
        """Indexes runs in a date folder, via metadata.json or directory walk."""
        meta_file = os.path.join(folder_path, "metadata.json")
        runs_found = []

        # 1. Try reading metadata.json if present
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r', encoding='utf-8', errors='ignore') as mf:
                    data = json.load(mf)
                trips = data.get("trips", {})
                for trip_id, run_list in trips.items():
                    for r_idx, r_item in enumerate(run_list, 1):
                        r_name = r_item.get("run_id")
                        st_str = r_item.get("start_time_ist")
                        start_dt = None
                        if st_str and st_str != "N/A":
                            try:
                                start_dt = datetime.strptime(st_str, "%Y-%m-%d %H:%M:%S")
                            except ValueError:
                                pass
                        if not start_dt and r_name:
                            start_dt = parse_timestamp_from_run_name(r_name)

                        if r_name:
                            run_info = {
                                "run_name": r_name,
                                "start_dt": start_dt,
                                "folder_path": folder_path,
                                "trip": str(trip_id),
                                "run_idx": r_idx,
                                "norm_date": norm_date
                            }
                            runs_found.append(run_info)
            except Exception:
                pass

        # 2. If metadata.json was not present or yielded no runs, scan folder structure
        if not runs_found:
            runs_found = self._scan_date_folder_fs(folder_path, norm_date)

        if runs_found:
            self._register_runs(norm_date, runs_found)

    def _scan_date_folder_fs(self, date_dir: str, norm_date: str) -> List[Dict[str, Any]]:
        """Scans filesystem under date_dir for TRIP and CAMERA run folders."""
        runs = []
        trip_dirs = [d for d in os.listdir(date_dir)
                     if os.path.isdir(os.path.join(date_dir, d)) and "TRIP" in d.upper()]

        if trip_dirs:
            for t_dir in sorted(trip_dirs):
                t_match = re.search(r'\d+', t_dir)
                trip_num = int(t_match.group(0)) if t_match else 1
                t_path = os.path.join(date_dir, t_dir)
                
                # Check for CAMERA folder or run directly inside
                cam_path = os.path.join(t_path, "CAMERA")
                target_path = cam_path if os.path.exists(cam_path) else t_path
                
                run_folders = sorted([d for d in os.listdir(target_path)
                                      if os.path.isdir(os.path.join(target_path, d)) and d.startswith("run_")])
                for r_idx, r_name in enumerate(run_folders, 1):
                    start_dt = parse_timestamp_from_run_name(r_name)
                    runs.append({
                        "run_name": r_name,
                        "start_dt": start_dt,
                        "folder_path": os.path.join(target_path, r_name),
                        "trip": str(trip_num),
                        "run_idx": r_idx,
                        "norm_date": norm_date
                    })

                # Fallback: Check for video files or ffmpeg log if no run_ subfolders
                if not run_folders and os.path.exists(cam_path):
                    # Check for ffmpeg log in Lens subfolders
                    found_run_name = None
                    for root_cam, _, fnames in os.walk(cam_path):
                        for fn in fnames:
                            if fn.endswith(".log"):
                                try:
                                    with open(os.path.join(root_cam, fn), 'r', errors='ignore') as logf:
                                        content = logf.read(2048)
                                        m_log = re.search(r'run_\d{8}_\d{6}(?:_\d+)?', content)
                                        if m_log:
                                            found_run_name = m_log.group(0)
                                            break
                                except Exception:
                                    pass
                        if found_run_name:
                            break

                    if not found_run_name and norm_date == "16012026":
                        found_run_name = "run_20260116_160248_1126"

                    if found_run_name:
                        start_dt = parse_timestamp_from_run_name(found_run_name)
                        runs.append({
                            "run_name": found_run_name,
                            "start_dt": start_dt,
                            "folder_path": cam_path,
                            "trip": str(trip_num),
                            "run_idx": 1,
                            "norm_date": norm_date
                        })
        else:
            # Directly check for CAMERA folder or run_ folders under date_dir
            cam_path = os.path.join(date_dir, "CAMERA")
            target_path = cam_path if os.path.exists(cam_path) else date_dir
            try:
                run_folders = sorted([d for d in os.listdir(target_path)
                                      if os.path.isdir(os.path.join(target_path, d)) and d.startswith("run_")])
                for r_idx, r_name in enumerate(run_folders, 1):
                    start_dt = parse_timestamp_from_run_name(r_name)
                    runs.append({
                        "run_name": r_name,
                        "start_dt": start_dt,
                        "folder_path": os.path.join(target_path, r_name),
                        "trip": "1",
                        "run_idx": r_idx,
                        "norm_date": norm_date
                    })
            except Exception:
                pass

        return runs

    def resolve_run(
        self,
        date_val: Any,
        video_val: Optional[str] = None,
        trip_val: Optional[Any] = None,
        run_val: Optional[Any] = None,
        timestamp_val: Optional[Any] = None
    ) -> Tuple[Optional[str], Optional[datetime], str]:
        """
        Resolves video string, trip/run numbers, or timestamp + date into
        (resolved_runtime_folder, start_datetime, status_message).
        The returned resolved_runtime_folder is ALWAYS the exact run_xyz name (e.g. run_YYYYMMDD_HHMMSS...).
        """
        v_clean = str(video_val).strip() if video_val is not None else ""
        norm_date = self._normalize_date_str(date_val) if date_val else ""

        # 1. Check if video_val itself contains a full run timestamp (run_YYYYMMDD_HHMMSS...)
        m = re.search(r'run_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})(?:_\d+)?', v_clean)
        if m:
            start_dt = parse_timestamp_from_run_name(m.group(0))
            return m.group(0), start_dt, "OK (from run name)"

        # 2. Check if video_val is known directly in name_to_run
        if v_clean in self.name_to_run:
            info = self.name_to_run[v_clean]
            return info["run_name"], info["start_dt"], "OK (from dataset index)"

        # 3. Check explicit trip_val and run_val columns if provided
        t_raw = str(trip_val).strip() if trip_val is not None else ""
        r_raw = str(run_val).strip() if run_val is not None else ""
        if (t_raw or r_raw) and norm_date:
            m_t = re.search(r'\d+', t_raw)
            m_r = re.search(r'\d+', r_raw)
            t_num = m_t.group(0) if m_t else ("1" if t_raw else "")
            r_num = m_r.group(0) if m_r else ""
            if r_num:
                t_key = t_num if t_num else "1"
                if (norm_date, t_key, r_num) in self.trip_run_index:
                    info = self.trip_run_index[(norm_date, t_key, r_num)]
                    return info["run_name"], info["start_dt"], "OK (resolved from trip/run columns)"
                sh_key = f"TRIP{t_key}R{r_num}"
                if (norm_date, sh_key) in self.run_index:
                    info = self.run_index[(norm_date, sh_key)]
                    return info["run_name"], info["start_dt"], "OK (resolved from trip/run columns)"

        # 4. Check shorthand in v_clean (e.g. TRIP1R1, TRIP2R1, R1, TRIPR12, etc.)
        if v_clean and norm_date:
            shorthand_clean = re.sub(r'[\s_\-]+', '', v_clean).upper()
            if (norm_date, shorthand_clean) in self.run_index:
                info = self.run_index[(norm_date, shorthand_clean)]
                return info["run_name"], info["start_dt"], "OK (resolved shorthand)"

            # Check pattern: TRIP<T>R<N> or TRIP<T>RUN<N>
            m_tr = re.search(r'TRIP(\d+)R(?:UN)?(\d+)', shorthand_clean)
            if m_tr:
                t_key, r_key = m_tr.group(1), m_tr.group(2)
                if (norm_date, t_key, r_key) in self.trip_run_index:
                    info = self.trip_run_index[(norm_date, t_key, r_key)]
                    return info["run_name"], info["start_dt"], "OK (resolved trip/run shorthand)"

            # Check pattern: TRIPR<N> or TRIPRUN<N>
            m_tr1 = re.search(r'TRIPR(?:UN)?(\d+)', shorthand_clean)
            if m_tr1:
                r_key = m_tr1.group(1)
                if (norm_date, "1", r_key) in self.trip_run_index:
                    info = self.trip_run_index[(norm_date, "1", r_key)]
                    return info["run_name"], info["start_dt"], "OK (resolved trip1 shorthand)"
                r_target = int(r_key)
                runs_on_date = self.date_runs.get(norm_date, [])
                if 0 < r_target <= len(runs_on_date):
                    info = runs_on_date[r_target - 1]
                    return info["run_name"], info["start_dt"], "OK (resolved by run sequence)"

            # Check pattern: R<N> or RUN<N>
            m_r = re.match(r'^R(?:UN)?(\d+)$', shorthand_clean)
            if m_r:
                target_idx = int(m_r.group(1))
                runs_on_date = self.date_runs.get(norm_date, [])
                if 0 < target_idx <= len(runs_on_date):
                    info = runs_on_date[target_idx - 1]
                    return info["run_name"], info["start_dt"], "OK (resolved by run index)"

            # Check pattern: T<T>R<N>
            m_t_r = re.match(r'^T(\d+)R(\d+)$', shorthand_clean)
            if m_t_r:
                t_key, r_key = m_t_r.group(1), m_t_r.group(2)
                if (norm_date, t_key, r_key) in self.trip_run_index:
                    info = self.trip_run_index[(norm_date, t_key, r_key)]
                    return info["run_name"], info["start_dt"], "OK (resolved T<T>R<N>)"

        # 5. Check NA / blank on a date with a single run (e.g. 16/01/2026)
        clean_upper = re.sub(r'[\s_\-]+', '', v_clean).upper()
        if clean_upper in ["NA", "N/A", "NONE", ""] and norm_date:
            runs_on_date = self.date_runs.get(norm_date, [])
            if len(runs_on_date) == 1:
                info = runs_on_date[0]
                return info["run_name"], info["start_dt"], "OK (single run for date)"

        # 6. Check timestamp matching across runs on that date
        if timestamp_val and norm_date:
            target_ts = None
            if isinstance(timestamp_val, datetime):
                target_ts = timestamp_val
            elif isinstance(timestamp_val, str):
                s_ts = timestamp_val.strip()
                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%H:%M:%S"]:
                    try:
                        p_ts = datetime.strptime(s_ts, fmt)
                        if fmt == "%H:%M:%S" and date_val:
                            d_obj = parse_date_string(date_val)
                            if d_obj:
                                target_ts = datetime(d_obj.year, d_obj.month, d_obj.day, p_ts.hour, p_ts.minute, p_ts.second)
                        else:
                            target_ts = p_ts
                        break
                    except ValueError:
                        pass

            if target_ts:
                runs_on_date = self.date_runs.get(norm_date, [])
                for r in runs_on_date:
                    st = r.get("start_dt")
                    et = r.get("end_dt")
                    if st and et and st <= target_ts <= et:
                        return r["run_name"], r["start_dt"], "OK (matched by timestamp window)"

                # Proximity match within 15 minutes of start_dt
                best_r = None
                min_diff = 15 * 60
                for r in runs_on_date:
                    st = r.get("start_dt")
                    if st:
                        diff = abs((target_ts - st).total_seconds())
                        if diff < min_diff:
                            min_diff = diff
                            best_r = r
                if best_r:
                    return best_r["run_name"], best_r["start_dt"], "OK (matched by timestamp proximity)"

        return None, None, f"Unresolved run for date '{date_val}' (video='{video_val}', trip='{trip_val}', run='{run_val}')"


# ==============================================================================
# 3. CORE CSV TIMESTAMP MAPPING ENGINE
# ==============================================================================

def map_csv_timestamps(
    input_csv_path: str,
    output_csv_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    verbose: bool = True
) -> str:
    """
    Reads an input real-world campaign CSV, resolves missing timestamps using
    playback time and video metadata, and outputs a clean generic CSV ready for
    batch processing by the AI pipeline.
    """
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv_path}")

    # Determine default output path if not provided
    if not output_csv_path:
        dir_name = os.path.dirname(os.path.abspath(input_csv_path))
        base_name = os.path.basename(input_csv_path)
        output_csv_path = os.path.join(dir_name, f"MAPPED_{base_name}")

    if verbose:
        UI.header("TIMESTAMP MAPPING & RUN RESOLUTION HUB")
        UI.info(f"Input CSV   : {input_csv_path}")
        UI.info(f"Output CSV  : {output_csv_path}")

    # Determine search directories for data
    search_dirs = []
    if data_dir and os.path.exists(data_dir):
        search_dirs.append(data_dir)

    default_paths = [
        r"E:\MUMMAS DATA COLLECTION-1080p",
        r"D:\MUMMAS DATA COLLECTION-1080p",
        r"D:\MUMMAS\MUMMAS DATA COLLECTION-1080p",
        r"D:\MUMMAS\MUMMAS DATA COLLECTION",
        r"D:\MUMMAS DATA COLLECTION",
        r"D:\31012026",
        r"D:\\",
        r"E:\\"
    ]
    for p in default_paths:
        if os.path.exists(p) and p not in search_dirs:
            search_dirs.append(p)

    if verbose:
        UI.info(f"Scanning {len(search_dirs)} directory paths for run sessions...")
        for sd in search_dirs:
            print(f"   -> {sd}")

    scanner = DatasetScanner(search_dirs)
    if verbose:
        total_runs_indexed = len(scanner.name_to_run)
        UI.info(f"Total session runs indexed in RAM: {total_runs_indexed}")

    # Read raw CSV lines with utf-8-sig to safely handle BOM
    with open(input_csv_path, 'r', encoding='utf-8-sig', errors='replace') as f:
        reader = csv.reader(f)
        all_rows = list(reader)

    if not all_rows:
        raise ValueError("Input CSV file is empty.")

    # Locate column positions from header
    header = [c.strip() for c in all_rows[0]]
    
    def find_col_idx(candidates: List[str], default_idx: int, substring_keywords: Optional[List[str]] = None) -> int:
        for idx, col in enumerate(header):
            c_low = col.strip().lower()
            if c_low in candidates:
                return idx
        if substring_keywords:
            for idx, col in enumerate(header):
                c_low = col.strip().lower()
                if any(kw in c_low for kw in substring_keywords):
                    return idx
        return default_idx

    col_sno = find_col_idx(['s no', 'sno', 's.no', 'sl no', 'sl.no', 'id'], -1, ['s no', 'sno', 's.no', 'sl no', 'sl.no'])
    col_date = find_col_idx(['date', 'dates', 'collection_date'], -1, ['date'])
    col_video = find_col_idx(['video name', 'video', 'video_name', 'videoname', 'run name', 'run_name', 'run_xyz', 'run id', 'run_id'], -1, ['video', 'run_name', 'run name'])
    col_folder = find_col_idx(['runtime_folder', 'run_folder', 'run folder', 'session', 'folder'], -1, ['folder', 'session'])
    col_trip = find_col_idx(['trip no', 'trip_no', 'trip', 'trip number', 'trip num', 'trip_num', 'trip id', 'trip_id'], -1, ['trip'])
    col_run = find_col_idx(['run no', 'run_no', 'run number', 'run num', 'run_num', 'run'], -1, ['run no', 'run_no', 'run num'])
    col_ts = find_col_idx(['time stamp', 'timestamp', 'time_stamp', 'target_time', 'ist_time', 'datetime', 'date_time'], -1, ['time stamp', 'timestamp', 'time_stamp', 'target_time', 'ist_time'])
    col_lat = find_col_idx(['lat', 'latitude', 'y', 'gnss_latitude', 'filter_lla_lat'], -1, ['lat', 'latitude'])
    col_long = find_col_idx(['long', 'longitude', 'x', 'lon', 'gnss_longitude', 'filter_lla_lon'], -1, ['long', 'longitude', 'lon'])
    col_cat = find_col_idx(['category', 'class', 'lcz class', 'lcz', 'class_name', 'classname', 'label'], -1, ['cat', 'class', 'lcz', 'label'])
    col_pt = find_col_idx(['playback time', 'playback_time', 'playbacktime', 'playback', 'playback time in minutes'], -1, ['playback'])

    mapped_rows = []
    cur_date_str = ""
    cur_video_str = ""
    cur_trip_str = ""
    cur_run_str = ""
    cur_date_obj: Optional[date] = None

    stats = {
        "total_rows": 0,
        "mapped_existing_ts": 0,
        "mapped_run_name": 0,
        "mapped_shorthand": 0,
        "unmapped_no_time": 0,
        "unmapped_drive_missing": 0,
        "empty_skipped": 0
    }

    for row_idx, row in enumerate(all_rows[1:], start=2):
        # Skip completely blank lines
        if not any(cell.strip() for cell in row):
            stats["empty_skipped"] += 1
            continue

        def get_cell(idx: int) -> str:
            return row[idx].strip() if 0 <= idx < len(row) else ""

        raw_sno = get_cell(col_sno)
        raw_date = get_cell(col_date)
        raw_video = get_cell(col_video)
        raw_folder = get_cell(col_folder)
        raw_trip = get_cell(col_trip)
        raw_run = get_cell(col_run)
        raw_ts = get_cell(col_ts)
        raw_lat = get_cell(col_lat)
        raw_long = get_cell(col_long)
        raw_cat = get_cell(col_cat)
        raw_pt = get_cell(col_pt)

        # Update forward-filled date
        if raw_date and raw_date.lower() != 'nan':
            cur_date_str = raw_date
            cur_date_obj = parse_date_string(raw_date)

        # Update forward-filled trip and run
        if raw_trip and raw_trip.lower() != 'nan':
            cur_trip_str = raw_trip
        if raw_run and raw_run.lower() != 'nan':
            cur_run_str = raw_run

        # Update forward-filled video name or folder
        if raw_video and raw_video.lower() != 'nan':
            cur_video_str = raw_video
        elif raw_folder and raw_folder.lower() != 'nan':
            cur_video_str = raw_folder

        # If row has no category, timestamp, playback time, or video/trip info, skip
        if not raw_cat and not raw_ts and not raw_pt and not raw_video and not (cur_trip_str or cur_run_str):
            stats["empty_skipped"] += 1
            continue

        stats["total_rows"] += 1

        # Clean category name
        category_clean = raw_cat.strip() if raw_cat else ""
        if category_clean.upper() in ['NONE', 'NA', 'N/A']:
            category_clean = "NONE"

        # Determine effective date from video run name if available
        # (e.g. if video is run_20260119_... but date says 19/05/2026, the run name is authoritative)
        m_run_date = re.search(r'run_(\d{4})(\d{2})(\d{2})', cur_video_str)
        effective_date_obj = cur_date_obj
        if m_run_date:
            try:
                effective_date_obj = date(int(m_run_date.group(1)), int(m_run_date.group(2)), int(m_run_date.group(3)))
            except ValueError:
                pass

        final_timestamp = ""
        resolved_folder = ""
        pt_sec = parse_playback_time(raw_pt)

        # -------------------------------------------------------------
        # Case A: Row already has explicit time stamp (e.g. 10:59:31)
        # -------------------------------------------------------------
        if raw_ts and raw_ts.lower() not in ['nan', 'none', '']:
            clean_ts = raw_ts.strip()
            # If timestamp already includes date
            if len(clean_ts) >= 19 and '-' in clean_ts:
                final_timestamp = clean_ts
            elif effective_date_obj:
                # Combine date + HH:MM:SS
                final_timestamp = f"{effective_date_obj.strftime('%Y-%m-%d')} {clean_ts}"
            else:
                final_timestamp = clean_ts

            # Resolve the runtime folder to run_xyz!
            res_folder, _, _ = scanner.resolve_run(
                date_val=effective_date_obj or cur_date_str,
                video_val=cur_video_str,
                trip_val=cur_trip_str,
                run_val=cur_run_str,
                timestamp_val=final_timestamp
            )
            resolved_folder = res_folder or cur_video_str
            stats["mapped_existing_ts"] += 1

        # -------------------------------------------------------------
        # Case B: Compute timestamp using Playback Time & Run Start Time
        # -------------------------------------------------------------
        elif pt_sec is not None:
            # 1. Resolve runtime folder and start datetime
            res_folder, start_dt, status = scanner.resolve_run(
                date_val=effective_date_obj or cur_date_str,
                video_val=cur_video_str,
                trip_val=cur_trip_str,
                run_val=cur_run_str
            )
            resolved_folder = res_folder or cur_video_str

            if start_dt is not None:
                # Compute exact timestamp
                target_dt = start_dt + timedelta(seconds=pt_sec)
                final_timestamp = target_dt.strftime("%Y-%m-%d %H:%M:%S")

                if "from run name" in status:
                    stats["mapped_run_name"] += 1
                else:
                    stats["mapped_shorthand"] += 1
            else:
                # Shorthand could not be resolved because date folder is not connected
                stats["unmapped_drive_missing"] += 1
        else:
            # No playback time and no explicit timestamp
            res_folder, _, _ = scanner.resolve_run(
                date_val=effective_date_obj or cur_date_str,
                video_val=cur_video_str,
                trip_val=cur_trip_str,
                run_val=cur_run_str
            )
            resolved_folder = res_folder or cur_video_str
            stats["unmapped_no_time"] += 1

        # Clean longitude if it contained lens1 artifact
        clean_lat = raw_lat if not raw_lat.lower().startswith("lens") else ""
        clean_long = raw_long if not raw_long.lower().startswith("lens") else ""

        mapped_rows.append({
            "category": category_clean,
            "timestamp": final_timestamp,
            "runtime_folder": resolved_folder,
            "date": effective_date_obj.strftime("%Y-%m-%d") if effective_date_obj else cur_date_str,
            "video_name": resolved_folder,
            "playback_time": raw_pt,
            "playback_seconds": pt_sec if pt_sec is not None else "",
            "LAT": clean_lat,
            "LONG": clean_long,
            "s_no": raw_sno
        })

    # -----------------------------------------------------------------
    # Save output CSV file
    # -----------------------------------------------------------------
    fieldnames = [
        "category",
        "timestamp",
        "runtime_folder",
        "date",
        "video_name",
        "playback_time",
        "playback_seconds",
        "LAT",
        "LONG",
        "s_no"
    ]

    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    while True:
        try:
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as out_f:
                writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(mapped_rows)
            break
        except PermissionError:
            print(f"\n{UI.RED}{UI.BOLD}[BLOCKED] Windows denied write access to: {output_csv_path}{UI.RESET}")
            print(f"{UI.YELLOW}Is '{os.path.basename(output_csv_path)}' open in Excel?{UI.RESET}")
            ans = UI.input("Close the file in Excel and press Enter to retry, or type 'alt' to save to a new file: ")
            if ans.strip().lower() in ['alt', 'a', 'new']:
                base, ext = os.path.splitext(output_csv_path)
                output_csv_path = f"{base}_{datetime.now().strftime('%H%M%S')}{ext}"
                UI.info(f"Saving to new file: {output_csv_path}")

    total_mapped = stats["mapped_existing_ts"] + stats["mapped_run_name"] + stats["mapped_shorthand"]

    if verbose:
        UI.success(f"Mapping Completed Successfully!")
        print(f"  -> Total Data Rows Analyzed   : {stats['total_rows']}")
        print(f"  -> Successfully Mapped Rows   : {total_mapped} / {stats['total_rows']}")
        print(f"     * From Existing Timestamp  : {stats['mapped_existing_ts']}")
        print(f"     * From Direct Run Timestamp: {stats['mapped_run_name']}")
        print(f"     * From Dataset Scan (R/TRIP): {stats['mapped_shorthand']}")
        if stats['unmapped_drive_missing'] > 0:
            print(f"  -> Unresolved Shorthands      : {UI.YELLOW}{stats['unmapped_drive_missing']}{UI.RESET} (Mount external data drive with dates to resolve)")
        if stats['unmapped_no_time'] > 0:
            print(f"  -> Missing Playback Time Rows : {stats['unmapped_no_time']}")
        print(f"  -> Saved Output File          : {UI.GREEN}{output_csv_path}{UI.RESET}\n")

    return output_csv_path


# ==============================================================================
# 4. COMMAND LINE INTERFACE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Map real-world campaign CSV timestamps and runtime folders using playback time."
    )
    parser.add_argument(
        "-i", "--input",
        dest="input_csv",
        default=None,
        help="Path to the real-world input CSV file."
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_csv",
        default=None,
        help="Optional path for the output mapped CSV file."
    )
    parser.add_argument(
        "-d", "--data_dir",
        dest="data_dir",
        default=None,
        help="Path to the root data collection directory or external hard drive."
    )

    args = parser.parse_args()

    # Determine input CSV: prompt user if not provided via CLI
    input_path = args.input_csv
    if not input_path:
        UI.header("360 CAMPAIGN TIMESTAMP & RUN MAPPING")
        input_path = UI.input_file("Enter the path to the CSV file to map: ")
    elif not os.path.isfile(input_path):
        UI.error(f"Provided file does not exist: '{input_path}'")
        input_path = UI.input_file("Enter the path to the CSV file to map: ")

    # Determine Data Collection Root / Hard Drive path
    data_path = args.data_dir
    if not data_path:
        # Check standard default candidate paths
        cand_default = None
        for cand in [
            r"E:\MUMMAS DATA COLLECTION-1080p",
            r"D:\MUMMAS DATA COLLECTION-1080p",
            r"D:\MUMMAS\MUMMAS DATA COLLECTION-1080p",
            r"D:\MUMMAS\MUMMAS DATA COLLECTION",
            r"D:\MUMMAS DATA COLLECTION",
            r"D:\\",
            r"E:\\"
        ]:
            if os.path.exists(cand):
                cand_default = cand
                break

        data_path = UI.input_dir(
            "Enter Data Collection Root / Hard Drive path (or press Enter for default)",
            default_dir=cand_default
        )

    map_csv_timestamps(
        input_csv_path=input_path,
        output_csv_path=args.output_csv,
        data_dir=data_path,
        verbose=True
    )


if __name__ == "__main__":
    main()
