import os
import shutil
import subprocess
import json
import csv
import re
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

MAX_CONCURRENT_VIDEOS = 2 

RESOLUTIONS = {
    "1080p": {
        "name": "1080p (Full HD)",
        "scale_cuda": "scale_cuda=1920:1080",
        "scale_cpu": "scale=1920:1080",
        "width": 1920,
        "height": 1080
    },
    "720p": {
        "name": "720p (HD)",
        "scale_cuda": "scale_cuda=1280:720",
        "scale_cpu": "scale=1280:720",
        "width": 1280,
        "height": 720
    },
    "420p": {
        "name": "420p (Low Res)",
        "scale_cuda": "scale_cuda=746:420",
        "scale_cpu": "scale=746:420",
        "width": 746,
        "height": 420
    },
    "480p": {
        "name": "480p (SD)",
        "scale_cuda": "scale_cuda=854:480",
        "scale_cpu": "scale=854:480",
        "width": 854,
        "height": 480
    }
}

def parse_utc_to_ist(utc_str):
    """Converts UTC / ISO timestamp string to IST naive datetime."""
    if not utc_str:
        return None
    s = str(utc_str).strip().replace("Z", "+00:00").replace("z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is not None:
            utc_dt = dt.astimezone(timezone.utc)
            return (utc_dt + timedelta(hours=5, minutes=30)).replace(tzinfo=None)
        return dt + timedelta(hours=5, minutes=30)
    except Exception:
        pass

    clean = s.replace("T", " ").split(".")[0]
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%d-%m-%Y %H:%M:%S"]:
        try:
            return datetime.strptime(clean, fmt) + timedelta(hours=5, minutes=30)
        except ValueError:
            pass
    return None

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

def process_single_file(src_file, dest_file, do_compress=True, resolution="1080p"):
    """
    Processes a single file. If do_compress is True and file is .mp4,
    downscales using CUDA NVENC with automatic fallback to CPU libx264.
    If do_compress is False or file is non-video, copies the file directly.
    """
    if src_file.lower().endswith(".mp4") and do_compress:
        res_cfg = RESOLUTIONS.get(resolution.lower(), RESOLUTIONS["1080p"])
        scale_cuda = res_cfg["scale_cuda"]
        scale_cpu = res_cfg["scale_cpu"]
        
        print(f"[COMPRESSING -> {res_cfg['name']}] {src_file}...")
        
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        
        # 1. Attempt GPU Acceleration (CUDA NVENC)
        # 'medium' is universally supported across both legacy and modern NVENC builds (maps to p4 / HQ 1-pass)
        cmd_cuda = [
            "ffmpeg", 
            "-hwaccel", "cuda", 
            "-hwaccel_device", "0",
            "-hwaccel_output_format", "cuda", 
            "-i", src_file,
            "-vf", scale_cuda, 
            "-c:v", "h264_nvenc",  
            "-cq", "28",          
            "-preset", "medium",       
            "-c:a", "aac", 
            "-y", 
            dest_file
        ]
        
        try:
            subprocess.run(cmd_cuda, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            print(f"[DONE (GPU)] {dest_file}")
            return True
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.decode("utf-8", errors="replace").strip() if e.stderr else str(e)
            # If preset was rejected by an unusual build, retry once with 'fast'
            if "preset" in err_msg.lower():
                try:
                    cmd_cuda_retry = [arg for i, arg in enumerate(cmd_cuda) if arg != "-preset" and (i == 0 or cmd_cuda[i-1] != "-preset")] + ["-preset", "fast"]
                    subprocess.run(cmd_cuda_retry, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    print(f"[DONE (GPU - fallback preset)] {dest_file}")
                    return True
                except Exception:
                    pass
            last_err = "\n   ".join(err_msg.splitlines()[-4:]) if err_msg else str(e)
            print(f"[GPU FAILED / FALLBACK TO CPU] {src_file}\n   Reason: {last_err}")
        except FileNotFoundError:
            print(f"[GPU FAILED / FALLBACK TO CPU] {src_file}: 'ffmpeg' executable was not found in system PATH.")
            
        # 2. CPU Fallback
        cmd_cpu = [
            "ffmpeg", 
            "-i", src_file,
            "-vf", scale_cpu, 
            "-c:v", "libx264",  
            "-crf", "28",          
            "-preset", "fast",       
            "-c:a", "aac", 
            "-y", 
            dest_file
        ]
        try:
            subprocess.run(cmd_cpu, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            print(f"[DONE (CPU)] {dest_file}")
            return True
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.decode("utf-8", errors="replace").strip() if e.stderr else str(e)
            last_err = "\n   ".join(err_msg.splitlines()[-4:]) if err_msg else str(e)
            print(f"[ERROR] Failed to compress {src_file} on CPU as well:\n   Reason: {last_err}")
            return False
    else:
        # User opted not to compress OR non-video file: copy as-is
        action = "[COPYING VIDEO (NO COMPRESSION)]" if src_file.lower().endswith(".mp4") else "[COPYING]"
        print(f"{action} {src_file}...")
        try:
            shutil.copy2(src_file, dest_file)
            return True
        except Exception as e:
            print(f"[ERROR] Copy failed for {src_file}: {e}")
            return False

def build_session_registry_for_mapping(base_dir):
    """
    Builds session registry mapping folder paths to start_ist, fps, and total frames.
    """
    registry = []
    if not os.path.exists(base_dir):
        return registry

    for root_path, dirs, files in os.walk(base_dir):
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

                    # Determine FPS and frames from LENS1
                    fps = 29.97
                    frames = 9000
                    l1_candidates = [os.path.join(root_path, "LENS1"), os.path.join(root_path, "lens1")]
                    vid_path = None
                    for l1 in l1_candidates:
                        if os.path.isdir(l1):
                            for vf in os.listdir(l1):
                                if vf.lower().endswith('.mp4'):
                                    vid_path = os.path.join(l1, vf)
                                    break
                        if vid_path:
                            break

                    if vid_path and os.path.exists(vid_path):
                        try:
                            import cv2
                            cap = cv2.VideoCapture(vid_path)
                            if cap.isOpened():
                                fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
                                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 9000
                                cap.release()
                        except Exception:
                            pass

                    registry.append({
                        "folder_path": os.path.normpath(root_path),
                        "folder_name": os.path.basename(root_path),
                        "start_ist": start_ist,
                        "end_ist": start_ist + timedelta(seconds=frames / fps),
                        "fps": fps,
                        "total_frames": frames
                    })
                except Exception as e:
                    print(f"[WARN] Error reading registry for {root_path}: {e}")
                break

    return registry

def generate_master_mapping(source_dir, output_csv_path):
    """
    Scans the directory for IMU/GPS CSVs and session metadata,
    generating a synchronized Master_Mapping.csv.
    """
    print(f"\n[INFO] Building Master Mapping from: {source_dir}...")
    registry = build_session_registry_for_mapping(source_dir)
    if not registry:
        print("[WARN] No camera sessions with metadata found for Master Mapping.")
        return []

    # Find candidate IMU / GPS CSV files
    imu_files = []
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith('.csv') and ('imu' in f.lower() or 'gps' in f.lower()):
                imu_files.append(os.path.join(root, f))

    if not imu_files:
        print("[WARN] No IMU/GPS CSV files found to generate Master Mapping.")
        return []

    total_mapped = 0
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    file_exists = os.path.exists(output_csv_path)
    existing_keys = set()
    
    if file_exists:
        try:
            with open(output_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if 'ist_time' in r and 'source_folder' in r:
                        existing_keys.add((r['ist_time'], r['source_folder']))
        except Exception:
            pass

    for imu_path in imu_files:
        try:
            def skip_comments(file_obj):
                for line in file_obj:
                    if line.strip() and not line.strip().startswith('#'):
                        yield line

            with open(imu_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(skip_comments(f))
                if not reader.fieldnames:
                    continue
                
                # Identify timestamp column: local IST, UTC, or Unix
                time_col = None
                time_mode = 'local' # 'local', 'utc', or 'unix'

                for c in reader.fieldnames:
                    clow = c.lower() if c else ""
                    if 'timestamp_local' in clow or 'local_timestamp' in clow or 'timestamp_ist' in clow:
                        time_col = c
                        time_mode = 'local'
                        break
                    elif 'timestamp_utc' in clow or 'utc_timestamp' in clow:
                        time_col = c
                        time_mode = 'utc'
                        break
                    elif 't_unix' in clow or 'unix_timestamp' in clow or 'timestamp_unix' in clow:
                        time_col = c
                        time_mode = 'unix'
                        break
                    elif clow == 'timestamp':
                        time_col = c
                        time_mode = 'local'
                        break

                if not time_col:
                    continue

                new_rows = []
                seen_in_file = set()

                for row in reader:
                    val = row.get(time_col, '').strip()
                    if not val:
                        continue

                    ist_dt = None
                    utc_dt = None

                    if time_mode == 'unix':
                        try:
                            unix_val = float(val)
                            utc_dt = datetime.fromtimestamp(unix_val, timezone.utc)
                            ist_dt = (utc_dt + timedelta(hours=5, minutes=30)).replace(tzinfo=None)
                        except ValueError:
                            continue
                    elif time_mode == 'utc':
                        try:
                            clean_utc = val.replace("T", " ").replace("Z", "").split('.')[0]
                            utc_dt = datetime.strptime(clean_utc, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                            ist_dt = (utc_dt + timedelta(hours=5, minutes=30)).replace(tzinfo=None)
                        except ValueError:
                            continue
                    else: # local / IST
                        try:
                            clean_ist = val.replace("T", " ").split('.')[0]
                            ist_dt = datetime.strptime(clean_ist, "%Y-%m-%d %H:%M:%S")
                            utc_dt = (ist_dt - timedelta(hours=5, minutes=30)).replace(tzinfo=timezone.utc)
                        except ValueError:
                            continue

                    clean_time_str = ist_dt.strftime("%Y-%m-%d %H:%M:%S")
                    if clean_time_str in seen_in_file:
                        continue

                    # Match session in registry
                    best_session = None
                    min_diff = float('inf')
                    for s in registry:
                        if s["start_ist"] <= ist_dt <= s["end_ist"]:
                            best_session = s
                            break
                        diff = abs((s["start_ist"] - ist_dt).total_seconds())
                        if diff < min_diff and diff <= 15 * 60:
                            min_diff = diff
                            best_session = s

                    if not best_session:
                        continue

                    elapsed_sec = max(0.0, (ist_dt - best_session["start_ist"]).total_seconds())
                    key = (clean_time_str, best_session["folder_name"])
                    if key in existing_keys:
                        continue

                    new_rows.append({
                        "ist_time": clean_time_str,
                        "utc_time": utc_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "unix_time": f"{utc_dt.timestamp():.3f}",
                        "source_folder": best_session["folder_name"],
                        "folder_path": best_session["folder_path"],
                        "playback_time_sec": f"{elapsed_sec:.3f}",
                        "frame_no": int(elapsed_sec * best_session["fps"])
                    })
                    existing_keys.add(key)
                    seen_in_file.add(clean_time_str)

                if new_rows:
                    with open(output_csv_path, 'a', newline='', encoding='utf-8') as mf:
                        writer = csv.DictWriter(mf, fieldnames=[
                            "ist_time", "utc_time", "unix_time", "source_folder", 
                            "folder_path", "playback_time_sec", "frame_no"
                        ])
                        if not file_exists:
                            writer.writeheader()
                            file_exists = True
                        writer.writerows(new_rows)
                    total_mapped += len(new_rows)

        except Exception as e:
            print(f"[WARN] Failed to process IMU CSV {imu_path}: {e}")

    print(f"[SUCCESS] Master Mapping complete! Appended {total_mapped} rows to: {output_csv_path}")
    return total_mapped

def compress_and_duplicate_folder(source_dir, dest_dir, do_compress=True, resolution="1080p", generate_master_map=True):
    """
    Mirrors source_dir into dest_dir. If do_compress is True,
    compresses MP4 files to the chosen resolution. Also optionally
    generates Master_Mapping.csv.
    """
    tasks = []

    for root, dirs, files in os.walk(source_dir):
        rel_path = os.path.relpath(root, source_dir)
        target_dir = os.path.join(dest_dir, rel_path)
        
        # Always ensure the target directory exists
        os.makedirs(target_dir, exist_ok=True)

        for file in files:
            # Skip macOS hidden metadata files
            if file.startswith("._"):
                continue
                
            src_file = os.path.join(root, file)
            dest_file = os.path.join(target_dir, file)
            
            # Check if the specific file already exists in the destination
            if os.path.exists(dest_file):
                print(f"[SKIPPING FILE] '{file}' already exists in destination.")
                continue  
                
            tasks.append((src_file, dest_file))
            
    print(f"\n[INFO] Queued {len(tasks)} files for processing (Compression: {do_compress}, Resolution: {resolution}).")
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_VIDEOS) as executor:
        for src, dest in tasks:
            executor.submit(process_single_file, src, dest, do_compress, resolution)

    if generate_master_map:
        master_csv_path = os.path.join(dest_dir, "Master_Mapping.csv")
        generate_master_mapping(source_dir, master_csv_path)

def run_compression_interactive(default_src=None, default_dst=None):
    """Interactive CLI runner for the compression tool with flexible input/destination paths."""
    print("\n" + "=" * 55)
    print(" 4K VIDEO COMPRESSION & MASTER MAPPING HUB ")
    print("=" * 55)

    base_src = default_src if default_src else r"D:\MUMMAS\MUMMAS DATA COLLECTION"
    base_dst = default_dst if default_dst else r"D:\MUMMAS\MUMMAS DATA COLLECTION-1080p"

    print(f"\n[CONFIGURED DIRECTORIES]")
    print(f"  Source Input Path      : {base_src}")
    print(f"  Destination Output Path: {base_dst}")
    print("\nPress ENTER to keep the current paths, or type a new path to change it.")

    src_in = input(f"Enter Source Folder [{base_src}]: ").strip()
    source_folder = src_in if src_in else base_src

    while not os.path.exists(source_folder):
        print(f"\n[ERROR] Source folder does not exist: {source_folder}")
        retry = input("Enter a valid Source Folder (or 'q' to cancel): ").strip()
        if retry.lower() == 'q':
            return
        source_folder = retry if retry else source_folder

    dst_in = input(f"Enter Destination Folder [{base_dst}]: ").strip()
    destination_folder = dst_in if dst_in else base_dst

    # 1. Option: Compress or Not
    compress_choice = input("\nDo you want to compress video files? [y/n] (Default: y): ").strip().lower()
    do_compress = compress_choice != 'n'

    resolution = "1080p"
    if do_compress:
        print("\nSelect Compression Target Resolution:")
        print(" 1) 1080p (1920x1080 - Recommended Full HD)")
        print(" 2) 720p  (1280x720  - HD)")
        print(" 3) 420p  (746x420   - Low Storage)")
        print(" 4) 480p  (854x480   - Standard Definition)")
        res_opt = input("Select an option (1-4) [default: 1]: ").strip()
        
        if res_opt == '2':
            resolution = "720p"
        elif res_opt == '3':
            resolution = "420p"
        elif res_opt == '4':
            resolution = "480p"
        else:
            resolution = "1080p"
        print(f"-> Selected Resolution: {resolution}")
    else:
        print("-> Videos will be copied in original 4K resolution without compression.")

    # 2. Master map option
    map_choice = input("\nGenerate Master Mapping CSV after processing? [y/n] (Default: y): ").strip().lower()
    gen_map = map_choice != 'n'

    print(f"\nStarting Execution:")
    print(f"  Source Input      : {source_folder}")
    print(f"  Destination Output: {destination_folder}")
    print(f"  Compression Mode  : {'Compress -> ' + resolution if do_compress else 'Full 4K Copy (No Compression)'}")
    print(f"  Master Mapping    : {'Yes (Master_Mapping.csv)' if gen_map else 'No'}\n")

    compress_and_duplicate_folder(
        source_folder, 
        destination_folder, 
        do_compress=do_compress, 
        resolution=resolution, 
        generate_master_map=gen_map
    )
    print("\n[SUCCESS] Batch process and mapping complete!")

if __name__ == "__main__":
    run_compression_interactive()