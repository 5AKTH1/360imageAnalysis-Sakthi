import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

MAX_CONCURRENT_VIDEOS = 2 

def process_single_file(src_file, dest_file):
    # Standard file processing logic
    if src_file.lower().endswith(".mp4"):
        print(f"[COMPRESSING] {src_file}...")
        cmd = [
            "ffmpeg", 
            "-hwaccel", "cuda", 
            "-hwaccel_output_format", "cuda", 
            "-i", src_file,
            "-vf", "scale_cuda=1920:1080", 
            "-c:v", "h264_nvenc",  
            "-cq", "28",          
            "-preset", "p4",       
            "-c:a", "aac", 
            "-y", 
            dest_file
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            print(f"[DONE] {dest_file}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to compress {src_file}. {e}")
    else:
        print(f"[COPYING] {src_file}...")
        shutil.copy2(src_file, dest_file)

def compress_and_duplicate_folder(source_dir, dest_dir):
    tasks = []

    for root, dirs, files in os.walk(source_dir):
        rel_path = os.path.relpath(root, source_dir)
        target_dir = os.path.join(dest_dir, rel_path)
        
        # Always ensure the target directory exists
        os.makedirs(target_dir, exist_ok=True)

        for file in files:
            # --- NEW: Skip macOS hidden metadata files ---
            if file.startswith("._"):
                continue
                
            src_file = os.path.join(root, file)
            dest_file = os.path.join(target_dir, file)
            
            # Check if the specific file already exists in the destination
            if os.path.exists(dest_file):
                print(f"[SKIPPING FILE] '{file}' already exists in destination.")
                continue  
                
            # If the file does not exist, add it to the tasks queue
            tasks.append((src_file, dest_file))
            
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_VIDEOS) as executor:
        for src, dest in tasks:
            executor.submit(process_single_file, src, dest)

if __name__ == "__main__":
    SOURCE_FOLDER = r"G:\MUMMAS DATA COLLECTION"
    DESTINATION_FOLDER = r"F:\MUMMAS DATA COLLECTION-1080p"
    
    compress_and_duplicate_folder(SOURCE_FOLDER, DESTINATION_FOLDER)
    print("Batch process complete!")