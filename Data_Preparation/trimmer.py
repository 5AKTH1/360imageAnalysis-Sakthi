import os
import subprocess

# --- CONFIGURATION ---
INPUT_FILE = r"C:\IITM\Results\Final_Dynamic_Stitch.mp4"
OUTPUT_DIR = r"C:\IITM\Results"
OUTPUT_FILE_NAME = "trimmed_SIFT_Shifted.mp4"

# Trim Settings (Format: HH:MM:SS)
START_TIME = "00:04:15"
END_TIME = "00:04:25"

def trim_video():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Input Video: {INPUT_FILE}")
    print(f"Trimming ({START_TIME} - {END_TIME})...")
    print("-" * 50)

    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Could not find: {INPUT_FILE}")
        return

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME)
    
    # --- FFmpeg Command ---
    cmd = [
        "ffmpeg", 
        "-ss", START_TIME,   # Placed BEFORE input for instant seeking
        "-to", END_TIME,
        "-i", INPUT_FILE,
        "-c:v", "copy",      # Strictly copy only the Video stream
        "-c:a", "copy",      # Strictly copy only the Audio stream
        "-y",                # Overwrite output file if it exists
        output_path
    ]
        
    try:
        # Run FFmpeg and capture errors
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[SUCCESS] Saved trimmed video to: {output_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] FFmpeg failed to process the video.")
        print("--- FFmpeg Error Output ---")
        # Decode the error message so you can actually read what went wrong
        print(e.stderr.decode('utf-8', errors='ignore'))

    print("-" * 50)

if __name__ == "__main__":
    trim_video()