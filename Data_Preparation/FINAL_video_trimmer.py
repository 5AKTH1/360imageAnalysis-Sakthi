import os
import subprocess

# --- CONFIGURATION ---
# Point this to the main "Camera" folder that holds the "lens1", "lens2" folders
BASE_DIR = r"D:\MUMMAS DATA COLLECTION\24032026\CAMERA\run_20260324_104420_2890"   # Folder with your raw 1-hour files
OUTPUT_DIR = r"C:\IITM\CAMERA_Short_Clips"  # Where the trimmed files will be saved

# Trim Settings (Format: HH:MM:SS)
START_TIME = "00:00:00"
END_TIME = "00:02:00"

# Unique Name for this batch
SESSION_NAME = "SESSION002"

def process_nested_structure():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Scanning structure in: {BASE_DIR}")
    print(f"Trimming ({START_TIME} - {END_TIME}) and Renaming...")
    print("-" * 50)

    processed_count = 0

    # Iterate strictly through lenses 1 to 6
    for i in range(1, 7):
        # Insta360 expects 0-indexed files (0-5), but your folders are 1-indexed (1-6)
        insta_index = i - 1
        
        # Define the specific path structure you described:
        # Camera/lens1/video_lens1.mp4
        lens_folder = f"LENS{i}"
        filename = f"video_lens{i}.mp4"
        input_path = os.path.join(BASE_DIR, lens_folder, filename)
        
        # Verify file exists before trying to process
        if not os.path.exists(input_path):
            print(f"[MISSING] Could not find: {input_path}")
            continue

        # Construct the official Insta360 Output Name
        # Format: origin_{lens_index}_{unique_session_id}.mp4
        final_name = f"origin_{insta_index}_{SESSION_NAME}.mp4"
        output_path = os.path.join(OUTPUT_DIR, final_name)
        
        # --- FFmpeg Command ---
        cmd = [
            "ffmpeg", 
            "-i", input_path,
            "-ss", START_TIME,
            "-to", END_TIME,
            "-c", "copy",        # Stream Copy (No Re-encoding)
            "-map", "0",         # Copy Video, Audio, Gyro, etc.
            "-copy_unknown",     # Force copy of proprietary data
            "-y",                # Overwrite
            output_path
        ]
        
        print(f"Processing Lens {i}: {filename} -> {final_name}")
        
        try:
            # Run FFmpeg silently
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            processed_count += 1
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to process {filename}")

    print("-" * 50)
    
    if processed_count == 6:
        print("\nSUCCESS: All 6 lenses processed!")
    else:
        print(f"\nWARNING: Only {processed_count}/6 lenses found.")
        print("Please check your folder structure.")

if __name__ == "__main__":
    process_nested_structure()