import os

def format_size(size_in_bytes):
    """Converts bytes to a human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0

def get_directory_size(start_path):
    """Calculates the total size of a directory and all its contents."""
    total_size = 0
    for dirpath, _, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # Skip if it is a symbolic link
            if not os.path.islink(fp):
                try:
                    total_size += os.path.getsize(fp)
                except (OSError, FileNotFoundError):
                    pass # Ignore files that are deleted or lack permissions during scan
    return total_size

def analyze_folders(base_directory):
    matched_folders = {}
    total_bytes = 0

    print(f"Scanning directory: {base_directory}\nThis might take a moment depending on the size of the drive...\n")

    for root, dirs, files in os.walk(base_directory):
        # We iterate over a copy of 'dirs' so we can safely modify the original list
        for folder_name in list(dirs):
            folder_lower = folder_name.lower()
            
            # Get all parent folders in the current path to check for "calibration"
            path_parts = [p.lower() for p in root.split(os.sep)]
            
            # --- 1. Identify Inclusion Criteria ---
            is_camera = "camera" in folder_lower
            is_imu = "imu" in folder_lower
            is_run = folder_lower.startswith("run_")
            
            # --- 2. Identify Exclusion Criteria ---
            # If it's a camera folder, check if "calibration" is anywhere in the parent path
            under_calibration = any("calibration" in part for part in path_parts)
            if is_camera and under_calibration:
                continue # Skip this folder entirely
            
            # --- 3. Process Matched Folders ---
            if is_camera or is_imu or is_run:
                full_path = os.path.join(root, folder_name)
                
                # Calculate size
                size = get_directory_size(full_path)
                matched_folders[full_path] = size
                total_bytes += size
                
                print(f"Found: {full_path} -> {format_size(size)}")
                
                # IMPORTANT: Remove this folder from 'dirs' so os.walk doesn't 
                # go inside it and double-count sub-folders that also match the criteria!
                dirs.remove(folder_name)

    return matched_folders, total_bytes

if __name__ == "__main__":
    # Prompt the user for the target directory
    target_dir = input("Enter the path to scan (e.g., D:\\MUMMAS DATA COLLECTION): ").strip()
    
    if os.path.exists(target_dir):
        print("-" * 60)
        folders, grand_total = analyze_folders(target_dir)
        print("-" * 60)
        
        print(f"\nTotal Matched Folders: {len(folders)}")
        print(f"GRAND TOTAL DATA SIZE: {format_size(grand_total)}\n")
    else:
        print("Error: Directory does not exist. Please check the path and try again.")