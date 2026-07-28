import os
import pandas as pd
from pathlib import Path

def find_header_index(filepath, target_column='timestamp_local'):
    """
    Peeks into the file to find the row number that contains the actual headers.
    This safely bypasses any random metadata at the top of sensor logs.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if target_column in line:
                    return i
    except Exception:
        pass
    return 0 # Fallback to standard reading if we can't find it

def get_timestamps_in_area(base_dir, corners, max_gap_seconds=5):
    """
    Scans through IMU-GPS directories, handles file headers,
    finds coordinates within the specified corners, and returns formatted 
    continuous timestamp ranges.
    """
    # 1. Define the bounding box from the 4 corners
    lats = [c[0] for c in corners]
    lons = [c[1] for c in corners]
    
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    
    print(f"Filtering for Latitudes: {min_lat} to {max_lat}")
    print(f"Filtering for Longitudes: {min_lon} to {max_lon}\n")
    
    base_path = Path(base_dir)
    results = {}
    valid_folders = {'imu', 'imu-gps', 'imugps', 'imu_gps'}

    for file_path in base_path.rglob('*'):
        is_in_target_folder = any(part.lower() in valid_folders for part in file_path.parts)
        
        # Make sure we only try to parse CSV files
        if file_path.is_file() and file_path.suffix.lower() == '.csv' and is_in_target_folder:
            try:
                # 2. Find the correct header row to skip metadata
                header_idx = find_header_index(file_path, 'timestamp_local')
                
                # Read the file starting right at the header row.
                # low_memory=False helps prevent warnings when dealing with 88+ mixed-data columns
                df = pd.read_csv(file_path, skiprows=header_idx, low_memory=False)
                
                req_cols = {'timestamp_local', 'gnss_latitude', 'gnss_longitude'}
                if not req_cols.issubset(df.columns):
                    continue
                
                # 3. Filter the dataframe based on the bounding box
                # First, ensure lat/lon are numeric (sometimes sensor logs write "NaN" or strings)
                df['gnss_latitude'] = pd.to_numeric(df['gnss_latitude'], errors='coerce')
                df['gnss_longitude'] = pd.to_numeric(df['gnss_longitude'], errors='coerce')
                
                inside_df = df[
                    (df['gnss_latitude'] >= min_lat) & (df['gnss_latitude'] <= max_lat) &
                    (df['gnss_longitude'] >= min_lon) & (df['gnss_longitude'] <= max_lon)
                ].copy()
                
                if inside_df.empty:
                    continue
                
                # 4. Process and group continuous timestamps
                inside_df['timestamp_local'] = pd.to_datetime(inside_df['timestamp_local'])
                inside_df = inside_df.sort_values('timestamp_local')
                
                time_diffs = inside_df['timestamp_local'].diff()
                group_ids = (time_diffs > pd.Timedelta(seconds=max_gap_seconds)).cumsum()
                
                ranges = inside_df.groupby(group_ids)['timestamp_local'].agg(['min', 'max'])
                
                file_ranges = []
                for _, row in ranges.iterrows():
                    start_time = row['min'].strftime('%H:%M:%S')
                    end_time = row['max'].strftime('%H:%M:%S')
                    
                    if start_time == end_time:
                        file_ranges.append(f"{start_time}") 
                    else:
                        file_ranges.append(f"{start_time}-{end_time}")
                
                results[str(file_path)] = file_ranges
                
            except pd.errors.EmptyDataError:
                pass # Skip silently if the CSV is totally empty
            except Exception as e:
                print(f"Error reading {file_path.name}: {e}")

    # 5. Print the Results
    print("--- Results ---")
    if not results:
        print("No matching data found in the specified area.")
    for file, time_ranges in results.items():
        print(f"\nFile: {file}")
        print(f"Continuous Timestamps: {', '.join(time_ranges)}")

if __name__ == "__main__":
    DIRECTORY_PATH = "F:\\MUMMAS DATA COLLECTION" # Change this to your actual directory path
    
    FOUR_CORNERS = [
        (13.010, 80.261), 
        (13.028, 80.249), 
        (12.992, 80.249), 
        (12.992, 80.261)  
    ]
    
    get_timestamps_in_area(base_dir=DIRECTORY_PATH, corners=FOUR_CORNERS, max_gap_seconds=1200)