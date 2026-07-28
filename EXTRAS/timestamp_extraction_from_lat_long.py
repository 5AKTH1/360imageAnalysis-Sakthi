import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path

def read_imu_file(file_path):
    try:
        header_idx = 0
        # Find the header row
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for idx, line in enumerate(f):
                if idx > 50:
                    break
                line_lower = line.lower()
                if 'gnss_latitude' in line_lower and 'gnss_longitude' in line_lower:
                    header_idx = idx
                    break
                    
        # Removed the invalid 'errors' argument from pd.read_csv
        df = pd.read_csv(file_path, header=header_idx, on_bad_lines='skip', encoding='utf-8')
        df.columns = df.columns.astype(str).str.strip().str.lower()
        df['gnss_latitude'] = pd.to_numeric(df['gnss_latitude'], errors='coerce')
        df['gnss_longitude'] = pd.to_numeric(df['gnss_longitude'], errors='coerce')
        
        return df.dropna(subset=['gnss_latitude', 'gnss_longitude'])
    except Exception as e:
        print(f"  [ERROR reading {file_path.name}]: {e}")
        return pd.DataFrame()

def sync_timestamps_from_mummas(merged_csv_path, mummas_base_path, output_path, target_format, max_distance=0.002):
    df_points = pd.read_csv(merged_csv_path)
    df_points['X'] = pd.to_numeric(df_points['X'], errors='coerce')
    df_points['Y'] = pd.to_numeric(df_points['Y'], errors='coerce')
    df_points['date_key'] = pd.to_datetime(df_points['timestamp'], format='mixed').dt.date
    
    final_groups = []
    unmatched_rows = []
    base_dir = Path(mummas_base_path)
    
    print("--- STARTING DIAGNOSTICS ---")
    
    for date_val, group in df_points.groupby('date_key'):
        d_ddmmyyyy = date_val.strftime('%d%m%Y')
        date_folder = base_dir / d_ddmmyyyy
        
        print(f"\nProcessing Date: {d_ddmmyyyy} ({len(group)} QGIS points)")
        
        if not date_folder.exists():
            print(f"  [ERROR] Date folder not found: {date_folder}")
            unmatched_rows.append(group)
            final_groups.append(group)
            continue
            
        print(f"  [INFO] Found date folder: {date_folder.name}")
        
        all_files_in_date = list(date_folder.rglob('*'))
        
        # UPDATED: Now checks if 'imu' is ANYWHERE in the file's path, catching deeply nested files
        sensor_files = [
            f for f in all_files_in_date 
            if f.is_file() 
            and 'imu' in [p.lower() for p in f.parts] 
            and not f.name.startswith('~')
        ]
        
        if not sensor_files:
            print(f"  [ERROR] Found folder, but NO files inside any 'IMU' subfolders.")
            unmatched_rows.append(group)
            final_groups.append(group)
            continue
            
        csv_files = [f for f in sensor_files if f.suffix.lower() == '.csv']
        if not csv_files:
            print(f"  [ERROR] None of the IMU files are CSV files.")
            unmatched_rows.append(group)
            final_groups.append(group)
            continue
            
        dfs = [read_imu_file(f) for f in csv_files]
        dfs = [d for d in dfs if not d.empty]
        
        if not dfs:
            print("  [ERROR] Failed to extract any valid GPS data from the CSV files.")
            unmatched_rows.append(group)
            final_groups.append(group)
            continue
            
        df_gps = pd.concat(dfs, ignore_index=True)
        print(f"  [INFO] Successfully extracted {len(df_gps)} IMU GPS coordinates")
        
        tree = cKDTree(df_gps[['gnss_latitude', 'gnss_longitude']].values)
        
        group = group.copy()
        distances, indices = tree.query(group[['X', 'Y']].values, distance_upper_bound=max_distance)
        
        valid_mask = distances != np.inf
        valid_indices = indices[valid_mask]
        
        matched_count = valid_mask.sum()
        print(f"  [RESULT] Matched {matched_count} out of {len(group)} points")
        
        group.loc[valid_mask, 'timestamp'] = df_gps.iloc[valid_indices]['timestamp_local'].values
        
        if not valid_mask.all():
            unmatched_rows.append(group[~valid_mask])
            
        final_groups.append(group)
        
    df_final = pd.concat(final_groups, ignore_index=True)
    df_final['timestamp'] = pd.to_datetime(df_final['timestamp'], format='mixed').dt.strftime(target_format)
    df_final.drop(columns=['date_key'], inplace=True)
    df_final.to_csv(output_path, index=False)
    
    print("\n--- FINISHED ---")
    if unmatched_rows:
        total_unmatched = sum(len(g) for g in unmatched_rows)
        print(f"Warning: {total_unmatched} rows could not be matched.")

sync_timestamps_from_mummas(
    r'C:\viswak_MUMMAS_360degcamera\Insta360ImageAnalysis\qgis Lat Long\attributes_8th_to_30thJune.csv',
    r'F:\MUMMAS DATA COLLECTION\MUMMAS AQI BUILDING', 
    r'C:\viswak_MUMMAS_360degcamera\Insta360ImageAnalysis\qgis Lat Long\attributes_time_synced.csv',
    '%Y-%m-%d %H:%M:%S'
)