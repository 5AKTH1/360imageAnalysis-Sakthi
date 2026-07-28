import os
import json
import pandas as pd
from datetime import datetime

class AssetDiscovery:
    _metadata_cache = {}

    @classmethod
    def load_metadata_from_drive(cls, root_drive):
        """Indexes metadata.json files by the 'date' field (e.g., 22012026)."""
        print("Indexing metadata.json files...")
        for root, _, files in os.walk(root_drive):
            if "metadata.json" in files:
                try:
                    with open(os.path.join(root, "metadata.json"), 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                        date_key = meta.get('date', 'N/A')
                        if date_key and str(date_key).upper() != 'N/A':
                            cls._metadata_cache[str(date_key)] = meta
                except Exception as e:
                    print(f"Error loading metadata in {root}: {e}")
        print(f"Indexed {len(cls._metadata_cache)} daily metadata files.")

    @classmethod
    def search_for_timestamp(cls, target_dt):
        """Checks availability by matching the DDMMYYYY date and checking run windows."""
        date_key = target_dt.strftime('%d%m%Y')
        daily_meta = cls._metadata_cache.get(date_key)
        
        if not daily_meta:
            return "no", "No session_meta.json files"
            
        target_naive = target_dt.replace(tzinfo=None)
        
        for trip_key, runs in daily_meta.get('trips', {}).items():
            for run in runs:
                try:
                    # --- Time Check ---
                    start_str = str(run.get('start_time_ist', 'N/A')).strip()
                    end_str = str(run.get('end_time_ist', 'N/A')).strip()

                    if start_str.upper() in ["N/A", "NONE", ""] or end_str.upper() in ["N/A", "NONE", ""]:
                        continue 
                        
                    start_ist = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')
                    end_ist = datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S')
                    
                    if start_ist <= target_naive <= end_ist:
                        
                        # --- IMU Check --- 
                        imu_val = run.get('imu-gps', 'N/A')
                        if not imu_val or str(imu_val).strip().upper() in ["N/A", "NONE", ""]:
                            return "no", "No IMU-CSV"

                        # --- Camera Data Check ---
                        camera_data = run.get('camera', 'N/A')
                        
                        # If the entire camera key is literally "N/A" or missing entirely
                        if not camera_data or (isinstance(camera_data, str) and camera_data.strip().upper() in ["N/A", "NONE", ""]):
                            return "no", "No Camera Data"
                        
                        # If camera_data is a valid dictionary, check the lenses
                        missing_lenses = []
                        if isinstance(camera_data, dict):
                            for i in range(1, 7):
                                lens_key = f"lens{i}_frame_count"
                                lens_val = camera_data.get(lens_key, 'N/A')
                                
                                if str(lens_val).strip().upper() in ["N/A", "NONE", "NULL", ""] or lens_val == 0:
                                    missing_lenses.append(str(i))
                            
                            if len(missing_lenses) == 6:
                                return "no", "No Camera Data"
                            elif len(missing_lenses) > 0:
                                lens_str = " and ".join([f"Lens{x}" for x in missing_lenses])
                                return "no", f"{lens_str} data is missing"
                        else:
                            # Fallback if camera data exists but isn't formatted as expected
                            return "no", "No Camera Data"
                                
                        # SUCCESS
                        return "yes", "OK"
                        
                except Exception as e:
                    print(f"DEBUG - Error parsing run data for timestamp {target_dt}: {e}")
                    continue
                                
        # Fallback: Timestamp outside all run windows
        return "no", "No session_meta.json files"

def audit_dataset():
    csv_path = input("Enter path to input CSV: ").strip()
    hard_drive_root = input("Enter Hard Drive Root: ").strip()
    
    AssetDiscovery.load_metadata_from_drive(hard_drive_root)
    
    df = pd.read_csv(csv_path)
    
    ts_col = next((c for c in df.columns if 'timestamp' in str(c).lower()), None)
    if not ts_col:
        print(f"Error: Could not find 'timestamp' column. Available: {list(df.columns)}")
        return

    df[ts_col] = pd.to_datetime(df[ts_col])

    if 'available' not in df.columns:
        df['available'] = ""
    else:
        df['available'] = df['available'].astype(str) 

    if 'failure_reason' not in df.columns:
        df['failure_reason'] = ""
    else:
        df['failure_reason'] = df['failure_reason'].astype(str)

    print(f"\nAuditing {len(df)} rows...")
    for idx, row in df.iterrows():
        available, reason = AssetDiscovery.search_for_timestamp(row[ts_col])
        df.at[idx, 'available'] = available
        df.at[idx, 'failure_reason'] = reason if available == "no" else ""

    out_name = csv_path.rsplit('.', 1)[0] + '_audited.csv'
    df.to_csv(out_name, index=False)
    print(f"\nAudit complete! Saved to {out_name}")

if __name__ == "__main__":
    audit_dataset()