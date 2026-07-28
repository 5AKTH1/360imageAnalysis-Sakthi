import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class TelemetrySync:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, comment="#")
        self.df.columns = self.df.columns.str.strip()
        
        self.df["t_unix"] = pd.to_numeric(self.df["t_unix"], errors="coerce")
        
        if "timestamp_utc" in self.df.columns:
            self.df["timestamp_utc"] = pd.to_datetime(self.df["timestamp_utc"], errors="coerce")
            
        for col in ["filter_lla_lat", "filter_lla_lon", "yaw_deg", "speed_mps"]: 
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
                
        df_clean = self.df.dropna(subset=["t_unix", "filter_lla_lat", "filter_lla_lon"])
        df_clean = df_clean.sort_values("t_unix").reset_index(drop=True)
        
        self.interp_lat = interp1d(df_clean["t_unix"], df_clean["filter_lla_lat"], fill_value="extrapolate")
        self.interp_lon = interp1d(df_clean["t_unix"], df_clean["filter_lla_lon"], fill_value="extrapolate")
        
        if "yaw_deg" in df_clean.columns:
            yaw_rad = np.unwrap(np.deg2rad(df_clean["yaw_deg"].to_numpy()))
            self.interp_yaw = interp1d(df_clean["t_unix"], np.rad2deg(yaw_rad), fill_value="extrapolate")
        else:
            self.interp_yaw = None
            
        if "speed_mps" in df_clean.columns:
            self.interp_speed = interp1d(df_clean["t_unix"], df_clean["speed_mps"], fill_value="extrapolate")
        else:
            self.interp_speed = None

    def get_unix_from_utc(self, target_utc_str):
        if "timestamp_utc" not in self.df.columns:
            raise ValueError("Column 'timestamp_utc' not found in CSV.")
            
        target_dt = pd.to_datetime(target_utc_str)
        time_diffs = (self.df["timestamp_utc"] - target_dt).abs()
        closest_idx = time_diffs.idxmin()
        return float(self.df.loc[closest_idx, "t_unix"])

    def get_telemetry(self, t_unix):
        lat = float(self.interp_lat(t_unix))
        lon = float(self.interp_lon(t_unix))
        yaw = float(self.interp_yaw(t_unix)) % 360.0 if self.interp_yaw else 0.0
        speed = float(self.interp_speed(t_unix)) if self.interp_speed else 0.0
        return lat, lon, yaw, speed