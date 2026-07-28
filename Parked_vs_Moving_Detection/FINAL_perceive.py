import numpy as np
import math

class CalibratedProjector:
    def __init__(self, fx, fy, cx, cy, t_lidar_to_cam):
        self._K = np.array([
            [fx,  0.0, cx],
            [0.0, fy,  cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        _T = np.array(t_lidar_to_cam, dtype=np.float32)
        self._R = _T[:3, :3]
        self._t = _T[:3, 3]

    def project(self, pts_xyz, img_h, img_w):
        N = len(pts_xyz)
        if N == 0: 
            return np.array([]), np.array([]), np.array([])
            
        # Downsample extremely large point clouds for speed
        if N > 500_000: 
            step = N // 150_000
            pts_xyz = pts_xyz[::step]

        pts_xyz = pts_xyz.astype(np.float32)
        cam = np.dot(pts_xyz, self._R.T) + self._t
        
        x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]
        
        # Filter points that are behind the camera or too far away
        valid = (z > 1.5) & (z < 80.0) & (np.abs(x) < 50.0) & (y > -1.5) & (y < 12.0)
        x, y, z = x[valid], y[valid], z[valid]
        
        # Project 3D coordinates onto the 2D image plane
        u = self._K[0, 0] * x / z + self._K[0, 2]
        v = self._K[1, 1] * y / z + self._K[1, 2]
        
        in_frame = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
        return u[in_frame], v[in_frame], z[in_frame]

def get_bbox_depth(u, v, depths, bbox):
    """
    Finds vehicle depth using Histogram Peak Detection to lock onto the solid chassis
    and ignore smeared road noise and background objects.
    """
    x1, y1, x2, y2 = bbox
    
    # Mild core crop (middle 80%) to avoid the extreme edges of the bounding box
    w, h = x2 - x1, y2 - y1
    cx1, cx2 = x1 + (w * 0.1), x2 - (w * 0.1) 
    cy1, cy2 = y1 + (h * 0.1), y2 - (h * 0.1) 
    
    in_box = (u >= cx1) & (u <= cx2) & (v >= cy1) & (v <= cy2) & (depths > 2.0) & (depths < 80.0)
    valid_depths = depths[in_box]
    
    if len(valid_depths) < 3:
        return None
        
    # 1. DENSITY PEAK DETECTION (Group all LiDAR hits into 0.5-meter bins)
    bins = np.arange(2.0, 80.0, 0.5)
    hist, bin_edges = np.histogram(valid_depths, bins=bins)
    
    # Find the bin with the absolute highest number of LiDAR hits
    peak_bin_index = np.argmax(hist)
    
    # 2. ISOLATE THE CAR
    peak_min = bin_edges[peak_bin_index]
    peak_max = bin_edges[peak_bin_index + 1]
    
    # Extract only the points that hit the car's solid surface
    car_points = valid_depths[(valid_depths >= peak_min) & (valid_depths <= peak_max)]
    
    if len(car_points) > 0:
        # Take the 20th percentile to ensure we measure the front bumper, not the interior
        return float(np.percentile(car_points, 20))
    
    return None

def fast_project_gps(lat, lon, distance_m, bearing_deg):
    """
    Projects a new GPS coordinate given a starting coordinate, distance, and heading.
    """
    R = 6378137.0 # Earth's radius in meters
    br = math.radians(bearing_deg)
    lr = math.radians(lat)
    
    l2 = math.asin(math.sin(lr) * math.cos(distance_m/R) + math.cos(lr) * math.sin(distance_m/R) * math.cos(br))
    lo2 = math.radians(lon) + math.atan2(math.sin(br) * math.sin(distance_m/R) * math.cos(lr), math.cos(distance_m/R) - math.sin(lr) * math.sin(l2))
    
    return math.degrees(l2), math.degrees(lo2)

def fast_distance_m(lat1, lon1, lat2, lon2):
    """
    Calculates the Haversine distance in meters between two GPS coordinates.
    """
    R = 6378137.0 # Earth's radius in meters
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    
    a = math.sin(dp/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def is_vehicle_parked(tracker_history, vehicle_id, current_depth, current_t_unix, ego_lat, ego_lon, ego_yaw, bbox, fx=553.23, cx=540.0):
    if current_depth is None:
        return tracker_history.get(vehicle_id, {}).get("state", False)

    if vehicle_id not in tracker_history:
        tracker_history[vehicle_id] = {
            "lats": [], "lons": [], "state": False, "ever_moved": False,
            "f_lat": None, "f_lon": None,
            "first_ego_lat": ego_lat, 
            "first_ego_lon": ego_lon
        }
    hist = tracker_history[vehicle_id]
    
    if hist["ever_moved"]: 
        return False
        
    # --- EGO-TRAVEL PERSISTENCE ---
    ego_travel = fast_distance_m(hist["first_ego_lat"], hist["first_ego_lon"], ego_lat, ego_lon)
    if ego_travel > 75.0:
        hist["ever_moved"], hist["state"] = True, False
        return False
    
    # --- CALCULATE TRUE RADIAL GPS (The Fix) ---
    raw_cx = (bbox[0] + bbox[2]) / 2.0
    az_deg = math.degrees(math.atan2((raw_cx - cx), fx))
    az_rad = math.radians(az_deg)
    
    # Convert Z-Depth (straight ahead) to Hypotenuse Ground Distance (true distance)
    # This stops the car from "pulling inward" as it reaches the edge of the video!
    ground_distance = current_depth / math.cos(az_rad)
    
    t_br = (ego_yaw + az_deg) % 360.0
    raw_t_lat, raw_t_lon = fast_project_gps(ego_lat, ego_lon, ground_distance, t_br)
    
    # --- ABSOLUTE SMOOTHING ---
    if not hist["lats"]:
        smooth_lat, smooth_lon = raw_t_lat, raw_t_lon
    else:
        smooth_lat = (hist["lats"][-1] * 0.7) + (raw_t_lat * 0.3)
        smooth_lon = (hist["lons"][-1] * 0.7) + (raw_t_lon * 0.3)

    # --- THE LIFETIME TRAVEL CHECK (Expanded to 20m) ---
    if hist["f_lat"] is None:
        hist["f_lat"], hist["f_lon"] = smooth_lat, smooth_lon
        
    # Expanded slightly from 15m to 20m to safely absorb time-sync lag at high speeds
    if fast_distance_m(hist["f_lat"], hist["f_lon"], smooth_lat, smooth_lon) > 20.0:
        hist["ever_moved"], hist["state"] = True, False
        return False
        
    hist["lats"].append(smooth_lat)
    hist["lons"].append(smooth_lon)
    
    if len(hist["lats"]) > 30:
        hist["lats"].pop(0)
        hist["lons"].pop(0)
        
    if len(hist["lats"]) < 10: 
        return hist["state"]
    
    # --- RECENT MOVEMENT CHECK ---
    travel = fast_distance_m(
        np.mean(hist["lats"][:5]), np.mean(hist["lons"][:5]), 
        np.mean(hist["lats"][-5:]), np.mean(hist["lons"][-5:])
    )
    
    # Base 3.0m + 3% of true ground distance
    dynamic_threshold = 3.0 + (ground_distance * 0.03)
    
    hist["state"] = travel < dynamic_threshold
    return hist["state"]