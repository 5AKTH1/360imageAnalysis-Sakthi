import json
import cv2
import numpy as np
import math

from sync import TelemetrySync
import lidar_pcap
from perceive import CalibratedProjector, get_bbox_depth, fast_project_gps, fast_distance_m

IMU_CSV = r"D:\MUMMAS DATA COLLECTION\17012026\TRIP2\IMU-GPS\imu_all_topics_2026-01-17_15-21-18.csv"
L1_PCAP = r"D:\MUMMAS DATA COLLECTION\17012026\TRIP2\LIDAR\LiDAR1\lidar1_raw_20260117_150950.pcap"
L2_PCAP = r"D:\MUMMAS DATA COLLECTION\17012026\TRIP2\LIDAR\LiDAR2\lidar2_raw_20260117_150950.pcap"
VIDEO_PATH = r"C:\IITM\Results\lens 1\1080p_lens1.mp4"
META_JSON = r"D:\MUMMAS DATA COLLECTION\17012026\TRIP2\CAMERA\run_20260117_152118\session_meta.json"
DETECTIONS_JSON = r"C:\IITM\Results\lens 1\final_lens1_detected.json"

# --- Output video path ---
OUTPUT_VIDEO_PATH = r"C:\IITM\Lidar_Camera_Projection\output2_tracked_lens1.mp4"

VIDEO_OFFSET_MINUTES = 20.0 
VIDEO_OFFSET_SECONDS = VIDEO_OFFSET_MINUTES * 60.0

# Scaled intrinsic parameters (from 2160x3840 down to 1080x1920)
FX, FY = 553.23, 538.88
CX, CY = 540.0, 960.0    

T_LIDAR_TO_CAM = np.array([
    [0.20927,   -0.977663, -0.0195126, 2.05416],
    [0.0386136,  0.0282009, -0.998856,  0.401883],
    [0.977095,   0.208277,  0.0436527, -0.035744],
    [0.0,        0.0,       0.0,       1.0]
])

def main():
    with open(META_JSON, "r") as f: meta_data = json.load(f)
    original_video_utc = meta_data["created_utc"]

    with open(DETECTIONS_JSON, "r") as f: detection_data = json.load(f)
    frame_detections = {int(d["frame"]): d["objects"] for d in detection_data["detections"]}

    telemetry = TelemetrySync(IMU_CSV)
    t_start_unix = telemetry.get_unix_from_utc(original_video_utc)
    
    # Grab video stats but DO NOT keep the video open yet.
    cap = cv2.VideoCapture(VIDEO_PATH)
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    video_start_unix = t_start_unix + VIDEO_OFFSET_SECONDS
    video_end_unix = video_start_unix + (total_frames / fps)
    
    t_min_lidar = video_start_unix - 1.0
    t_max_lidar = video_end_unix + 1.0
    
    print("[INFO] Loading Point Clouds...")
    pts1, pts2 = lidar_pcap.load_both(L1_PCAP, L2_PCAP, t_min=t_min_lidar, t_max=t_max_lidar)
    projector = CalibratedProjector(FX, FY, CX, CY, T_LIDAR_TO_CAM)

    # ---------------------------------------------------------
    # PASS 1: KINEMATIC DATA COLLECTION (No Video Playback)
    # ---------------------------------------------------------
    print(f"[INFO] Pass 1: Calculating 3D GPS Projections for {total_frames} frames...")
    id_kinematics = {}
    frame_cache = {}

    for frame_idx in range(total_frames):
        current_t_unix = t_start_unix + VIDEO_OFFSET_SECONDS + (frame_idx / fps)
        ego_lat, ego_lon, ego_yaw, ego_speed = telemetry.get_telemetry(current_t_unix)
        rel_current_t = current_t_unix - t_min_lidar

        if len(pts1) > 0:
            i1_start = np.searchsorted(pts1[:, 0], rel_current_t - 0.05)
            i1_end = np.searchsorted(pts1[:, 0], rel_current_t + 0.05)
            frame_pts1 = pts1[i1_start:i1_end, 1:4]
        else: frame_pts1 = np.empty((0, 3), dtype=np.float32)

        if len(pts2) > 0:
            i2_start = np.searchsorted(pts2[:, 0], rel_current_t - 0.05)
            i2_end = np.searchsorted(pts2[:, 0], rel_current_t + 0.05)
            frame_pts2 = pts2[i2_start:i2_end, 1:4]
        else: frame_pts2 = np.empty((0, 3), dtype=np.float32)

        frame_pts_xyz = np.vstack([frame_pts1, frame_pts2])
        if len(frame_pts_xyz) > 0:
            u, v, depths = projector.project(frame_pts_xyz, img_h, img_w)
        else:
            u, v, depths = np.array([]), np.array([]), np.array([])

        frame_cache[frame_idx] = {}
        objects_in_frame = frame_detections.get(frame_idx, [])
        
        for obj in objects_in_frame:
            if obj["label"] not in ["car", "truck", "bus", "motorbike", "auto"]: continue
            box = obj["box"]
            obj_id = obj["id"]
            
            depth = get_bbox_depth(u, v, depths, box) if len(u) > 0 else None
            frame_cache[frame_idx][obj_id] = {
                "box": box, "label": obj["label"], "depth": depth, "ego_speed": ego_speed
            }

            if depth is not None:
                # Calculate true Radial GPS (preventing the edge-of-frame pulling illusion)
                raw_cx = (box[0] + box[2]) / 2.0
                az_deg = math.degrees(math.atan2((raw_cx - CX), FX))
                
                # Convert Z-Depth to Hypotenuse Ground Distance
                ground_distance = depth / math.cos(math.radians(az_deg))
                t_br = (ego_yaw + az_deg) % 360.0
                
                t_lat, t_lon = fast_project_gps(ego_lat, ego_lon, ground_distance, t_br)
                
                if obj_id not in id_kinematics: 
                    id_kinematics[obj_id] = {"lats": [], "lons": []}
                id_kinematics[obj_id]["lats"].append(t_lat)
                id_kinematics[obj_id]["lons"].append(t_lon)

    # ---------------------------------------------------------
    # PASS 2: PERMANENT CLASSIFICATION 
    # ---------------------------------------------------------
    print("[INFO] Pass 2: Assigning permanent PARKED/MOVING labels...")
    id_final_states = {}
    
    for obj_id, data in id_kinematics.items():
        lats, lons = data["lats"], data["lons"]
        
        if len(lats) < 5:
            # If the car was only visible for < 5 frames, assume MOVING
            id_final_states[obj_id] = "MOVING"
            continue
            
        # Apply a heavy 5-frame moving average to obliterate single-frame LiDAR glitches
        smoothed_lats = np.convolve(lats, np.ones(5)/5, mode='valid')
        smoothed_lons = np.convolve(lons, np.ones(5)/5, mode='valid')
        
        start_lat, start_lon = smoothed_lats[0], smoothed_lons[0]
        max_travel = 0.0
        
        # Look into the future. What is the maximum distance this ID ever reached?
        for lat, lon in zip(smoothed_lats, smoothed_lons):
            dist = fast_distance_m(start_lat, start_lon, lat, lon)
            if dist > max_travel:
                max_travel = dist
                
        # If the car translated more than 15 meters during its entire life, it's MOVING. 
        # Otherwise, it's permanently PARKED. Zero flickering.
        if max_travel > 15.0:
            id_final_states[obj_id] = "MOVING"
        else:
            id_final_states[obj_id] = "PARKED"

    # ---------------------------------------------------------
    # PASS 3: RENDER THE VIDEO
    # ---------------------------------------------------------
    print("[INFO] Pass 3: Rendering the final tracked video...")
    cap = cv2.VideoCapture(VIDEO_PATH) # Re-open the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (img_w, img_h))
    
    final_analytics = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        current_t_unix = t_start_unix + VIDEO_OFFSET_SECONDS + (frame_idx / fps)
        objects_this_frame = frame_cache.get(frame_idx, {})
        
        for obj_id, data in objects_this_frame.items():
            box = data["box"]
            depth = data["depth"]
            
            # Fetch the permanent state we calculated in Pass 2
            final_status = id_final_states.get(obj_id, "MOVING")
            
            final_analytics.append({
                "frame": frame_idx, "timestamp_unix": round(current_t_unix, 3), 
                "vehicle_id": int(obj_id), "vehicle_type": data["label"], 
                "depth_meters": round(depth, 2) if depth is not None else None,
                "ego_speed_mps": round(data["ego_speed"], 2), "status": final_status
            })

            x1, y1, x2, y2 = map(int, box)
            depth_str = f"{depth:.1f}m" if depth is not None else "N/A"
            display_label = f"ID {obj_id}: {final_status} ({depth_str})"
            
            color = (0, 0, 255) if final_status == "PARKED" else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, display_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out_video.write(frame)
        cv2.imshow("Tracking Pipeline", cv2.resize(frame, (540, 960)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
            
        frame_idx += 1

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    
    with open(r"C:\IITM\Lidar_Camera_Projection\parked_analytics_output.json", "w") as f:
        json.dump(final_analytics, f, indent=4)
        
    print(f"[SUCCESS] Pipeline Complete. Video saved to: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__": 
    main()