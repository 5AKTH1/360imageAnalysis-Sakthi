import os
import cv2
import numpy as np
import json
import time
import math
from ultralytics import YOLO
from collections import Counter, defaultdict

# Imports from your existing sensor modules
from sync import TelemetrySync
import lidar_pcap
from perceive import CalibratedProjector, get_bbox_depth, fast_project_gps, fast_distance_m

class HybridLiDARFlowDetector:
    def __init__(self, model_path):
        print(f"[INFO] Loading Hybrid LiDAR-Flow Tracker...")
        self.model = YOLO(model_path)
        self.colors = {'auto': (0, 165, 255), 'bus': (255, 0, 0), 'car': (0, 255, 0), 'motorbike': (255, 255, 0), 'truck': (0, 0, 255)}
        self.excluded_classes = ['tractor', 'rickshaw', 'e-rickshaw', 'cart', 'person', 'cycle']
        self.CONF_FAR = 0.10        
        self.CONF_CLOSE = 0.30
        
        # --- OPTICAL FLOW PARAMETERS (For Box Propagation) ---
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners=20, qualityLevel=0.1, minDistance=7, blockSize=7)
        
        # --- SCALED INTRINSICS (1080x1920) ---
        self.FX, self.FY = 553.23, 538.88
        self.CX, self.CY = 540.0, 960.0    
        self.T_LIDAR_TO_CAM = np.array([
            [0.20927,   -0.977663, -0.0195126, 2.05416],
            [0.0386136,  0.0282009, -0.998856,  0.401883],
            [0.977095,   0.208277,  0.0436527, -0.035744],
            [0.0,        0.0,       0.0,       1.0]
        ])

    def setup_resolution(self, w, h):
        scale_x, scale_y = w / 3328, h / 1664
        self.SKY_Y_LIMIT = int(800 * scale_y)
        self.VEHICLE_Y_LIMIT = int(1150 * scale_y)
        self.VIP_Y_LIMIT = int(950 * scale_y)
        poly = [[0, 1021], [78, 1088], [341, 1165], [419, 1094], [612, 1121], [679, 1206], [875, 1223], [921, 1290], [1083, 1229], [1616, 1215], [1930, 1306], [2064, 1113], [2206, 1065], [2340, 1111], [2807, 1021], [3267, 1009], [3287, 1013], [3328, 1664], [0, 1664]]
        self.EGO_POLYGON = np.array([[int(x * scale_x), int(y * scale_y)] for x, y in poly], np.int32)

    def get_dynamic_threshold(self, bbox_bottom_y, label):
        if label in ['bus', 'truck']: return 0.15 if bbox_bottom_y > self.VIP_Y_LIMIT else 0.40  
        y_clamped = max(self.SKY_Y_LIMIT, min(bbox_bottom_y, self.VEHICLE_Y_LIMIT))
        ratio = 1.0 if self.VEHICLE_Y_LIMIT == self.SKY_Y_LIMIT else ((y_clamped - self.SKY_Y_LIMIT) / (self.VEHICLE_Y_LIMIT - self.SKY_Y_LIMIT)) ** 2 
        base_thresh = self.CONF_FAR + (self.CONF_CLOSE - self.CONF_FAR) * ratio
        if label == 'auto': return max(base_thresh, 0.60) 
        if label == 'motorbike': return max(base_thresh, 0.20)
        return max(base_thresh, 0.27)

    def propagate_missing_detections_with_flow(self, prev_gray, curr_gray, prev_dets, curr_dets, frame_w, frame_h):
        curr_ids = {det['id'] for det in curr_dets}
        propagated_dets = []
        for det in prev_dets:
            if det['id'] not in curr_ids:
                x1, y1, x2, y2 = det['box']
                roi = prev_gray[max(0, y1):min(frame_h, y2), max(0, x1):min(frame_w, x2)]
                if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5: continue
                
                p0 = cv2.goodFeaturesToTrack(roi, **self.feature_params)
                if p0 is not None:
                    p0 += np.array([max(0, x1), max(0, y1)], dtype=np.float32)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **self.lk_params)
                    good_new, good_old = p1[st == 1], p0[st == 1]
                    
                    if len(good_new) >= 3: 
                        movement = np.median(good_new - good_old, axis=0).flatten()
                        dx, dy = int(movement[0]), int(movement[1])
                        
                        new_box = np.array([x1 + dx, y1 + dy, x2 + dx, y2 + dy])
                        new_box[0], new_box[2] = np.clip([new_box[0], new_box[2]], 0, frame_w)
                        new_box[1], new_box[3] = np.clip([new_box[1], new_box[3]], 0, frame_h)
                        
                        if (new_box[2] - new_box[0]) > 10 and (new_box[3] - new_box[1]) > 10:
                            p_det = det.copy()
                            p_det['box'] = new_box
                            p_det['conf'] *= 0.9 
                            if p_det['conf'] > 0.15: propagated_dets.append(p_det)
        return propagated_dets

    def heal_broken_tracks(self, frame_cache, max_gap_frames=50, max_dist_pixels=150):
        print(f"[INFO] PASS 2: Running Track Healing & Stitching...")
        tracks, remap_dict = {}, {}
        for f_idx in sorted(frame_cache.keys()):
            for det in frame_cache[f_idx]:
                tid, box, lbl = det['id'], det['box'], det['lbl']
                if tid not in tracks: tracks[tid] = {"start_f": f_idx, "end_f": f_idx, "start_box": box, "end_box": box, "class_votes": [lbl]}
                else:
                    tracks[tid]["end_f"], tracks[tid]["end_box"] = f_idx, box
                    tracks[tid]["class_votes"].append(lbl)

        sorted_ids = sorted(tracks.keys(), key=lambda x: tracks[x]['start_f'])
        get_center = lambda b: ((b[0]+b[2])//2, (b[1]+b[3])//2)
        get_primary = lambda v: Counter(v).most_common(1)[0][0]

        for i in range(len(sorted_ids)):
            id_b = sorted_ids[i]
            if id_b in remap_dict: continue
            track_b = tracks[id_b]
            class_b, center_b = get_primary(track_b['class_votes']), get_center(track_b['start_box'])
            best_match, min_dist = None, float('inf')
            
            for j in range(i-1, -1, -1):
                id_a = sorted_ids[j]
                while id_a in remap_dict: id_a = remap_dict[id_a]
                if id_a == id_b: continue
                track_a = tracks[id_a]
                gap = track_b['start_f'] - track_a['end_f']
                if gap <= 0 or gap > max_gap_frames: continue
                
                class_a = get_primary(track_a['class_votes'])
                if class_a != class_b and not (class_a in ['bus', 'truck'] and class_b in ['bus', 'truck']): continue 
                
                dist = np.sqrt((get_center(track_a['end_box'])[0]-center_b[0])**2 + (get_center(track_a['end_box'])[1]-center_b[1])**2)
                if dist < max_dist_pixels and dist < min_dist: 
                    min_dist, best_match = dist, id_a
            
            if best_match:
                remap_dict[id_b] = best_match
                tracks[best_match]['end_f'], tracks[best_match]['end_box'] = track_b['end_f'], track_b['end_box']
                tracks[best_match]['class_votes'].extend(track_b['class_votes'])

        for f_idx in frame_cache:
            for det in frame_cache[f_idx]:
                if det['id'] in remap_dict:
                    parent = det['id']
                    while parent in remap_dict: parent = remap_dict[parent]
                    det['id'] = parent
        return frame_cache

    def run_hybrid_pipeline(self, video_path, output_path, imu_csv, l1_pcap, l2_pcap, original_video_utc):
        cap = cv2.VideoCapture(video_path)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.setup_resolution(w, h)
        
        # TIME SYNC
        VIDEO_OFFSET_SECONDS = 20.0 * 60.0
        telemetry = TelemetrySync(imu_csv)
        t_start_unix = telemetry.get_unix_from_utc(original_video_utc)
        video_start_unix = t_start_unix + VIDEO_OFFSET_SECONDS
        
        print(f"[INFO] Loading Point Clouds...")
        t_min_lidar, t_max_lidar = video_start_unix - 1.0, video_start_unix + (total_frames / fps) + 1.0
        pts1, pts2 = lidar_pcap.load_both(l1_pcap, l2_pcap, t_min=t_min_lidar, t_max=t_max_lidar)
        projector = CalibratedProjector(self.FX, self.FY, self.CX, self.CY, self.T_LIDAR_TO_CAM)

        temp_id_hits, frame_cache = {}, {}
        prev_gray, prev_valid_dets = None, []
        
        # ---------------------------------------------------------
        # PASS 1: VISION & OPTICAL FLOW PROPAGATION
        # ---------------------------------------------------------
        print(f"[INFO] PASS 1: YOLO Detection + Optical Flow Propagation...")
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            results = self.model.track(frame, persist=True, verbose=False, conf=0.25, iou=0.45, imgsz=1920, tracker=r"C:\IITM\Vehicle_detection_360\FINAL_custom_track.yaml")[0]
            temp_dets = []
            
            if results.boxes.id is not None:
                for box, t_id, cls_idx, conf in zip(results.boxes.xyxy.cpu().numpy().astype(int), results.boxes.id.int().cpu().tolist(), results.boxes.cls.int().cpu().tolist(), results.boxes.conf.cpu().tolist()):
                    if box[3] < self.SKY_Y_LIMIT or cv2.pointPolygonTest(self.EGO_POLYGON, (int((box[0] + box[2]) / 2), int(box[3])), False) >= 0: continue
                    raw_label = self.model.names[cls_idx]
                    if raw_label in self.excluded_classes: continue
                    
                    req_conf = self.get_dynamic_threshold(box[3], raw_label)
                    final_thresh = req_conf if raw_label == 'auto' else (req_conf * 0.80 if temp_id_hits.get(t_id, 0) > 3 else req_conf)
                    if conf >= final_thresh: temp_dets.append({'box': box, 'id': t_id, 'conf': round(conf, 3), 'lbl': raw_label})

            # The Flow Fix: Keep dropped boxes alive!
            if prev_gray is not None and prev_valid_dets:
                flow_dets = self.propagate_missing_detections_with_flow(prev_gray, curr_gray, prev_valid_dets, temp_dets, w, h)
                temp_dets.extend(flow_dets)

            # Swallow NMS (Truck eating Car fix)
            valid_dets = []
            for i, det1 in enumerate(temp_dets):
                is_swallowed = False
                for j, det2 in enumerate(temp_dets):
                    if i == j: continue
                    if det2['lbl'] in ['bus', 'truck'] and det1['lbl'] not in ['bus', 'truck']:
                        x_left, y_top = max(det1['box'][0], det2['box'][0]), max(det1['box'][1], det2['box'][1])
                        x_right, y_bottom = min(det1['box'][2], det2['box'][2]), min(det1['box'][3], det2['box'][3])
                        if x_right > x_left and y_bottom > y_top and (((x_right - x_left) * (y_bottom - y_top)) / ((det1['box'][2] - det1['box'][0]) * (det1['box'][3] - det1['box'][1]))) > 0.85:
                            is_swallowed = True; break
                if not is_swallowed:
                    valid_dets.append(det1)
                    temp_id_hits[det1['id']] = temp_id_hits.get(det1['id'], 0) + 1

            frame_cache[frame_idx] = valid_dets
            prev_gray, prev_valid_dets = curr_gray, valid_dets
            frame_idx += 1
            if frame_idx % 20 == 0: print(f"   Processed Vision Frame {frame_idx}/{total_frames}...", end='\r')
        
        cap.release()
        print()
        
        # ---------------------------------------------------------
        # PASS 2: HEAL BROKEN TRACKS
        # ---------------------------------------------------------
        frame_cache = self.heal_broken_tracks(frame_cache)

        # ---------------------------------------------------------
        # PASS 3: LIDAR KINEMATICS PROJECTION
        # ---------------------------------------------------------
        print(f"[INFO] PASS 3: LiDAR Geospatial Projection on Healed Tracks...")
        id_kinematics = defaultdict(lambda: {"lats": [], "lons": [], "depths": []})
        
        for f_idx in sorted(frame_cache.keys()):
            current_t_unix = video_start_unix + (f_idx / fps)
            ego_lat, ego_lon, ego_yaw, ego_speed = telemetry.get_telemetry(current_t_unix)
            rel_t = current_t_unix - t_min_lidar

            if len(pts1) > 0:
                i1_s, i1_e = np.searchsorted(pts1[:, 0], rel_t - 0.05), np.searchsorted(pts1[:, 0], rel_t + 0.05)
                f_pts1 = pts1[i1_s:i1_e, 1:4]
            else: f_pts1 = np.empty((0, 3), dtype=np.float32)

            if len(pts2) > 0:
                i2_s, i2_e = np.searchsorted(pts2[:, 0], rel_t - 0.05), np.searchsorted(pts2[:, 0], rel_t + 0.05)
                f_pts2 = pts2[i2_s:i2_e, 1:4]
            else: f_pts2 = np.empty((0, 3), dtype=np.float32)

            frame_pts = np.vstack([f_pts1, f_pts2])
            proj_u, proj_v, proj_d = projector.project(frame_pts, h, w) if len(frame_pts) > 0 else (np.array([]), np.array([]), np.array([]))
            
            for det in frame_cache[f_idx]:
                depth = get_bbox_depth(proj_u, proj_v, proj_d, det['box']) if len(proj_u) > 0 else None
                det['depth'] = depth
                
                if depth is not None:
                    # True Ground Distance math to avoid edge pulling
                    raw_cx = (det['box'][0] + det['box'][2]) / 2.0
                    az_deg = math.degrees(math.atan2((raw_cx - self.CX), self.FX))
                    ground_dist = depth / math.cos(math.radians(az_deg))
                    t_br = (ego_yaw + az_deg) % 360.0
                    
                    t_lat, t_lon = fast_project_gps(ego_lat, ego_lon, ground_dist, t_br)
                    id_kinematics[det['id']]["lats"].append(t_lat)
                    id_kinematics[det['id']]["lons"].append(t_lon)
                    id_kinematics[det['id']]["depths"].append(depth)

        # ---------------------------------------------------------
        # PASS 4: PERMANENT CLASSIFICATION (The Hybrid Gate)
        # ---------------------------------------------------------
        print("[INFO] PASS 4: Assigning permanent PARKED/MOVING labels using Hybrid Kinematics...")
        id_final_states = {}
        # 45 frames = 1.5 seconds. Long enough to ignore bounding box sliding, 
        # short enough to catch oncoming cars.
        time_window_frames = int(fps * 1.5) 
        
        for obj_id, data in id_kinematics.items():
            lats, lons = data["lats"], data["lons"]
            
            # If we see it for less than 10 frames, assume MOVING to be safe
            if len(lats) < 10:
                id_final_states[obj_id] = "MOVING"
                continue
                
            # 5-Frame Smoothing to destroy LiDAR 1-frame glitches
            sm_lats = np.convolve(lats, np.ones(5)/5, mode='valid')
            sm_lons = np.convolve(lons, np.ones(5)/5, mode='valid')
            
            start_lat, start_lon = sm_lats[0], sm_lons[0]
            
            # GATE 1: THE OLD RELIABLE (Absolute Lifetime Distance)
            # If it EVER moves more than 15 meters from its starting point, it is moving.
            # (Parked cars will never trigger this, even with box sliding).
            max_travel = max([fast_distance_m(start_lat, start_lon, lat, lon) for lat, lon in zip(sm_lats, sm_lons)])
            
            if max_travel > 15.0:
                id_final_states[obj_id] = "MOVING"
                continue
                
            # GATE 2: THE SHORT-LIVED SUSTAINED SPEED CHECK
            # For cars that didn't live long enough to hit 15m.
            is_moving_fast = False
            if len(sm_lats) > time_window_frames:
                for i in range(len(sm_lats) - time_window_frames):
                    # Check distance traveled over a 1.5 second gap
                    dist_over_time = fast_distance_m(sm_lats[i], sm_lons[i], sm_lats[i+time_window_frames], sm_lons[i+time_window_frames])
                    # If it traveled more than 5 meters in 1.5 seconds (approx 12 km/h)
                    # This safely ignores a 2.5m bounding box slide!
                    if dist_over_time > 5.0:
                        is_moving_fast = True
                        break
            
            if is_moving_fast:
                id_final_states[obj_id] = "MOVING"
            else:
                id_final_states[obj_id] = "PARKED"

        # Resolve Labels & States back into the frame cache
        history_map = defaultdict(list)
        for f in frame_cache:
            for det in frame_cache[f]:
                history_map[det['id']].append(det['lbl'])
                
        master_id_map = {}
        for t_id, history in history_map.items():
            counts = Counter(history)
            final_lbl = counts.most_common(1)[0][0]
            if final_lbl in ['motorbike', 'auto'] and counts.get('car', 0) / len(history) > 0.20: final_lbl = 'car'
            master_id_map[t_id] = {'label': final_lbl, 'status': id_final_states.get(t_id, "MOVING")}

        # ---------------------------------------------------------
        # PASS 5: RENDER VIDEO
        # ---------------------------------------------------------
        print(f"[INFO] PASS 5: Rendering Perfect Tracked Video...")
        cap = cv2.VideoCapture(video_path)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        live_counts, counted_ids, f_idx = {name: 0 for name in ['car', 'auto', 'bus', 'truck', 'motorbike']}, set(), 0
        
        while cap.isOpened() and f_idx < total_frames:
            ret, frame = cap.read()
            if not ret: break
            
            for det in frame_cache.get(f_idx, []):
                t_id = det['id']
                if t_id in master_id_map:
                    lbl, status = master_id_map[t_id]['label'], master_id_map[t_id]['status']
                    if t_id not in counted_ids and lbl in live_counts:
                        live_counts[lbl] += 1
                        counted_ids.add(t_id)
                        
                    depth_str = f" {det.get('depth'):.1f}m" if det.get('depth') else ""
                    color = (0, 0, 255) if status == "PARKED" else self.colors.get(lbl, (255, 255, 255))
                    
                    cv2.rectangle(frame, (det['box'][0], det['box'][1]), (det['box'][2], det['box'][3]), color, 3) 
                    cv2.putText(frame, f"{t_id} {lbl} ({status}){depth_str}", (det['box'][0], det['box'][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            out.write(frame)
            f_idx += 1
            if f_idx % 20 == 0: print(f"   Rendering Frame {f_idx}/{total_frames}...", end='\r')
            
        cap.release()
        out.release()
        print(f"\n[SUCCESS] Pipeline Complete. Saved to: {output_path}")


if __name__ == "__main__":
    SOURCE_VIDEO = r"C:\IITM\Results\lens 1\1080p_lens1.mp4" 
    MODEL_WEIGHTS = r"C:\IITM\Vehicle_detection_360\runs\detect\FINAL_balanced_v11_6classes\weights\best.pt"
    
    IMU_DATA = r"E:\MUMMAS DATA COLLECTION-1080p\17012026\TRIP2\IMU-GPS\imu_all_topics_2026-01-17_15-21-18.csv"
    L1_DATA = r"E:\MUMMAS DATA COLLECTION-1080p\17012026\TRIP2\LIDAR\LiDAR1\lidar1_raw_20260117_150950.pcap"
    L2_DATA = r"E:\MUMMAS DATA COLLECTION-1080p\17012026\TRIP2\LIDAR\LiDAR2\lidar2_raw_20260117_150950.pcap"
    
    # Needs to extract UTC time from your metadata
    with open(r"E:\MUMMAS DATA COLLECTION-1080p\17012026\TRIP2\CAMERA\run_20260117_152118\session_meta.json", "r") as f: 
        UTC_START = json.load(f)["created_utc"]
    
    detector = HybridLiDARFlowDetector(model_path=MODEL_WEIGHTS)
    detector.run_hybrid_pipeline(
        video_path=SOURCE_VIDEO, 
        output_path=SOURCE_VIDEO.replace(".mp4", "_UltimateHybrid.mp4"),
        imu_csv=IMU_DATA,
        l1_pcap=L1_DATA,
        l2_pcap=L2_DATA,
        original_video_utc=UTC_START
    )

'''
import os
import cv2
import numpy as np
import json
import time
import math
from ultralytics import YOLO
from collections import Counter, defaultdict

# Imports from your existing sensor modules
from sync import TelemetrySync
import lidar_pcap
from perceive import CalibratedProjector, get_bbox_depth, fast_project_gps, fast_distance_m

class HybridLiDARFlowDetector:
    def __init__(self, model_path):
        print(f"[INFO] Loading Hybrid LiDAR-Flow Tracker...")
        self.model = YOLO(model_path)
        self.colors = {'auto': (0, 165, 255), 'bus': (255, 0, 0), 'car': (0, 255, 0), 'motorbike': (255, 255, 0), 'truck': (0, 0, 255)}
        self.excluded_classes = ['tractor', 'rickshaw', 'e-rickshaw', 'cart', 'person', 'cycle']
        self.CONF_FAR = 0.10        
        self.CONF_CLOSE = 0.30
        
        # --- OPTICAL FLOW PARAMETERS (For Box Propagation) ---
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners=20, qualityLevel=0.1, minDistance=7, blockSize=7)
        
        # --- SCALED INTRINSICS (1080x1920) ---
        self.FX, self.FY = 553.23, 538.88
        self.CX, self.CY = 540.0, 960.0    
        self.T_LIDAR_TO_CAM = np.array([
            [0.20927,   -0.977663, -0.0195126, 2.05416],
            [0.0386136,  0.0282009, -0.998856,  0.401883],
            [0.977095,   0.208277,  0.0436527, -0.035744],
            [0.0,        0.0,       0.0,       1.0]
        ])

    def setup_resolution(self, w, h):
        scale_x, scale_y = w / 3328, h / 1664
        self.SKY_Y_LIMIT = int(800 * scale_y)
        self.VEHICLE_Y_LIMIT = int(1150 * scale_y)
        self.VIP_Y_LIMIT = int(950 * scale_y)
        poly = [[0, 1021], [78, 1088], [341, 1165], [419, 1094], [612, 1121], [679, 1206], [875, 1223], [921, 1290], [1083, 1229], [1616, 1215], [1930, 1306], [2064, 1113], [2206, 1065], [2340, 1111], [2807, 1021], [3267, 1009], [3287, 1013], [3328, 1664], [0, 1664]]
        self.EGO_POLYGON = np.array([[int(x * scale_x), int(y * scale_y)] for x, y in poly], np.int32)

    def get_dynamic_threshold(self, bbox_bottom_y, label):
        if label in ['bus', 'truck']: return 0.15 if bbox_bottom_y > self.VIP_Y_LIMIT else 0.40  
        y_clamped = max(self.SKY_Y_LIMIT, min(bbox_bottom_y, self.VEHICLE_Y_LIMIT))
        ratio = 1.0 if self.VEHICLE_Y_LIMIT == self.SKY_Y_LIMIT else ((y_clamped - self.SKY_Y_LIMIT) / (self.VEHICLE_Y_LIMIT - self.SKY_Y_LIMIT)) ** 2 
        base_thresh = self.CONF_FAR + (self.CONF_CLOSE - self.CONF_FAR) * ratio
        if label == 'auto': return max(base_thresh, 0.60) 
        if label == 'motorbike': return max(base_thresh, 0.20)
        return max(base_thresh, 0.27)

    def propagate_missing_detections_with_flow(self, prev_gray, curr_gray, prev_dets, curr_dets, frame_w, frame_h):
        curr_ids = {det['id'] for det in curr_dets}
        propagated_dets = []
        for det in prev_dets:
            if det['id'] not in curr_ids:
                x1, y1, x2, y2 = det['box']
                roi = prev_gray[max(0, y1):min(frame_h, y2), max(0, x1):min(frame_w, x2)]
                if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5: continue
                
                p0 = cv2.goodFeaturesToTrack(roi, **self.feature_params)
                if p0 is not None:
                    p0 += np.array([max(0, x1), max(0, y1)], dtype=np.float32)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **self.lk_params)
                    good_new, good_old = p1[st == 1], p0[st == 1]
                    
                    if len(good_new) >= 3: 
                        movement = np.median(good_new - good_old, axis=0).flatten()
                        dx, dy = int(movement[0]), int(movement[1])
                        
                        new_box = np.array([x1 + dx, y1 + dy, x2 + dx, y2 + dy])
                        new_box[0], new_box[2] = np.clip([new_box[0], new_box[2]], 0, frame_w)
                        new_box[1], new_box[3] = np.clip([new_box[1], new_box[3]], 0, frame_h)
                        
                        if (new_box[2] - new_box[0]) > 10 and (new_box[3] - new_box[1]) > 10:
                            p_det = det.copy()
                            p_det['box'] = new_box
                            p_det['conf'] *= 0.9 
                            if p_det['conf'] > 0.15: propagated_dets.append(p_det)
        return propagated_dets

    def heal_broken_tracks(self, frame_cache, max_gap_frames=50, max_dist_pixels=150):
        print(f"[INFO] PASS 2: Running Track Healing & Stitching...")
        tracks, remap_dict = {}, {}
        for f_idx in sorted(frame_cache.keys()):
            for det in frame_cache[f_idx]:
                tid, box, lbl = det['id'], det['box'], det['lbl']
                if tid not in tracks: tracks[tid] = {"start_f": f_idx, "end_f": f_idx, "start_box": box, "end_box": box, "class_votes": [lbl]}
                else:
                    tracks[tid]["end_f"], tracks[tid]["end_box"] = f_idx, box
                    tracks[tid]["class_votes"].append(lbl)

        sorted_ids = sorted(tracks.keys(), key=lambda x: tracks[x]['start_f'])
        get_center = lambda b: ((b[0]+b[2])//2, (b[1]+b[3])//2)
        get_primary = lambda v: Counter(v).most_common(1)[0][0]

        for i in range(len(sorted_ids)):
            id_b = sorted_ids[i]
            if id_b in remap_dict: continue
            track_b = tracks[id_b]
            class_b, center_b = get_primary(track_b['class_votes']), get_center(track_b['start_box'])
            best_match, min_dist = None, float('inf')
            
            for j in range(i-1, -1, -1):
                id_a = sorted_ids[j]
                while id_a in remap_dict: id_a = remap_dict[id_a]
                if id_a == id_b: continue
                track_a = tracks[id_a]
                gap = track_b['start_f'] - track_a['end_f']
                if gap <= 0 or gap > max_gap_frames: continue
                
                class_a = get_primary(track_a['class_votes'])
                if class_a != class_b and not (class_a in ['bus', 'truck'] and class_b in ['bus', 'truck']): continue 
                
                dist = np.sqrt((get_center(track_a['end_box'])[0]-center_b[0])**2 + (get_center(track_a['end_box'])[1]-center_b[1])**2)
                if dist < max_dist_pixels and dist < min_dist: 
                    min_dist, best_match = dist, id_a
            
            if best_match:
                remap_dict[id_b] = best_match
                tracks[best_match]['end_f'], tracks[best_match]['end_box'] = track_b['end_f'], track_b['end_box']
                tracks[best_match]['class_votes'].extend(track_b['class_votes'])

        for f_idx in frame_cache:
            for det in frame_cache[f_idx]:
                if det['id'] in remap_dict:
                    parent = det['id']
                    while parent in remap_dict: parent = remap_dict[parent]
                    det['id'] = parent
        return frame_cache

    def run_hybrid_pipeline(self, video_path, output_path, imu_csv, l1_pcap, l2_pcap, original_video_utc):
        cap = cv2.VideoCapture(video_path)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.setup_resolution(w, h)
        
        # TIME SYNC
        VIDEO_OFFSET_SECONDS = 20.0 * 60.0
        telemetry = TelemetrySync(imu_csv)
        t_start_unix = telemetry.get_unix_from_utc(original_video_utc)
        video_start_unix = t_start_unix + VIDEO_OFFSET_SECONDS
        
        print(f"[INFO] Loading Point Clouds...")
        t_min_lidar, t_max_lidar = video_start_unix - 1.0, video_start_unix + (total_frames / fps) + 1.0
        pts1, pts2 = lidar_pcap.load_both(l1_pcap, l2_pcap, t_min=t_min_lidar, t_max=t_max_lidar)
        projector = CalibratedProjector(self.FX, self.FY, self.CX, self.CY, self.T_LIDAR_TO_CAM)

        temp_id_hits, frame_cache = {}, {}
        prev_gray, prev_valid_dets = None, []
        
        # ---------------------------------------------------------
        # PASS 1: VISION & OPTICAL FLOW PROPAGATION
        # ---------------------------------------------------------
        print(f"[INFO] PASS 1: YOLO Detection + Optical Flow Propagation...")
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            results = self.model.track(frame, persist=True, verbose=False, conf=0.25, iou=0.45, imgsz=1920, tracker=r"C:\IITM\Vehicle_detection_360\FINAL_custom_track.yaml")[0]
            temp_dets = []
            
            if results.boxes.id is not None:
                for box, t_id, cls_idx, conf in zip(results.boxes.xyxy.cpu().numpy().astype(int), results.boxes.id.int().cpu().tolist(), results.boxes.cls.int().cpu().tolist(), results.boxes.conf.cpu().tolist()):
                    if box[3] < self.SKY_Y_LIMIT or cv2.pointPolygonTest(self.EGO_POLYGON, (int((box[0] + box[2]) / 2), int(box[3])), False) >= 0: continue
                    raw_label = self.model.names[cls_idx]
                    if raw_label in self.excluded_classes: continue
                    
                    req_conf = self.get_dynamic_threshold(box[3], raw_label)
                    final_thresh = req_conf if raw_label == 'auto' else (req_conf * 0.80 if temp_id_hits.get(t_id, 0) > 3 else req_conf)
                    if conf >= final_thresh: temp_dets.append({'box': box, 'id': t_id, 'conf': round(conf, 3), 'lbl': raw_label})

            # The Flow Fix: Keep dropped boxes alive!
            if prev_gray is not None and prev_valid_dets:
                flow_dets = self.propagate_missing_detections_with_flow(prev_gray, curr_gray, prev_valid_dets, temp_dets, w, h)
                temp_dets.extend(flow_dets)

            # Swallow NMS (Truck eating Car fix)
            valid_dets = []
            for i, det1 in enumerate(temp_dets):
                is_swallowed = False
                for j, det2 in enumerate(temp_dets):
                    if i == j: continue
                    if det2['lbl'] in ['bus', 'truck'] and det1['lbl'] not in ['bus', 'truck']:
                        x_left, y_top = max(det1['box'][0], det2['box'][0]), max(det1['box'][1], det2['box'][1])
                        x_right, y_bottom = min(det1['box'][2], det2['box'][2]), min(det1['box'][3], det2['box'][3])
                        if x_right > x_left and y_bottom > y_top and (((x_right - x_left) * (y_bottom - y_top)) / ((det1['box'][2] - det1['box'][0]) * (det1['box'][3] - det1['box'][1]))) > 0.85:
                            is_swallowed = True; break
                if not is_swallowed:
                    valid_dets.append(det1)
                    temp_id_hits[det1['id']] = temp_id_hits.get(det1['id'], 0) + 1

            frame_cache[frame_idx] = valid_dets
            prev_gray, prev_valid_dets = curr_gray, valid_dets
            frame_idx += 1
            if frame_idx % 20 == 0: print(f"   Processed Vision Frame {frame_idx}/{total_frames}...", end='\r')
        
        cap.release()
        print()
        
        # ---------------------------------------------------------
        # PASS 2: HEAL BROKEN TRACKS
        # ---------------------------------------------------------
        frame_cache = self.heal_broken_tracks(frame_cache)

        # ---------------------------------------------------------
        # PASS 3: LIDAR KINEMATICS PROJECTION
        # ---------------------------------------------------------
        print(f"[INFO] PASS 3: LiDAR Geospatial Projection on Healed Tracks...")
        id_kinematics = defaultdict(lambda: {"lats": [], "lons": [], "depths": []})
        
        for f_idx in sorted(frame_cache.keys()):
            current_t_unix = video_start_unix + (f_idx / fps)
            ego_lat, ego_lon, ego_yaw, ego_speed = telemetry.get_telemetry(current_t_unix)
            rel_t = current_t_unix - t_min_lidar

            if len(pts1) > 0:
                i1_s, i1_e = np.searchsorted(pts1[:, 0], rel_t - 0.05), np.searchsorted(pts1[:, 0], rel_t + 0.05)
                f_pts1 = pts1[i1_s:i1_e, 1:4]
            else: f_pts1 = np.empty((0, 3), dtype=np.float32)

            if len(pts2) > 0:
                i2_s, i2_e = np.searchsorted(pts2[:, 0], rel_t - 0.05), np.searchsorted(pts2[:, 0], rel_t + 0.05)
                f_pts2 = pts2[i2_s:i2_e, 1:4]
            else: f_pts2 = np.empty((0, 3), dtype=np.float32)

            frame_pts = np.vstack([f_pts1, f_pts2])
            proj_u, proj_v, proj_d = projector.project(frame_pts, h, w) if len(frame_pts) > 0 else (np.array([]), np.array([]), np.array([]))
            
            for det in frame_cache[f_idx]:
                depth = get_bbox_depth(proj_u, proj_v, proj_d, det['box']) if len(proj_u) > 0 else None
                det['depth'] = depth
                
                if depth is not None:
                    # True Ground Distance math to avoid edge pulling
                    raw_cx = (det['box'][0] + det['box'][2]) / 2.0
                    az_deg = math.degrees(math.atan2((raw_cx - self.CX), self.FX))
                    ground_dist = depth / math.cos(math.radians(az_deg))
                    t_br = (ego_yaw + az_deg) % 360.0
                    
                    t_lat, t_lon = fast_project_gps(ego_lat, ego_lon, ground_dist, t_br)
                    id_kinematics[det['id']]["lats"].append(t_lat)
                    id_kinematics[det['id']]["lons"].append(t_lon)
                    id_kinematics[det['id']]["depths"].append(depth)

        # ---------------------------------------------------------
        # PASS 4: PERMANENT CLASSIFICATION
        # ---------------------------------------------------------
        print("[INFO] PASS 4: Assigning permanent PARKED/MOVING labels using GPS Paths...")
        id_final_states = {}
        for obj_id, data in id_kinematics.items():
            lats, lons = data["lats"], data["lons"]
            if len(lats) < 5:
                id_final_states[obj_id] = "MOVING"
                continue
                
            # 5-Frame Smoothing Obliterates any single-frame LiDAR spikes
            sm_lats = np.convolve(lats, np.ones(5)/5, mode='valid')
            sm_lons = np.convolve(lons, np.ones(5)/5, mode='valid')
            
            start_lat, start_lon = sm_lats[0], sm_lons[0]
            max_travel = max([fast_distance_m(start_lat, start_lon, lat, lon) for lat, lon in zip(sm_lats, sm_lons)])
            
            id_final_states[obj_id] = "MOVING" if max_travel > 15.0 else "PARKED"

        # Resolve Labels & States back into the frame cache
        history_map = defaultdict(list)
        for f in frame_cache:
            for det in frame_cache[f]:
                history_map[det['id']].append(det['lbl'])
                
        master_id_map = {}
        for t_id, history in history_map.items():
            counts = Counter(history)
            final_lbl = counts.most_common(1)[0][0]
            if final_lbl in ['motorbike', 'auto'] and counts.get('car', 0) / len(history) > 0.20: final_lbl = 'car'
            master_id_map[t_id] = {'label': final_lbl, 'status': id_final_states.get(t_id, "MOVING")}

        # ---------------------------------------------------------
        # PASS 5: RENDER VIDEO
        # ---------------------------------------------------------
        print(f"[INFO] PASS 5: Rendering Perfect Tracked Video...")
        cap = cv2.VideoCapture(video_path)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        live_counts, counted_ids, f_idx = {name: 0 for name in ['car', 'auto', 'bus', 'truck', 'motorbike']}, set(), 0
        
        while cap.isOpened() and f_idx < total_frames:
            ret, frame = cap.read()
            if not ret: break
            
            for det in frame_cache.get(f_idx, []):
                t_id = det['id']
                if t_id in master_id_map:
                    lbl, status = master_id_map[t_id]['label'], master_id_map[t_id]['status']
                    if t_id not in counted_ids and lbl in live_counts:
                        live_counts[lbl] += 1
                        counted_ids.add(t_id)
                        
                    depth_str = f" {det.get('depth'):.1f}m" if det.get('depth') else ""
                    color = (0, 0, 255) if status == "PARKED" else self.colors.get(lbl, (255, 255, 255))
                    
                    cv2.rectangle(frame, (det['box'][0], det['box'][1]), (det['box'][2], det['box'][3]), color, 3) 
                    cv2.putText(frame, f"{t_id} {lbl} ({status}){depth_str}", (det['box'][0], det['box'][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            out.write(frame)
            f_idx += 1
            if f_idx % 20 == 0: print(f"   Rendering Frame {f_idx}/{total_frames}...", end='\r')
            
        cap.release()
        out.release()
        print(f"\n[SUCCESS] Pipeline Complete. Saved to: {output_path}")


if __name__ == "__main__":
    SOURCE_VIDEO = r"C:\IITM\Results\lens 1\1080p_lens1.mp4" 
    MODEL_WEIGHTS = r"C:\IITM\Vehicle_detection_360\runs\detect\FINAL_balanced_v11_6classes\weights\best.pt"
    
    IMU_DATA = r"E:\MUMMAS DATA COLLECTION-1080p\17012026\TRIP2\IMU-GPS\imu_all_topics_2026-01-17_15-21-18.csv"
    L1_DATA = r"E:\MUMMAS DATA COLLECTION-1080p\17012026\TRIP2\LIDAR\LiDAR1\lidar1_raw_20260117_150950.pcap"
    L2_DATA = r"E:\MUMMAS DATA COLLECTION-1080p\17012026\TRIP2\LIDAR\LiDAR2\lidar2_raw_20260117_150950.pcap"
    
    # Needs to extract UTC time from your metadata
    with open(r"E:\MUMMAS DATA COLLECTION-1080p\17012026\TRIP2\CAMERA\run_20260117_152118\session_meta.json", "r") as f: 
        UTC_START = json.load(f)["created_utc"]
    
    detector = HybridLiDARFlowDetector(model_path=MODEL_WEIGHTS)
    detector.run_hybrid_pipeline(
        video_path=SOURCE_VIDEO, 
        output_path=SOURCE_VIDEO.replace(".mp4", "_UltimateHybrid.mp4"),
        imu_csv=IMU_DATA,
        l1_pcap=L1_DATA,
        l2_pcap=L2_DATA,
        original_video_utc=UTC_START
    )
    '''