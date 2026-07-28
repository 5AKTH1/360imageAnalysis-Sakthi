'''import cv2
import json
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict

SOURCE_VIDEO_PATH = "C:/IITM/Results/Final_SIFT_Shifted_1080p.mp4"
OUTPUT_VIDEO_PATH = "C:/IITM/Results/Human_detection_360/pedestrians_double_checked.mp4"
JSON_OUTPUT_PATH = "C:/IITM/Results/Human_detection_360/pedestrian_data.json"
MODEL_NAME = "yolo11m.pt"
TARGET_FPS = 20

# SET YOUR CALIBRATED SPEED HERE
# (If a person moves faster than this in pixels per second, they are dropped)
MAX_SPEED_PX_PER_SEC = 150 

def main():
    model = YOLO(MODEL_NAME)
    tracker = sv.ByteTrack(frame_rate=TARGET_FPS, track_activation_threshold=0.15, lost_track_buffer=TARGET_FPS)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), TARGET_FPS, (width, height))
    json_data = []
    
    frame_stride = original_fps / TARGET_FPS
    original_frame_count = 0
    target_frame_count = 0

    VEHICLE_CLASSES = [1, 2, 3, 5, 7]
    track_history = defaultdict(list)

    print("Starting Double-Check Inference (Spatial + Kinematic)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        original_frame_count += 1
        if original_frame_count < target_frame_count * frame_stride:
            continue
            
        target_frame_count += 1
        
        results = model(frame, imgsz=1280, conf=0.15, iou=0.6, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        people = detections[detections.class_id == 0]
        vehicles = detections[np.isin(detections.class_id, VEHICLE_CLASSES)]

        # ==========================================
        # CHECK 1: SPATIAL FILTER (Overlap & Feet)
        # ==========================================
        pedestrian_mask = np.ones(len(people), dtype=bool)

        for i, p_box in enumerate(people.xyxy):
            p_area = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
            px_center = (p_box[0] + p_box[2]) / 2
            py_bottom = p_box[3]
            
            for v_box in vehicles.xyxy:
                xA = max(p_box[0], v_box[0])
                yA = max(p_box[1], v_box[1])
                xB = min(p_box[2], v_box[2])
                yB = min(p_box[3], v_box[3])
                
                interArea = max(0, xB - xA) * max(0, yB - yA)
                
                if p_area > 0:
                    overlap_ratio = interArea / p_area
                    feet_inside = (v_box[0] - 15 <= px_center <= v_box[2] + 15) and \
                                  (v_box[1] - 15 <= py_bottom <= v_box[3] + 15)
                    
                    if overlap_ratio > 0.30 or feet_inside:
                        pedestrian_mask[i] = False
                        break

        # Apply Check 1 before sending to tracker
        spatial_filtered_people = people[pedestrian_mask]
        tracked_pedestrians = tracker.update_with_detections(spatial_filtered_people)

        # ==========================================
        # CHECK 2: KINEMATIC FILTER (Speed)
        # ==========================================
        current_speeds = {}
        
        if tracked_pedestrians.tracker_id is not None and len(tracked_pedestrians.tracker_id) > 0:
            speed_mask = [] # Reset mask as a list of booleans
            
            for bbox, track_id in zip(tracked_pedestrians.xyxy, tracked_pedestrians.tracker_id):
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                
                track_history[track_id].append((cx, cy))
                
                if len(track_history[track_id]) > TARGET_FPS:
                    track_history[track_id].pop(0)
                
                is_pedestrian = True 
                
                if len(track_history[track_id]) >= 5:
                    old_cx, old_cy = track_history[track_id][0]
                    dist = np.sqrt((cx - old_cx)**2 + (cy - old_cy)**2)
                    time_elapsed = (len(track_history[track_id]) - 1) / TARGET_FPS
                    
                    speed_px_per_sec = dist / time_elapsed
                    current_speeds[track_id] = speed_px_per_sec
                    
                    if speed_px_per_sec > MAX_SPEED_PX_PER_SEC:
                        is_pedestrian = False
                
                speed_mask.append(is_pedestrian)
                
            # Convert list to a boolean numpy array to avoid IndexError
            tracked_pedestrians = tracked_pedestrians[np.array(speed_mask, dtype=bool)]

        # ==========================================
        # ANNOTATION & EXPORT
        # ==========================================
        labels = []
        if tracked_pedestrians.tracker_id is not None:
            for t_id, c in zip(tracked_pedestrians.tracker_id, tracked_pedestrians.confidence):
                speed = current_speeds.get(t_id, 0)
                labels.append(f"#{t_id} | {speed:.0f}px/s")
        
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=tracked_pedestrians)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_pedestrians, labels=labels)

        frame_entries = []
        if tracked_pedestrians.tracker_id is not None:
            for bbox, track_id, conf in zip(tracked_pedestrians.xyxy, tracked_pedestrians.tracker_id, tracked_pedestrians.confidence):
                frame_entries.append({
                    "id": int(track_id),
                    "confidence": float(round(conf, 4)),
                    "speed_px_s": float(round(current_speeds.get(track_id, 0), 2)),
                    "bbox": [float(x) for x in bbox]
                })

        json_data.append({
            "frame": target_frame_count,
            "timestamp": round(target_frame_count / TARGET_FPS, 3),
            "total_pedestrians": len(frame_entries),
            "detections": frame_entries
        })

        out.write(annotated_frame)
        if target_frame_count % 10 == 0:
            print(f"Processed frame {target_frame_count}...", end='\r')

    cap.release()
    out.release()
    
    with open(JSON_OUTPUT_PATH, 'w') as f: 
        json.dump(json_data, f, indent=4)
        
    print(f"\nDone! Video saved to {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()'''

import cv2
import json
import torch
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, Counter

# --- CONFIGURATION ---
CUSTOM_MODEL_PATH = r"C:\IITM\Human_detection_360\yolo11m.pt"
VEHICLE_JSON_PATH = r"C:\IITM\Vehicle_detection_360\FINAL_panoramic_detection.json"
SOURCE_VIDEO_PATH = r"C:\IITM\Results\Final_SIFT_Shifted_1080p.mp4"
OUTPUT_VIDEO_PATH = r"C:\IITM\Results\Human_detection_360\pedestrians_double_checked.mp4"
JSON_OUTPUT_PATH = r"C:\IITM\Results\Human_detection_360\pedestrian_data.json"

TARGET_FPS = 20 
MAX_SPEED_PX_PER_SEC = 120
MIN_MOVEMENT_PX = 15
VEHICLE_BOX_PADDING = 30
EGO_STOPPED_THRESHOLD_PX = 2.0  # If background shifts < 2px per frame, car is stopped

def load_vehicle_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    vehicle_dict = {}
    for entry in data.get("detections", []):
        frame_num = entry.get("frame", 0)
        boxes = [obj["bbox"] for obj in entry.get("objects", [])]
        vehicle_dict[frame_num] = boxes
    return vehicle_dict

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    person_model = YOLO(CUSTOM_MODEL_PATH).to(device)
    vehicle_data = load_vehicle_data(VEHICLE_JSON_PATH)

    tracker = sv.ByteTrack(frame_rate=TARGET_FPS, track_activation_threshold=0.15, lost_track_buffer=TARGET_FPS)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    # --- MEMORY BANKS ---
    track_history = defaultdict(list)
    track_total_age = defaultdict(int)    # Tracks absolute lifespan
    track_stopped_age = defaultdict(int)  # Tracks lifespan while vehicle was stopped

    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), TARGET_FPS, (width, height))
    json_data = []
    
    frame_stride = original_fps / TARGET_FPS
    original_frame_count = 0
    target_frame_count = 0

    print("Starting Dynamic Inference & Tracking...")

    # ==========================================
    # PHASE 1: VIDEO PROCESSING
    # ==========================================
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if original_frame_count < target_frame_count * frame_stride:
            original_frame_count += 1
            continue
        original_frame_count += 1
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        camera_dx, camera_dy = 0.0, 0.0
        ego_movement = 0.0

        if 'prev_gray' in locals():
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
            if prev_pts is not None:
                curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
                if status is not None:
                    mask = status.flatten() == 1
                    good_prev = prev_pts[mask].reshape(-1, 2)
                    good_curr = curr_pts[mask].reshape(-1, 2)

                    if len(good_prev) > 0:
                        dxs = good_curr[:, 0] - good_prev[:, 0]
                        dys = good_curr[:, 1] - good_prev[:, 1]
                        camera_dx = float(np.median(dxs))
                        camera_dy = float(np.median(dys))
                        ego_movement = np.sqrt(camera_dx**2 + camera_dy**2)

        prev_gray = gray.copy()
        is_vehicle_stopped = ego_movement < EGO_STOPPED_THRESHOLD_PX

        for t_id in track_history.keys():
            for i in range(len(track_history[t_id])):
                old_cx, old_cy = track_history[t_id][i]
                track_history[t_id][i] = (old_cx + camera_dx, old_cy + camera_dy)

        results = person_model(frame, imgsz=1280, conf=0.15, iou=0.6, verbose=False, half=True)[0]
        detections = sv.Detections.from_ultralytics(results)
        people = detections[detections.class_id == 0]

        current_vehicle_bboxes = vehicle_data.get(original_frame_count - 1, [])
        pedestrian_mask = np.ones(len(people), dtype=bool)

        for i, p_box in enumerate(people.xyxy):
            p_area = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
            px_center = (p_box[0] + p_box[2]) / 2
            py_bottom = p_box[3]
            
            for v_box in current_vehicle_bboxes:
                v_xA = v_box[0] - VEHICLE_BOX_PADDING
                v_yA = v_box[1] - VEHICLE_BOX_PADDING
                v_xB = v_box[2] + VEHICLE_BOX_PADDING
                v_yB = v_box[3] + VEHICLE_BOX_PADDING
                
                xA, yA = max(p_box[0], v_xA), max(p_box[1], v_yA)
                xB, yB = min(p_box[2], v_xB), min(p_box[3], v_yB)
                interArea = max(0, xB - xA) * max(0, yB - yA)
                
                if p_area > 0:
                    overlap_ratio = interArea / p_area
                    feet_inside = (v_xA <= px_center <= v_xB) and (v_yA <= py_bottom <= v_yB)
                    
                    if overlap_ratio > 0.15 or feet_inside:
                        pedestrian_mask[i] = False
                        break

        spatial_filtered_people = people[pedestrian_mask]
        tracked_pedestrians = tracker.update_with_detections(spatial_filtered_people)

        current_speeds = {}
        if tracked_pedestrians.tracker_id is not None and len(tracked_pedestrians.tracker_id) > 0:
            speed_mask = []
            
            for bbox, track_id in zip(tracked_pedestrians.xyxy, tracked_pedestrians.tracker_id):
                cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
                track_history[track_id].append((cx, cy))
                
                if len(track_history[track_id]) > TARGET_FPS:
                    track_history[track_id].pop(0)
                
                is_pedestrian = True 
                
                if len(track_history[track_id]) >= 3:
                    old_cx, old_cy = track_history[track_id][0]
                    dist = np.sqrt((cx - old_cx)**2 + (cy - old_cy)**2)
                    time_elapsed = (len(track_history[track_id]) - 1) / TARGET_FPS
                    speed_px_per_sec = dist / time_elapsed
                    current_speeds[track_id] = speed_px_per_sec
                    
                    if speed_px_per_sec > MAX_SPEED_PX_PER_SEC:
                        is_pedestrian = False
                        
                    if len(track_history[track_id]) == TARGET_FPS:
                        if dist < MIN_MOVEMENT_PX:
                            is_pedestrian = False 
                
                speed_mask.append(is_pedestrian)
                
            tracked_pedestrians = tracked_pedestrians[np.array(speed_mask, dtype=bool)]

        # =========================================================
        # --- NEW: DYNAMIC VISIBILITY FILTER (3 to 60 RULE) ---
        # =========================================================
        valid_mask = []
        if tracked_pedestrians.tracker_id is not None:
            for t_id in tracked_pedestrians.tracker_id:
                
                # Update absolute ages
                track_total_age[t_id] += 1
                if is_vehicle_stopped:
                    track_stopped_age[t_id] += 1
                
                age = track_total_age[t_id]
                
                if age >= 3:
                    if age > 60 and track_stopped_age[t_id] < 10:
                        # Dropped: Survived > 60 frames while car was mostly moving
                        valid_mask.append(False)
                    else:
                        # Kept: Under 60 frames OR vehicle was stopped
                        valid_mask.append(True)
                else:
                    # Dropped: Hasn't proven existence (under 3 frames)
                    valid_mask.append(False)
            
            tracked_pedestrians = tracked_pedestrians[np.array(valid_mask, dtype=bool)]
        # =========================================================

        labels = []
        if tracked_pedestrians.tracker_id is not None:
            for t_id in tracked_pedestrians.tracker_id:
                speed = current_speeds.get(t_id, 0)
                labels.append(f"#{t_id} | {speed:.0f}px/s")
        
        annotated_frame = frame.copy()
        
        # NOTE: Red and Yellow debugging boxes removed here
        
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_pedestrians)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_pedestrians, labels=labels)

        frame_entries = []
        if tracked_pedestrians.tracker_id is not None:
            for bbox, track_id, conf in zip(tracked_pedestrians.xyxy, tracked_pedestrians.tracker_id, tracked_pedestrians.confidence):
                frame_entries.append({
                    "id": int(track_id),
                    "confidence": float(round(conf, 4)),
                    "speed_px_s": float(round(current_speeds.get(track_id, 0.0), 2)),
                    "bbox": [float(x) for x in bbox]
                })

        json_data.append({
            "frame": int(target_frame_count),
            "timestamp": float(round(target_frame_count / TARGET_FPS, 3)),
            "total_pedestrians": int(len(frame_entries)),
            "detections": frame_entries
        })

        out.write(annotated_frame)
        
        target_frame_count += 1 
        
        if target_frame_count % 10 == 0:
            status = "STOPPED" if is_vehicle_stopped else "MOVING"
            print(f"Processed frame {target_frame_count}... | Car: {status} ", end='\r')

    # End of while loop
    cap.release()
    out.release()
    
    with open(JSON_OUTPUT_PATH, 'w') as f: 
        json.dump(json_data, f, indent=4)

    # ==========================================
    # PHASE 2: DATA ANALYSIS & REPORTING
    # ==========================================
    print("\n\n" + "="*40)
    print("ANALYSIS COMPLETE. CALCULATING RESULTS...")
    
    unique_ids = set()

    for frame_data in json_data:
        for detection in frame_data.get("detections", []):
            ped_id = detection["id"]
            unique_ids.add(ped_id)

    print("=" * 40)
    print(f"Video Saved : {OUTPUT_VIDEO_PATH}")
    print(f"Data Saved  : {JSON_OUTPUT_PATH}")
    print("-" * 40)
    print(f"Total Unique Valid Pedestrians Detected : {len(unique_ids)}")
    print("=" * 40)

if __name__ == "__main__":
    main()