'''import cv2
import json
import torch
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, Counter

# --- SAHI IMPORTS ---
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# --- CONFIGURATION ---
CUSTOM_MODEL_PATH = r"C:\IITM\Human_detection_360\yolo11m.pt"
VEHICLE_JSON_PATH = r"C:\IITM\Vehicle_detection_360\FINAL_panoramic_detection.json"
SOURCE_VIDEO_PATH = r"C:\IITM\Results\Final_Dynamic_Stitch_shortened.mp4" 
OUTPUT_VIDEO_PATH = r"C:\IITM\Results\Human_detection_360\pedestrians_double_checked.mp4"
JSON_OUTPUT_PATH = r"C:\IITM\Results\Human_detection_360\pedestrian_data.json"

TARGET_FPS = 30                 
MAX_SPEED_PX_PER_SEC = 130      
MIN_MOVEMENT_PX = 40            
VEHICLE_BOX_PADDING = 100       
EGO_STOPPED_THRESHOLD_PX = 3.5  

# --- CUSTOM EGO-VEHICLE POLYGON ---
EGO_POLYGON = np.array([
    [0, 1021], [78, 1088], [341, 1165], [419, 1094], 
    [612, 1121], [679, 1206], [875, 1223], [921, 1290], 
    [1083, 1229], [1616, 1215], [1930, 1306], [2064, 1113], 
    [2206, 1065], [2340, 1111], [2807, 1021], [3267, 1009], 
    [3287, 1013], 
    [3328, 1664], # Locked to bottom-right corner
    [0, 1664]     # Locked to bottom-left corner
], np.int32)

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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print("Loading SAHI Model...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics', 
        model_path=CUSTOM_MODEL_PATH,
        confidence_threshold=0.15,
        device=device,
    )
    
    vehicle_data = load_vehicle_data(VEHICLE_JSON_PATH)

    tracker = sv.ByteTrack(frame_rate=TARGET_FPS, track_activation_threshold=0.15, lost_track_buffer=TARGET_FPS)
    
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    track_history = defaultdict(list)
    track_total_age = defaultdict(int)    
    track_stopped_age = defaultdict(int)  

    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), TARGET_FPS, (width, height))
    json_data = []
    
    frame_stride = original_fps / TARGET_FPS
    original_frame_count = 0
    target_frame_count = 0

    print(f"Starting SAHI Inference on {width}x{height} at {TARGET_FPS} FPS...")
    print("Applying Custom Ego-Vehicle Polygon Mask...")
    
    seen_pedestrian_ids = set()

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

        # --- EGO MOTION ---
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

        # =========================================================
        # --- OPTIMIZED SAHI INFERENCE WITH POLYGON MASK ---
        # =========================================================
        
        # Create a copy of the frame just for SAHI inference
        sahi_frame = frame.copy()
        # Draw a solid black polygon over the car so YOLO ignores it completely
        cv2.fillPoly(sahi_frame, [EGO_POLYGON], (0, 0, 0))
        
        result = get_sliced_prediction(
            sahi_frame,
            detection_model,
            slice_height=832,         
            slice_width=832,           
            overlap_height_ratio=0.1,  
            overlap_width_ratio=0.15,  
            verbose=0
        )
        
        xyxy = []
        confidence = []
        class_id = []
        for obj in result.object_prediction_list:
            if obj.category.id == 0: 
                xyxy.append([
                    obj.bbox.minx, 
                    obj.bbox.miny, 
                    obj.bbox.maxx, 
                    obj.bbox.maxy
                ])
                confidence.append(obj.score.value)
                class_id.append(obj.category.id)

        if len(xyxy) > 0:
            people = sv.Detections(
                xyxy=np.array(xyxy),
                confidence=np.array(confidence),
                class_id=np.array(class_id)
            )
        else:
            people = sv.Detections.empty()

        # --- SPATIAL FILTER (VEHICLE RIDERS) ---
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

        # --- KINEMATIC FILTER (SPEED & POSTERS) ---
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

        # --- DYNAMIC VISIBILITY FILTER (3 to 90 RULE) ---
        valid_mask = []
        if tracked_pedestrians.tracker_id is not None:
            for t_id in tracked_pedestrians.tracker_id:
                
                track_total_age[t_id] += 1
                if is_vehicle_stopped:
                    track_stopped_age[t_id] += 1
                
                age = track_total_age[t_id]
                
                if age >= 3:
                    if age > 90 and track_stopped_age[t_id] < 10:
                        valid_mask.append(False)
                    else:
                        valid_mask.append(True)
                else:
                    valid_mask.append(False)
            
            tracked_pedestrians = tracked_pedestrians[np.array(valid_mask, dtype=bool)]

        if tracked_pedestrians.tracker_id is not None:
            for t_id in tracked_pedestrians.tracker_id:
                seen_pedestrian_ids.add(int(t_id))

        # --- DRAWING ---
        labels = []
        if tracked_pedestrians.tracker_id is not None:
            for t_id in tracked_pedestrians.tracker_id:
                speed = current_speeds.get(t_id, 0)
                labels.append(f"#{t_id} | {speed:.0f}px/s")
        
        annotated_frame = frame.copy()
        
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_pedestrians)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_pedestrians, labels=labels)

        current_cumulative_count = len(seen_pedestrian_ids)
        cv2.putText(
            annotated_frame,
            f"Total Pedestrians: {current_cumulative_count}",
            (40, 70), 
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5, 
            (0, 255, 255), 
            4, 
            cv2.LINE_AA
        )

        # --- EXPORT ---
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

    cap.release()
    out.release()
    
    with open(JSON_OUTPUT_PATH, 'w') as f: 
        json.dump(json_data, f, indent=4)

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
    main()'''

import cv2
import json
import torch
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, Counter

# --- SAHI IMPORTS ---
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# --- NEW: DEEPSORT IMPORT ---
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- CONFIGURATION ---
CUSTOM_MODEL_PATH = r"C:\IITM\Human_detection_360\yolo11m.pt"
SOURCE_VIDEO_PATH = r"C:\IITM\Results\Final_SIFT_Shifted_1080p.mp4" 
OUTPUT_VIDEO_PATH = r"C:\IITM\Results\Human_detection_360\pedestrians_double_checked.mp4"
JSON_OUTPUT_PATH = r"C:\IITM\Results\Human_detection_360\pedestrian_data.json"

TARGET_FPS = 30                 
MAX_SPEED_PX_PER_SEC = 75       # SCALED for 1920px width
EGO_STOPPED_THRESHOLD_PX = 2.0  # SCALED for 1920px width

# --- BIOLOGICAL GEOMETRY LIMITS ---
MAX_PED_WIDTH_PX = 260          # SCALED: Paintings close to the car will exceed this
MAX_PED_HEIGHT_PX = 520         # SCALED: Paintings close to the car will exceed this
MIN_ASPECT_RATIO = 1.1          # Height MUST be at least 1.1x the Width

# --- CUSTOM EGO-VEHICLE POLYGON (Based on original 3328x1664) ---
# Note: The script will automatically scale these coordinates to fit your 1920x960 video
BASE_EGO_POLYGON = np.array([
    [0, 1021], [78, 1088], [341, 1165], [419, 1094], 
    [612, 1121], [679, 1206], [875, 1223], [921, 1290], 
    [1083, 1229], [1616, 1215], [1930, 1306], [2064, 1113], 
    [2206, 1065], [2340, 1111], [2807, 1021], [3267, 1009], 
    [3287, 1013], 
    [3328, 1664], 
    [0, 1664]     
], np.int32)

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print("Loading SAHI Model...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics', 
        model_path=CUSTOM_MODEL_PATH,
        confidence_threshold=0.15,
        image_size=640, # SCALED: 640 is optimal for 1920x960 video
        device=device,
    )

    # --- REPLACED BYTETRACK WITH DEEPSORT ---
    print("Initializing DeepSORT ReID Tracker...")
    tracker = DeepSort(
        max_age=TARGET_FPS * 5, 
        n_init=3, 
        nms_max_overlap=1.0, 
        max_cosine_distance=0.2,
        embedder="mobilenet",
        half=True if device == "cuda:0" else False
    )
    
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    track_history = defaultdict(list)
    track_total_age = defaultdict(int)    
    track_stopped_age = defaultdict(int)  

    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # --- NEW: Dynamically scale the ego polygon to match the video resolution ---
    scale_x = width / 3328.0
    scale_y = height / 1664.0
    scaled_ego_polygon = (BASE_EGO_POLYGON * [scale_x, scale_y]).astype(np.int32)
    # -------------------------------------------------------------------------
    
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), TARGET_FPS, (width, height))
    json_data = []
    
    frame_stride = original_fps / TARGET_FPS
    original_frame_count = 0
    target_frame_count = 0

    print(f"Starting SAHI Inference on {width}x{height} at {TARGET_FPS} FPS...")
    print("Applying Scaled Biological Geometry Filters & Tuned Vehicle Padding...")
    
    seen_pedestrian_ids = set()

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

        # =========================================================
        # --- OPTIMIZED SAHI INFERENCE ---
        # =========================================================
        sahi_frame = frame.copy()
        # Draw the dynamically scaled polygon
        cv2.fillPoly(sahi_frame, [scaled_ego_polygon], (0, 0, 0))
        
        result = get_sliced_prediction(
            sahi_frame,
            detection_model,
            slice_height=640,          # SCALED for 960p height
            slice_width=640,           # SCALED for 1920p width
            overlap_height_ratio=0.15,  
            overlap_width_ratio=0.25,        
            postprocess_type="NMS",          
            postprocess_match_threshold=0.40, 
            verbose=0
        )
        
        xyxy = []
        confidence = []
        class_id = []
        current_vehicle_bboxes = [] 
        
        for obj in result.object_prediction_list:
            bbox = [obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy]
            
            if obj.category.id == 0: 
                xyxy.append(bbox)
                confidence.append(obj.score.value)
                class_id.append(obj.category.id)
                
            elif obj.category.id in [2, 3, 5, 7]:
                current_vehicle_bboxes.append((bbox, obj.category.id))

        if len(xyxy) > 0:
            people = sv.Detections(
                xyxy=np.array(xyxy),
                confidence=np.array(confidence),
                class_id=np.array(class_id)
            )
        else:
            people = sv.Detections.empty()

        # --- BIOLOGICAL & SPATIAL FILTER ---
        pedestrian_mask = np.ones(len(people), dtype=bool)

        for i, p_box in enumerate(people.xyxy):
            box_width = p_box[2] - p_box[0]
            box_height = p_box[3] - p_box[1]
            p_area = box_width * box_height
            px_center = (p_box[0] + p_box[2]) / 2
            py_bottom = p_box[3]
            
            # --- Size & Aspect Ratio Filter ---
            if box_width > 0:
                aspect_ratio = box_height / box_width
            else:
                aspect_ratio = 0
                
            if box_width > MAX_PED_WIDTH_PX or box_height > MAX_PED_HEIGHT_PX or aspect_ratio < MIN_ASPECT_RATIO:
                pedestrian_mask[i] = False
                continue 

            # --- SCALED VEHICLE PADDING ---
            for v_data in current_vehicle_bboxes:
                v_box, v_class = v_data
                
                # Scaled down padding for 1920x960 resolution
                if v_class == 3: # Motorcycle
                    pad_x = 60
                    pad_y = 45
                else:            # Car, Bus, Truck
                    pad_x = 10
                    pad_y = 10
                
                v_xA = v_box[0] - pad_x
                v_yA = v_box[1] - pad_y
                v_xB = v_box[2] + pad_x
                v_yB = v_box[3] + pad_y
                
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
        
        # =========================================================
        # --- DEEPSORT UPDATE LOGIC ---
        # =========================================================
        deepsort_bbs = []
        for i in range(len(spatial_filtered_people)):
            bbox = spatial_filtered_people.xyxy[i]
            conf = spatial_filtered_people.confidence[i]
            cls = spatial_filtered_people.class_id[i]
            
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            deepsort_bbs.append(([x1, y1, w, h], conf, cls))

        ds_tracks = tracker.update_tracks(deepsort_bbs, frame=frame)
        
        ds_xyxy = []
        ds_tracker_id = []
        ds_conf = []
        ds_class_id = []

        for track in ds_tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
                
            ltrb = track.to_ltrb()
            ds_xyxy.append([ltrb[0], ltrb[1], ltrb[2], ltrb[3]])
            ds_tracker_id.append(int(track.track_id))
            ds_conf.append(track.get_det_conf() if track.get_det_conf() is not None else 1.0)
            ds_class_id.append(0) # Reattach Class 0 (Person) for annotator

        if len(ds_xyxy) > 0:
            tracked_pedestrians = sv.Detections(
                xyxy=np.array(ds_xyxy),
                confidence=np.array(ds_conf),
                tracker_id=np.array(ds_tracker_id),
                class_id=np.array(ds_class_id)
            )
        else:
            tracked_pedestrians = sv.Detections.empty()
        # =========================================================

        # --- KINEMATIC FILTER (SPEED & POSTERS) ---
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
                        box_width = bbox[2] - bbox[0]
                        dynamic_min_movement = max(5.0, box_width * 0.15) 
                        
                        if dist < dynamic_min_movement:
                            is_pedestrian = False 
                
                speed_mask.append(is_pedestrian)
                
            tracked_pedestrians = tracked_pedestrians[np.array(speed_mask, dtype=bool)]

        # --- DYNAMIC VISIBILITY FILTER (3 to 150 RULE) ---
        valid_mask = []
        if tracked_pedestrians.tracker_id is not None:
            for t_id in tracked_pedestrians.tracker_id:
                
                track_total_age[t_id] += 1
                if is_vehicle_stopped:
                    track_stopped_age[t_id] += 1
                
                age = track_total_age[t_id]
                
                if age >= 3:
                    if age > 150 and track_stopped_age[t_id] < 10:
                        valid_mask.append(False)
                    else:
                        valid_mask.append(True)
                else:
                    valid_mask.append(False)
            
            tracked_pedestrians = tracked_pedestrians[np.array(valid_mask, dtype=bool)]

        if tracked_pedestrians.tracker_id is not None:
            for t_id in tracked_pedestrians.tracker_id:
                seen_pedestrian_ids.add(int(t_id))

        # --- DRAWING ---
        labels = []
        if tracked_pedestrians.tracker_id is not None:
            for t_id in tracked_pedestrians.tracker_id:
                speed = current_speeds.get(t_id, 0)
                labels.append(f"#{t_id} | {speed:.0f}px/s")
        
        annotated_frame = frame.copy()
        
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_pedestrians)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_pedestrians, labels=labels)

        current_cumulative_count = len(seen_pedestrian_ids)
        cv2.putText(
            annotated_frame,
            f"Total Pedestrians: {current_cumulative_count}",
            (40, 70), 
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5, 
            (0, 255, 255), 
            4, 
            cv2.LINE_AA
        )

        # --- EXPORT ---
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

    cap.release()
    out.release()
    
    with open(JSON_OUTPUT_PATH, 'w') as f: 
        json.dump(json_data, f, indent=4)

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