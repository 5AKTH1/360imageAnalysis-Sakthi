'''import os
import cv2
import numpy as np
import json
from ultralytics import YOLO
from collections import Counter

class PanoramicDirectDetector:
    def __init__(self, model_path):
        print(f"[INFO] Loading Model: {model_path}...")
        self.model = YOLO(model_path)
        self.colors = {
            'auto': (0, 165, 255), 'bus': (255, 0, 0), 'car': (0, 255, 0),
            'motorbike': (255, 255, 0), 'truck': (0, 0, 255)
        }
        self.excluded_classes = ['tractor', 'rickshaw', 'e-rickshaw', 'cart', 'person', 'cycle']
        
        # --- CONFIGURATION ---
        self.SKY_Y_LIMIT = 350      
        self.VEHICLE_Y_LIMIT = 960  
        self.CONF_FAR = 0.25
        self.CONF_CLOSE = 0.65

    def get_dynamic_threshold(self, bbox_bottom_y, label):
        y_clamped = max(self.SKY_Y_LIMIT, min(bbox_bottom_y, self.VEHICLE_Y_LIMIT))
        ratio = (y_clamped - self.SKY_Y_LIMIT) / (self.VEHICLE_Y_LIMIT - self.SKY_Y_LIMIT)
        base_thresh = self.CONF_FAR + (self.CONF_CLOSE - self.CONF_FAR) * ratio
        
        if label == 'auto':
            return max(base_thresh, 0.55) 
            
        if label == 'motorbike':
            return max(base_thresh, 0.45)

        return base_thresh

    def heal_broken_tracks(self, frame_cache, max_gap_frames=50, max_dist_pixels=150):
        print(f"[INFO] Running Track Healing (Merging broken IDs)...")
        tracks = {}
        sorted_frames = sorted(frame_cache.keys())
        
        for f_idx in sorted_frames:
            for det in frame_cache[f_idx]:
                tid = det['id']
                box = det['box']
                lbl = det['lbl'] 
                if tid not in tracks:
                    tracks[tid] = {
                        "start_f": f_idx, "end_f": f_idx, 
                        "start_box": box, "end_box": box, 
                        "class_votes": [lbl]
                    }
                else:
                    tracks[tid]["end_f"] = f_idx
                    tracks[tid]["end_box"] = box
                    tracks[tid]["class_votes"].append(lbl)

        sorted_ids = sorted(tracks.keys(), key=lambda x: tracks[x]['start_f'])
        remap_dict = {} 
        
        def get_center(box): return ((box[0]+box[2])//2, (box[1]+box[3])//2)
        def get_primary_class(votes): return Counter(votes).most_common(1)[0][0]

        for i in range(len(sorted_ids)):
            id_b = sorted_ids[i]
            if id_b in remap_dict: continue
            
            track_b = tracks[id_b]
            class_b = get_primary_class(track_b['class_votes'])
            center_b = get_center(track_b['start_box'])
            
            best_match = None
            min_dist = float('inf')
            
            for j in range(i-1, -1, -1):
                id_a = sorted_ids[j]
                while id_a in remap_dict: id_a = remap_dict[id_a]
                if id_a == id_b: continue
                
                track_a = tracks[id_a]
                gap = track_b['start_f'] - track_a['end_f']
                
                if gap <= 0 or gap > max_gap_frames: continue
                
                class_a = get_primary_class(track_a['class_votes'])
                if class_a != class_b:
                    is_heavy_a = class_a in ['bus', 'truck']
                    is_heavy_b = class_b in ['bus', 'truck']
                    if not (is_heavy_a and is_heavy_b):
                        continue 
                
                center_a = get_center(track_a['end_box'])
                dist = np.sqrt((center_a[0]-center_b[0])**2 + (center_a[1]-center_b[1])**2)
                
                if dist < max_dist_pixels and dist < min_dist:
                    min_dist = dist
                    best_match = id_a
            
            if best_match:
                remap_dict[id_b] = best_match
                tracks[best_match]['end_f'] = track_b['end_f']
                tracks[best_match]['end_box'] = track_b['end_box']
                tracks[best_match]['class_votes'].extend(track_b['class_votes'])

        print(f"[INFO] Healed {len(remap_dict)} broken tracks.")
        for f_idx in frame_cache:
            for det in frame_cache[f_idx]:
                if det['id'] in remap_dict:
                    parent = det['id']
                    while parent in remap_dict: parent = remap_dict[parent]
                    det['id'] = parent
        return frame_cache

    def run_video_inference(self, video_path, output_path):
        if not os.path.exists(video_path): return {}

        cap = cv2.VideoCapture(video_path)
        w, h = int(cap.get(3)), int(cap.get(4))
        fps = 20.0  
        
        full_metadata = {
            "source_file": video_path,
            "resolution": f"{w}x{h}",
            "processed_fps": fps,
            "summary_counts": {},
            "detections": [] 
        }
        
        temp_id_hits = {} 
        frame_cache = {}      
        
        print(f"[INFO] PASS 1: Analysis & Tracking...")
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            results = self.model.track(frame, 
                               persist=True, 
                               verbose=False, 
                               conf=0.20, 
                               iou=0.45, 
                               imgsz=1920, 
                               tracker="FINAL_custom_track.yaml")[0]
            
            frame_data = []
            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                ids = results.boxes.id.int().cpu().tolist()
                clss = results.boxes.cls.int().cpu().tolist()
                confs = results.boxes.conf.cpu().tolist()

                for box, t_id, cls_idx, conf in zip(boxes, ids, clss, confs):
                    if box[1] < self.SKY_Y_LIMIT: continue 
                    
                    raw_label = self.model.names[cls_idx]
                    if raw_label in self.excluded_classes: continue

                    req_conf = self.get_dynamic_threshold(box[3], raw_label)
                    is_established = temp_id_hits.get(t_id, 0) > 3
                    
                    if raw_label == 'auto':
                        final_thresh = req_conf 
                    else:
                        final_thresh = req_conf * 0.80 if is_established else req_conf
                    
                    if conf < final_thresh: continue
                    
                    frame_data.append({'box': box, 'id': t_id, 'conf': round(conf, 3), 'lbl': raw_label})
                    temp_id_hits[t_id] = temp_id_hits.get(t_id, 0) + 1
            
            frame_cache[frame_idx] = frame_data
            frame_idx += 1
            if frame_idx % 20 == 0: print(f"   Analyzing frame {frame_idx}...", end='\r')
        
        cap.release()

        frame_cache = self.heal_broken_tracks(frame_cache)

        history_map = {} 
        conf_map = {}    
        
        for f in frame_cache:
            for det in frame_cache[f]:
                tid = det['id']
                if tid not in history_map: 
                    history_map[tid] = []
                    conf_map[tid] = []
                history_map[tid].append(det['lbl'])
                conf_map[tid].append(det['conf'])

        master_id_map = {}
        for t_id, history in history_map.items():
            counts = Counter(history)
            final_label = counts.most_common(1)[0][0]
            
            if final_label in ['motorbike', 'auto'] and 'car' in counts:
                if counts['car'] / len(history) > 0.20: final_label = 'car'
            
            if 'bus' in counts or 'truck' in counts:
                 heavy = counts.get('bus',0)+counts.get('truck',0)
                 if heavy/len(history)>0.3: final_label = 'bus' if counts.get('bus',0)>counts.get('truck',0) else 'truck'
            
            avg = sum(conf_map[t_id])/len(conf_map[t_id])
            master_id_map[t_id] = {'label': final_label, 'avg_conf': round(avg, 3)}

        # --- RENDER WITH VISUAL UPDATES ---
        print(f"\n[INFO] PASS 2: Rendering Final Output...")
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        live_counts = {name: 0 for name in ['car', 'auto', 'bus', 'truck', 'motorbike']}
        counted_ids = set()

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            detections = frame_cache.get(frame_idx, [])
            frame_log = {"frame": frame_idx, "objects": []}
            
            for det in detections:
                t_id = det['id']
                if t_id in master_id_map:
                    label = master_id_map[t_id]['label']
                    avg_conf = master_id_map[t_id]['avg_conf']
                    box = det['box']
                    current_conf = det['conf']
                    
                    if t_id not in counted_ids:
                        if label in live_counts: live_counts[label] += 1
                        counted_ids.add(t_id)
                    
                    frame_log["objects"].append({
                        "id": t_id, "label": label, "bbox": box.tolist(),
                        "current_conf": current_conf, "avg_conf": avg_conf
                    })

                    color = self.colors.get(label, (255, 255, 255))
                    # THICKER BOX (Thickness 3)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 3) 
                    
                    # BIGGER FONT (Scale 0.7, Thickness 2) and Simplified Label (Confidence only)
                    label_text = f"{t_id} {label} {current_conf:.2f}"
                    cv2.putText(frame, label_text, 
                               (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # --- LIVE COUNTER OVERLAY ---
            y_offset = 50
            cv2.rectangle(frame, (10, 10), (250, 230), (0,0,0), -1) # Dark background for readability
            cv2.putText(frame, "LIVE COUNTER", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            for i, (v_type, count) in enumerate(live_counts.items()):
                text = f"{v_type.upper()}: {count}"
                cv2.putText(frame, text, (20, 80 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            full_metadata["detections"].append(frame_log)
            out.write(frame)
            frame_idx += 1
        
        cap.release(); out.release()
        
        full_metadata["summary_counts"] = live_counts
        with open("FINAL_panoramic_detection.json", "w") as f:
            json.dump(full_metadata, f, indent=4)
            
        print(f"[INFO] Metadata saved.")
        return live_counts

if __name__ == "__main__":
    SOURCE = r"C:\IITM\Results\Final_SIFT_Shifted_1080p.mp4" 
    MODEL = r"C:\IITM\Vehicle_detection_360\runs\detect\FINAL_balanced_v11_6classes\weights\best.pt"
    
    detector = PanoramicDirectDetector(model_path=MODEL)
    results = detector.run_video_inference(SOURCE, SOURCE.replace(".mp4", "_DynamicDetect_new.mp4"))
    
    print("\n" + "="*60)
    print(" FINAL TRAFFIC REPORT ")
    print("="*60)
    for v_type, count in results.items():
        print(f"{v_type.upper():<15} : {count}")
    print("="*60)
'''
import os
import cv2
import numpy as np
import json
from ultralytics import YOLO
from collections import Counter

class PanoramicDirectDetector:
    def __init__(self, model_path):
        print(f"[INFO] Loading Model: {model_path}...")
        self.model = YOLO(model_path)
        self.colors = {
            'auto': (0, 165, 255), 'bus': (255, 0, 0), 'car': (0, 255, 0),
            'motorbike': (255, 255, 0), 'truck': (0, 0, 255)
        }
        self.excluded_classes = ['tractor', 'rickshaw', 'e-rickshaw', 'cart', 'person', 'cycle']
        
        self.CONF_FAR = 0.10        
        self.CONF_CLOSE = 0.30

    def setup_resolution(self, w, h):
        """Auto-scales the 1664p limits and polygon to match the video's actual resolution."""
        print(f"[INFO] Auto-scaling spatial masks to video resolution: {w}x{h}")
        ref_w, ref_h = 3328, 1664
        scale_x, scale_y = w / ref_w, h / ref_h

        self.SKY_Y_LIMIT = int(800 * scale_y)
        self.VEHICLE_Y_LIMIT = int(1150 * scale_y)
        self.VIP_Y_LIMIT = int(950 * scale_y)

        poly = [
            [0, 1021], [78, 1088], [341, 1165], [419, 1094], 
            [612, 1121], [679, 1206], [875, 1223], [921, 1290], 
            [1083, 1229], [1616, 1215], [1930, 1306], [2064, 1113], 
            [2206, 1065], [2340, 1111], [2807, 1021], [3267, 1009], 
            [3287, 1013], [3328, 1664], [0, 1664]
        ]
        self.EGO_POLYGON = np.array([[int(x * scale_x), int(y * scale_y)] for x, y in poly], np.int32)

    def get_dynamic_threshold(self, bbox_bottom_y, label):
        # 1. VIP PASS FOR HEAVY VEHICLES ON THE ROAD
        if label in ['bus', 'truck']:
            if bbox_bottom_y > self.VIP_Y_LIMIT:
                return 0.15  
            else:
                return 0.40  

        # 2. EXPONENTIAL DISTANCE CURVE
        y_clamped = max(self.SKY_Y_LIMIT, min(bbox_bottom_y, self.VEHICLE_Y_LIMIT))
        if self.VEHICLE_Y_LIMIT == self.SKY_Y_LIMIT:
            ratio = 1.0
        else:
            linear_ratio = (y_clamped - self.SKY_Y_LIMIT) / (self.VEHICLE_Y_LIMIT - self.SKY_Y_LIMIT)
            ratio = linear_ratio ** 2 
            
        base_thresh = self.CONF_FAR + (self.CONF_CLOSE - self.CONF_FAR) * ratio
        
        # 3. CLASS PUNISHMENT
        if label == 'auto': return max(base_thresh, 0.60) 
        if label == 'motorbike': return max(base_thresh, 0.20)
        return base_thresh

    def heal_broken_tracks(self, frame_cache, max_gap_frames=50, max_dist_pixels=150):
        print(f"[INFO] Running Track Healing (Merging broken IDs)...")
        tracks = {}
        sorted_frames = sorted(frame_cache.keys())
        
        for f_idx in sorted_frames:
            for det in frame_cache[f_idx]:
                tid = det['id']
                box = det['box']
                lbl = det['lbl'] 
                if tid not in tracks:
                    tracks[tid] = {
                        "start_f": f_idx, "end_f": f_idx, 
                        "start_box": box, "end_box": box, 
                        "class_votes": [lbl]
                    }
                else:
                    tracks[tid]["end_f"] = f_idx
                    tracks[tid]["end_box"] = box
                    tracks[tid]["class_votes"].append(lbl)

        sorted_ids = sorted(tracks.keys(), key=lambda x: tracks[x]['start_f'])
        remap_dict = {} 
        
        def get_center(box): return ((box[0]+box[2])//2, (box[1]+box[3])//2)
        def get_primary_class(votes): return Counter(votes).most_common(1)[0][0]

        for i in range(len(sorted_ids)):
            id_b = sorted_ids[i]
            if id_b in remap_dict: continue
            
            track_b = tracks[id_b]
            class_b = get_primary_class(track_b['class_votes'])
            center_b = get_center(track_b['start_box'])
            
            best_match = None
            min_dist = float('inf')
            
            for j in range(i-1, -1, -1):
                id_a = sorted_ids[j]
                while id_a in remap_dict: id_a = remap_dict[id_a]
                if id_a == id_b: continue
                
                track_a = tracks[id_a]
                gap = track_b['start_f'] - track_a['end_f']
                
                if gap <= 0 or gap > max_gap_frames: continue
                
                class_a = get_primary_class(track_a['class_votes'])
                if class_a != class_b:
                    is_heavy_a = class_a in ['bus', 'truck']
                    is_heavy_b = class_b in ['bus', 'truck']
                    if not (is_heavy_a and is_heavy_b):
                        continue 
                
                center_a = get_center(track_a['end_box'])
                dist = np.sqrt((center_a[0]-center_b[0])**2 + (center_a[1]-center_b[1])**2)
                
                if dist < max_dist_pixels and dist < min_dist:
                    min_dist = dist
                    best_match = id_a
            
            if best_match:
                remap_dict[id_b] = best_match
                tracks[best_match]['end_f'] = track_b['end_f']
                tracks[best_match]['end_box'] = track_b['end_box']
                tracks[best_match]['class_votes'].extend(track_b['class_votes'])

        print(f"[INFO] Healed {len(remap_dict)} broken tracks.")
        for f_idx in frame_cache:
            for det in frame_cache[f_idx]:
                if det['id'] in remap_dict:
                    parent = det['id']
                    while parent in remap_dict: parent = remap_dict[parent]
                    det['id'] = parent
        return frame_cache

    def run_video_inference(self, video_path, output_path):
        if not os.path.exists(video_path): return {}

        cap = cv2.VideoCapture(video_path)
        w, h = int(cap.get(3)), int(cap.get(4))
        fps = 20.0  
        
        # Setup resolution-specific limits
        self.setup_resolution(w, h)
        
        full_metadata = {
            "source_file": video_path,
            "resolution": f"{w}x{h}",
            "processed_fps": fps,
            "summary_counts": {},
            "detections": [] 
        }
        
        temp_id_hits = {} 
        frame_cache = {}      
        
        print(f"[INFO] PASS 1: Tracking & Spatial Filtering...")
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Run tracker at a strong scale to capture small details
            results = self.model.track(frame, 
                                       persist=True, 
                                       verbose=False, 
                                       conf=0.10, 
                                       iou=0.45, 
                                       imgsz=1920, 
                                       tracker="FINAL_custom_track.yaml")[0]
            
            temp_dets = []
            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                ids = results.boxes.id.int().cpu().tolist()
                clss = results.boxes.cls.int().cpu().tolist()
                confs = results.boxes.conf.cpu().tolist()

                for box, t_id, cls_idx, conf in zip(boxes, ids, clss, confs):
                    # 1. SKY FILTER
                    if box[3] < self.SKY_Y_LIMIT: continue 
                    
                    # 2. EGO-VEHICLE POLYGON MASK
                    bottom_center = (int((box[0] + box[2]) / 2), int(box[3]))
                    if cv2.pointPolygonTest(self.EGO_POLYGON, bottom_center, False) >= 0:
                        continue

                    raw_label = self.model.names[cls_idx]
                    if raw_label in self.excluded_classes: continue

                    # 3. DYNAMIC THRESHOLDING
                    req_conf = self.get_dynamic_threshold(box[3], raw_label)
                    is_established = temp_id_hits.get(t_id, 0) > 3
                    
                    if raw_label == 'auto':
                        final_thresh = req_conf 
                    else:
                        final_thresh = req_conf * 0.80 if is_established else req_conf
                    
                    if conf < final_thresh: continue
                    
                    temp_dets.append({'box': box, 'id': t_id, 'conf': round(conf, 3), 'lbl': raw_label})

            # 4. IoA SWALLOWED BOX REJECTION
            valid_dets = []
            for i in range(len(temp_dets)):
                is_swallowed = False
                det1 = temp_dets[i]
                box1, label1 = det1['box'], det1['lbl']
                
                for j in range(len(temp_dets)):
                    if i == j: continue
                    det2 = temp_dets[j]
                    box2, label2 = det2['box'], det2['lbl']
                    
                    if label2 in ['bus', 'truck'] and label1 not in ['bus', 'truck']:
                        x_left = max(box1[0], box2[0])
                        y_top = max(box1[1], box2[1])
                        x_right = min(box1[2], box2[2])
                        y_bottom = min(box1[3], box2[3])
                        
                        if x_right > x_left and y_bottom > y_top:
                            intersection_area = (x_right - x_left) * (y_bottom - y_top)
                            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                            ioa = intersection_area / box1_area
                            
                            if ioa > 0.85:
                                is_swallowed = True
                                break
                
                if not is_swallowed:
                    valid_dets.append(det1)
                    # Update tracking hit count only if it survives
                    temp_id_hits[det1['id']] = temp_id_hits.get(det1['id'], 0) + 1

            frame_cache[frame_idx] = valid_dets
            frame_idx += 1
            if frame_idx % 20 == 0: print(f"   Analyzing frame {frame_idx}...", end='\r')
        
        cap.release()
        print() # Newline after progress bar

        frame_cache = self.heal_broken_tracks(frame_cache)

        # --- TEMPORAL SMOOTHING ---
        history_map = {} 
        conf_map = {}    
        
        for f in frame_cache:
            for det in frame_cache[f]:
                tid = det['id']
                if tid not in history_map: 
                    history_map[tid] = []
                    conf_map[tid] = []
                history_map[tid].append(det['lbl'])
                conf_map[tid].append(det['conf'])

        master_id_map = {}
        for t_id, history in history_map.items():
            counts = Counter(history)
            final_label = counts.most_common(1)[0][0]
            
            if final_label in ['motorbike', 'auto'] and 'car' in counts:
                if counts['car'] / len(history) > 0.20: final_label = 'car'
            
            if 'bus' in counts or 'truck' in counts:
                 heavy = counts.get('bus',0)+counts.get('truck',0)
                 if heavy/len(history)>0.3: final_label = 'bus' if counts.get('bus',0)>counts.get('truck',0) else 'truck'
            
            avg = sum(conf_map[t_id])/len(conf_map[t_id])
            master_id_map[t_id] = {'label': final_label, 'avg_conf': round(avg, 3)}

        # --- RENDER WITH VISUAL UPDATES ---
        print(f"[INFO] PASS 2: Rendering Final Output...")
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        live_counts = {name: 0 for name in ['car', 'auto', 'bus', 'truck', 'motorbike']}
        counted_ids = set()

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            detections = frame_cache.get(frame_idx, [])
            frame_log = {"frame": frame_idx, "objects": []}
            
            for det in detections:
                t_id = det['id']
                if t_id in master_id_map:
                    label = master_id_map[t_id]['label']
                    avg_conf = master_id_map[t_id]['avg_conf']
                    box = det['box']
                    current_conf = det['conf']
                    
                    if t_id not in counted_ids:
                        if label in live_counts: live_counts[label] += 1
                        counted_ids.add(t_id)
                    
                    frame_log["objects"].append({
                        "id": t_id, "label": label, "bbox": box.tolist(),
                        "current_conf": current_conf, "avg_conf": avg_conf
                    })

                    color = self.colors.get(label, (255, 255, 255))
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 3) 
                    label_text = f"{t_id} {label} {current_conf:.2f}"
                    cv2.putText(frame, label_text, 
                               (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Draw the custom Ego-Vehicle exclusion zone for visualization
            cv2.polylines(frame, [self.EGO_POLYGON], isClosed=True, color=(0, 0, 255), thickness=2)

            # --- LIVE COUNTER OVERLAY ---
            cv2.rectangle(frame, (10, 10), (250, 230), (0,0,0), -1) 
            cv2.putText(frame, "LIVE COUNTER", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            for i, (v_type, count) in enumerate(live_counts.items()):
                text = f"{v_type.upper()}: {count}"
                cv2.putText(frame, text, (20, 80 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            full_metadata["detections"].append(frame_log)
            out.write(frame)
            frame_idx += 1
        
        cap.release(); out.release()
        
        full_metadata["summary_counts"] = live_counts
        with open("FINAL_panoramic_detection2.json", "w") as f:
            json.dump(full_metadata, f, indent=4)
            
        print(f"[INFO] Metadata saved.")
        return live_counts

if __name__ == "__main__":
    SOURCE = r"C:\IITM\Results\Final_Dynamic_Stitch_shortened.mp4" 
    MODEL = r"C:\IITM\Vehicle_detection_360\runs\detect\FINAL_balanced_v11_6classes\weights\best.pt"
    
    detector = PanoramicDirectDetector(model_path=MODEL)
    results = detector.run_video_inference(SOURCE, SOURCE.replace(".mp4", "_DynamicDetect_new.mp4"))
    
    print("\n" + "="*60)
    print(" FINAL TRAFFIC REPORT ")
    print("="*60)
    for v_type, count in results.items():
        print(f"{v_type.upper():<15} : {count}")
    print("="*60)