import os
import cv2
import numpy as np
import json
import zipfile
import shutil
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
        """Calculates threshold based on Distance AND Class."""
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
                if tid not in tracks:
                    tracks[tid] = {
                        "start_f": f_idx, "end_f": f_idx, 
                        "start_box": det['box'], "end_box": det['box'], 
                        "class_votes": [det['lbl']]
                    }
                else:
                    tracks[tid]["end_f"] = f_idx
                    tracks[tid]["end_box"] = det['box']
                    tracks[tid]["class_votes"].append(det['lbl'])

        sorted_ids = sorted(tracks.keys(), key=lambda x: tracks[x]['start_f'])
        remap_dict = {} 
        
        def get_center(box): return ((box[0]+box[2])//2, (box[1]+box[3])//2)
        def get_primary_class(votes): return Counter(votes).most_common(1)[0][0]

        for i in range(len(sorted_ids)):
            id_b = sorted_ids[i]
            if id_b in remap_dict: continue
            track_b = tracks[id_b]; class_b = get_primary_class(track_b['class_votes']); center_b = get_center(track_b['start_box'])
            
            for j in range(i-1, -1, -1):
                id_a = sorted_ids[j]
                while id_a in remap_dict: id_a = remap_dict[id_a]
                if id_a == id_b: continue
                track_a = tracks[id_a]; gap = track_b['start_f'] - track_a['end_f']
                if gap <= 0 or gap > max_gap_frames: continue
                
                class_a = get_primary_class(track_a['class_votes'])
                if class_a != class_b and not (class_a in ['bus', 'truck'] and class_b in ['bus', 'truck']): continue
                
                center_a = get_center(track_a['end_box'])
                dist = np.sqrt((center_a[0]-center_b[0])**2 + (center_a[1]-center_b[1])**2)
                if dist < max_dist_pixels:
                    remap_dict[id_b] = id_a
                    tracks[id_a]['end_f'] = track_b['end_f']
                    tracks[id_a]['end_box'] = track_b['end_box']
                    tracks[id_a]['class_votes'].extend(track_b['class_votes'])
                    break

        for f_idx in frame_cache:
            for det in frame_cache[f_idx]:
                if det['id'] in remap_dict:
                    parent = det['id']
                    while parent in remap_dict: parent = remap_dict[parent]
                    det['id'] = parent
        return frame_cache

    def save_for_cvat(self, frame_cache, master_id_map, img_w, img_h, output_zip="cvat_prelabels.zip"):
        print(f"[INFO] Exporting to CVAT ZIP (YOLO 1.1 format)...")
        temp_dir = "obj_train_data"
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        inv_map = {v: k for k, v in self.model.names.items()}
        all_frames = sorted(frame_cache.keys())

        for f_idx in all_frames:
            with open(os.path.join(temp_dir, f"{f_idx:06d}.txt"), "w") as f:
                for det in frame_cache[f_idx]:
                    t_id = det['id']
                    if t_id in master_id_map:
                        label_name = master_id_map[t_id]['label']
                        cls_id = inv_map[label_name]
                        box = det['box']
                        x_c = ((box[0] + box[2]) / 2) / img_w
                        y_c = ((box[1] + box[3]) / 2) / img_h
                        bw = (box[2] - box[0]) / img_w
                        bh = (box[3] - box[1]) / img_h
                        f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

        with open("obj.names", "w") as f:
            for i in range(len(self.model.names)): f.write(f"{self.model.names[i]}\n")
        with open("obj.data", "w") as f:
            f.write(f"classes = {len(self.model.names)}\ntrain = train.txt\nnames = obj.names\n")
        with open("train.txt", "w") as f:
            for f_idx in all_frames: f.write(f"data/obj_train_data/{f_idx:06d}.txt\n")

        with zipfile.ZipFile(output_zip, 'w') as z:
            z.write("obj.names"); z.write("obj.data"); z.write("train.txt")
            for file in os.listdir(temp_dir): z.write(os.path.join(temp_dir, file), os.path.join("data", temp_dir, file))

        shutil.rmtree(temp_dir); os.remove("obj.names"); os.remove("obj.data"); os.remove("train.txt")
        print(f"[SUCCESS] CVAT Package saved as {output_zip}")

    def run_video_inference(self, video_path, output_path):
        if not os.path.exists(video_path): return {}

        cap = cv2.VideoCapture(video_path)
        w, h = int(cap.get(3)), int(cap.get(4))
        fps = 20.0  
        
        full_metadata = {
            "source_file": video_path, "resolution": f"{w}x{h}",
            "processed_fps": fps, "summary_counts": {}, "detections": [] 
        }
        
        temp_id_hits, frame_cache, frame_idx = {}, {}, 0
        
        print(f"[INFO] PASS 1: Analysis & Tracking...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            results = self.model.track(frame, 
                           persist=True, 
                           verbose=False, 
                           conf=0.20,     # Forces BoT-SORT to see horizon cars
                           iou=0.45,      # Helps separate crowded autos/bikes
                           imgsz=1920, 
                           tracker="custom_track.yaml")[0]
            
            frame_data = []
            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                ids = results.boxes.id.int().cpu().tolist()
                clss = results.boxes.cls.int().cpu().tolist()
                confs = results.boxes.conf.cpu().tolist()

                for box, t_id, cls_idx, conf in zip(boxes, ids, clss, confs):
                    if box[1] < self.SKY_Y_LIMIT: continue 
                    label = self.model.names[cls_idx]
                    if label in self.excluded_classes: continue

                    req_conf = self.get_dynamic_threshold(box[3], label)
                    is_established = temp_id_hits.get(t_id, 0) > 3
                    final_thresh = req_conf if label == 'auto' else (req_conf * 0.80 if is_established else req_conf)
                    
                    if conf >= final_thresh:
                        frame_data.append({'box': box, 'id': t_id, 'conf': round(conf, 3), 'lbl': label})
                        temp_id_hits[t_id] = temp_id_hits.get(t_id, 0) + 1
            
            frame_cache[frame_idx] = frame_data
            frame_idx += 1
            if frame_idx % 20 == 0: print(f"   Analyzing frame {frame_idx}...", end='\r')
        
        cap.release()
        frame_cache = self.heal_broken_tracks(frame_cache)

        # --- VOTING ---
        history_map, conf_map = {}, {}
        for f in frame_cache:
            for det in frame_cache[f]:
                tid = det['id']
                if tid not in history_map: history_map[tid] = []; conf_map[tid] = []
                history_map[tid].append(det['lbl']); conf_map[tid].append(det['conf'])

        master_id_map = {}
        for t_id, history in history_map.items():
            counts = Counter(history)
            final_label = counts.most_common(1)[0][0]
            if final_label in ['motorbike', 'auto'] and 'car' in counts and (counts['car']/len(history) > 0.20): final_label = 'car'
            if 'bus' in counts or 'truck' in counts:
                 heavy = counts.get('bus',0)+counts.get('truck',0)
                 if heavy/len(history)>0.3: final_label = 'bus' if counts.get('bus',0)>counts.get('truck',0) else 'truck'
            
            master_id_map[t_id] = {'label': final_label, 'avg_conf': round(np.mean(conf_map[t_id]), 3)}

        # --- RENDER PASS ---
        print(f"\n[INFO] PASS 2: Rendering & Logging...")
        cap = cv2.VideoCapture(video_path)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        live_counts, counted_ids, frame_idx = {n: 0 for n in ['car', 'auto', 'bus', 'truck', 'motorbike']}, set(), 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            detections = frame_cache.get(frame_idx, [])
            frame_log = {"frame": frame_idx, "objects": []}
            
            for det in detections:
                t_id = det['id']
                if t_id in master_id_map:
                    label = master_id_map[t_id]['label']
                    box, avg_conf, current_conf = det['box'], master_id_map[t_id]['avg_conf'], det['conf']
                    
                    if t_id not in counted_ids:
                        if label in live_counts: live_counts[label] += 1
                        counted_ids.add(t_id)
                    
                    frame_log["objects"].append({"id": t_id, "label": label, "bbox": box.tolist(), "current_conf": current_conf, "avg_conf": avg_conf})
                    color = self.colors.get(label, (255, 255, 255))
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 1) 
                    req_c = self.get_dynamic_threshold(box[3], label)
                    cv2.putText(frame, f"{t_id} {label} {current_conf:.2f}/{req_c:.2f}", (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            full_metadata["detections"].append(frame_log)
            out.write(frame); frame_idx += 1
        
        cap.release(); out.release()
        full_metadata["summary_counts"] = live_counts
        with open("panoramic_detection.json", "w") as f: json.dump(full_metadata, f, indent=4)
        
        # --- CVAT EXPORT ---
        self.save_for_cvat(frame_cache, master_id_map, w, h)
        return live_counts

if __name__ == "__main__":
    SOURCE = r"C:\IITM\Results\Final_SIFT_Shifted_1080p.mp4" 
    MODEL = r"C:\IITM\Vehicle_detection_360\runs\detect\balanced_v11_6classes\weights\best.pt"
    detector = PanoramicDirectDetector(MODEL)
    results = detector.run_video_inference(SOURCE, SOURCE.replace(".mp4", "_DynamicDetect2.mp4"))
    
    print("\n" + "="*60 + "\n FINAL TRAFFIC REPORT \n" + "="*60)
    for v_type, count in results.items(): print(f"{v_type.upper():<15} : {count}")
    print("="*60)