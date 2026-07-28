import os
import cv2
import numpy as np
import json
import zipfile
import shutil
from ultralytics import YOLO
from collections import Counter

class StreetViewDetector:
    def __init__(self, model_path):
        print(f"[INFO] Loading Model: {model_path}...")
        self.model = YOLO(model_path)
        self.colors = {
            'auto': (0, 165, 255), 'bus': (255, 0, 0), 'car': (0, 255, 0),
            'motorbike': (255, 255, 0), 'truck': (0, 0, 255)
        }
        self.excluded_classes = ['person', 'cycle', 'tractor', 'cart', 'rickshaw'] 
        
        # --- GEOMETRIC FILTERS ---
        self.SKY_LINE = 320  
        self.BOTTOM_MASK_HEIGHT = 120 

        # --- DISTANCE TUNING ---
        self.CONF_HORIZON = 0.15  
        self.CONF_CLOSE = 0.45    

    def get_dynamic_threshold(self, bbox_bottom_y, label):
        """Calculates threshold based on Distance."""
        valid_top = self.SKY_LINE
        valid_bottom = 1080 - self.BOTTOM_MASK_HEIGHT
        
        if bbox_bottom_y < (valid_top + 200): 
            return self.CONF_HORIZON
            
        ratio = (bbox_bottom_y - (valid_top + 200)) / (valid_bottom - (valid_top + 200))
        ratio = max(0, min(ratio, 1))
        
        return self.CONF_HORIZON + (ratio * (self.CONF_CLOSE - self.CONF_HORIZON))

    def heal_broken_tracks(self, frame_cache, max_gap_frames=50, max_dist_pixels=100):
        """Merges fragmented IDs."""
        tracks = {}
        for f_idx in sorted(frame_cache.keys()):
            for det in frame_cache[f_idx]:
                tid = det['id']
                if tid not in tracks:
                    tracks[tid] = {"start_f": f_idx, "end_f": f_idx, "start_box": det['box'], "end_box": det['box'], "votes": [det['lbl']]}
                else:
                    tracks[tid]["end_f"] = f_idx
                    tracks[tid]["end_box"] = det['box']
                    tracks[tid]["votes"].append(det['lbl'])

        sorted_ids = sorted(tracks.keys(), key=lambda x: tracks[x]['start_f'])
        remap = {}
        def get_center(box): return ((box[0]+box[2])//2, (box[1]+box[3])//2)
        def get_lbl(votes): return Counter(votes).most_common(1)[0][0]

        for i in range(len(sorted_ids)):
            id_b = sorted_ids[i]
            if id_b in remap: continue
            for j in range(i-1, -1, -1):
                id_a = sorted_ids[j]
                while id_a in remap: id_a = remap[id_a]
                if id_a == id_b: continue
                gap = tracks[id_b]['start_f'] - tracks[id_a]['end_f']
                if 0 < gap <= max_gap_frames:
                    cls_a = get_lbl(tracks[id_a]['votes'])
                    cls_b = get_lbl(tracks[id_b]['votes'])
                    if cls_a == cls_b or (cls_a in ['bus', 'truck'] and cls_b in ['bus', 'truck']):
                        dist = np.linalg.norm(np.array(get_center(tracks[id_a]['end_box'])) - np.array(get_center(tracks[id_b]['start_box'])))
                        if dist < max_dist_pixels:
                            remap[id_b] = id_a
                            tracks[id_a]['end_f'] = tracks[id_b]['end_f']
                            tracks[id_a]['end_box'] = tracks[id_b]['end_box']
                            tracks[id_a]['votes'].extend(tracks[id_b]['votes'])
                            break
        for f in frame_cache:
            for det in frame_cache[f]:
                while det['id'] in remap: det['id'] = remap[det['id']]
        return frame_cache

    def save_for_cvat(self, frame_cache, master_id_map, w, h, zip_path):
        temp_dir = "temp_cvat"
        os.makedirs(temp_dir, exist_ok=True)
        inv_map = {v: k for k, v in self.model.names.items()}
        for f_idx, detections in frame_cache.items():
            with open(os.path.join(temp_dir, f"{f_idx:06d}.txt"), "w") as f:
                for det in detections:
                    tid = det['id']
                    if tid in master_id_map:
                        lbl = master_id_map[tid]['label']
                        if lbl not in inv_map: continue
                        box = det['box']
                        xc, yc = ((box[0]+box[2])/2)/w, ((box[1]+box[3])/2)/h
                        bw, bh = (box[2]-box[0])/w, (box[3]-box[1])/h
                        f.write(f"{inv_map[lbl]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        with zipfile.ZipFile(zip_path, 'w') as z:
            for file in os.listdir(temp_dir): z.write(os.path.join(temp_dir, file), arcname=file)
        shutil.rmtree(temp_dir)

    def run_video_inference(self, video_path, output_path, lens_id):
        cap = cv2.VideoCapture(video_path)
        w, h = int(cap.get(3)), int(cap.get(4))
        
        # --- NEW FRAME LIMIT ---
        MAX_FRAMES = 3600
        
        print(f"[INFO] Lens {lens_id} - PASS 1: Tracking (Limit: {MAX_FRAMES} frames)...")
        frame_cache = {}
        frame_idx = 0
        
        while cap.isOpened():
            # Stop PASS 1 if limit reached
            if frame_idx >= MAX_FRAMES:
                print(f"\n[INFO] Reached limit of {MAX_FRAMES} frames.")
                break

            ret, frame = cap.read()
            if not ret: break
            
            results = self.model.track(frame, 
                                       persist=True, 
                                       verbose=False, 
                                       conf=0.15, 
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
                    lbl = self.model.names[cls_idx]
                    if lbl in self.excluded_classes: continue
                    
                    if box[3] < self.SKY_LINE: continue 
                    if box[3] > (h - self.BOTTOM_MASK_HEIGHT): continue
                    
                    thresh = self.get_dynamic_threshold(box[3], lbl)
                    if conf >= thresh:
                        frame_data.append({'box': box, 'id': t_id, 'conf': conf, 'lbl': lbl})
            
            frame_cache[frame_idx] = frame_data
            frame_idx += 1
            if frame_idx % 25 == 0: print(f"  Processed {frame_idx}/{MAX_FRAMES} frames...", end='\r')

        cap.release()
        total_processed = frame_idx # Store final count
        
        frame_cache = self.heal_broken_tracks(frame_cache)

        # Voting & Stability
        master_id_map = {}
        id_life_stats = {} 
        for f in frame_cache:
            for det in frame_cache[f]:
                tid = det['id']
                if tid not in id_life_stats: id_life_stats[tid] = {'votes': [], 'confs': [], 'frames': 0}
                id_life_stats[tid]['votes'].append(det['lbl'])
                id_life_stats[tid]['confs'].append(det['conf'])
                id_life_stats[tid]['frames'] += 1

        for tid, stats in id_life_stats.items():
            if stats['frames'] < 3 and np.mean(stats['confs']) < 0.6: continue
            
            counts = Counter(stats['votes'])
            final_lbl = counts.most_common(1)[0][0]
            if final_lbl in ['auto', 'motorbike'] and counts['car'] > 0:
                 if (counts['car'] / len(stats['votes'])) > 0.3: final_lbl = 'car'
            
            # --- FIX: Save the average confidence for the track ---
            master_id_map[tid] = {
                'label': final_lbl,
                'avg_conf': float(np.mean(stats['confs']))
            }

        print(f"\n[INFO] Lens {lens_id} - PASS 2: Rendering {total_processed} frames...")
        cap = cv2.VideoCapture(video_path)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))
        live_counts, counted_ids = {}, set()
        full_meta = {"lens": lens_id, "counts": {}, "detections": []}
        
        # Render only what was processed in PASS 1
        for f_idx in range(total_processed):
            ret, frame = cap.read()
            if not ret: break
            
            frame_objs = []
            cv2.line(frame, (0, self.SKY_LINE), (w, self.SKY_LINE), (0, 0, 255), 1)
            cv2.line(frame, (0, h - self.BOTTOM_MASK_HEIGHT), (w, h - self.BOTTOM_MASK_HEIGHT), (0, 0, 255), 1)
            
            for det in frame_cache.get(f_idx, []):
                tid = det['id']
                if tid in master_id_map:
                    lbl = master_id_map[tid]['label']
                    if tid not in counted_ids:
                        live_counts[lbl] = live_counts.get(lbl, 0) + 1
                        counted_ids.add(tid)
                    
                    box = det['box']
                    color = self.colors.get(lbl, (255, 255, 255))
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(frame, f"{tid} {lbl}", (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # --- FIX: Append both the current score and average track score to JSON ---
                    frame_objs.append({
                        "id": tid, 
                        "label": lbl, 
                        "box": box.tolist(),
                        "current_conf": round(float(det['conf']), 3),
                        "avg_conf": round(master_id_map[tid]['avg_conf'], 3)
                    })
            
            full_meta["detections"].append({"frame": f_idx, "objects": frame_objs})
            out.write(frame)
            if f_idx % 50 == 0: print(f"  Rendering {f_idx}/{total_processed}...", end='\r')
            
        cap.release(); out.release()
        full_meta["counts"] = live_counts
        with open(output_path.replace(".mp4", ".json"), "w") as f: json.dump(full_meta, f, indent=4)
        self.save_for_cvat(frame_cache, master_id_map, w, h, output_path.replace(".mp4", "_cvat.zip"))
        print(f"\nLens {lens_id} Counts: {json.dumps(live_counts, indent=2)}")
        return live_counts

if __name__ == "__main__":
    BASE_DIR = r"C:\IITM\Results"
    MODEL = r"C:\IITM\Vehicle_detection_360\runs\detect\FINAL_balanced_v11_6classes\weights\best.pt"
    detector = StreetViewDetector(MODEL)

    for i in range(1, 7):
        lens_folder = os.path.join(BASE_DIR, f"lens {i}")
        video_file = os.path.join(lens_folder, f"1080p_lens{i}.mp4")
        if os.path.exists(video_file):
            print(f"\n" + "="*50 + f"\nPROCESSING LENS {i}\n" + "="*50)
            output = os.path.join(lens_folder, f"final_lens{i}_detected.mp4")
            detector.run_video_inference(video_file, output, lens_id=i)
        else:
            print(f"[SKIP] Video not found for Lens {i}")

'''

import os
import cv2
import numpy as np
import json
import zipfile
import shutil
from ultralytics import YOLO
from collections import Counter

class StreetViewDetector:
    def __init__(self, model_path):
        print(f"[INFO] Loading Model: {model_path}...")
        self.model = YOLO(model_path)
        self.colors = {
            'auto': (0, 165, 255), 'bus': (255, 0, 0), 'car': (0, 255, 0),
            'motorbike': (255, 255, 0), 'truck': (0, 0, 255)
        }
        self.excluded_classes = ['tractor', 'rickshaw', 'e-rickshaw', 'cart', 'person', 'cycle']
        
        # --- CONFIGURATION FROM PANORAMIC CODE ---
        self.SKY_Y_LIMIT = 350      
        self.VEHICLE_Y_LIMIT = 960  
        self.CONF_FAR = 0.25
        self.CONF_CLOSE = 0.65

    def get_dynamic_threshold(self, bbox_bottom_y, label):
        """Calculates threshold based on Distance AND Class."""
        y_clamped = max(self.SKY_Y_LIMIT, min(bbox_bottom_y, self.VEHICLE_Y_LIMIT))
        ratio = (y_clamped - self.SKY_Y_LIMIT) / (self.VEHICLE_Y_LIMIT - self.SKY_Y_LIMIT)
        base_thresh = self.CONF_FAR + (self.CONF_CLOSE - self.CONF_FAR) * ratio
        
        if label == 'auto': return max(base_thresh, 0.55) 
        if label == 'motorbike': return max(base_thresh, 0.45)
        return base_thresh

    def heal_broken_tracks(self, frame_cache, max_gap_frames=50, max_dist_pixels=150):
        """Merges fragmented IDs that belong to the same vehicle."""
        tracks = {}
        for f_idx in sorted(frame_cache.keys()):
            for det in frame_cache[f_idx]:
                tid = det['id']
                if tid not in tracks:
                    tracks[tid] = {"start_f": f_idx, "end_f": f_idx, "start_box": det['box'], "end_box": det['box'], "class_votes": [det['lbl']]}
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
            track_b = tracks[id_b]
            class_b = get_primary_class(track_b['class_votes'])
            
            for j in range(i-1, -1, -1):
                id_a = sorted_ids[j]
                while id_a in remap_dict: id_a = remap_dict[id_a]
                if id_a == id_b: continue
                track_a = tracks[id_a]
                gap = track_b['start_f'] - track_a['end_f']
                if 0 < gap <= max_gap_frames:
                    class_a = get_primary_class(track_a['class_votes'])
                    # Class matching logic from Panoramic code
                    if class_a == class_b or (class_a in ['bus', 'truck'] and class_b in ['bus', 'truck']):
                        dist = np.linalg.norm(np.array(get_center(track_a['end_box'])) - np.array(get_center(track_b['start_box'])))
                        if dist < max_dist_pixels:
                            remap_dict[id_b] = id_a
                            tracks[id_a].update({"end_f": track_b['end_f'], "end_box": track_b['end_box']})
                            tracks[id_a]["class_votes"].extend(track_b['class_votes'])
                            break

        for f_idx in frame_cache:
            for det in frame_cache[f_idx]:
                while det['id'] in remap_dict: det['id'] = remap_dict[det['id']]
        return frame_cache

    def save_for_cvat(self, frame_cache, master_id_map, w, h, zip_path):
        """Generates YOLO 1.1 ZIP for CVAT."""
        temp_dir = "temp_cvat"
        os.makedirs(temp_dir, exist_ok=True)
        inv_map = {v: k for k, v in self.model.names.items()}
        
        for f_idx, detections in frame_cache.items():
            with open(os.path.join(temp_dir, f"{f_idx:06d}.txt"), "w") as f:
                for det in detections:
                    tid = det['id']
                    if tid in master_id_map:
                        lbl = master_id_map[tid]['label']
                        box = det['box']
                        xc, yc = ((box[0]+box[2])/2)/w, ((box[1]+box[3])/2)/h
                        bw, bh = (box[2]-box[0])/w, (box[3]-box[1])/h
                        f.write(f"{inv_map[lbl]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        
        # Zip and cleanup
        with zipfile.ZipFile(zip_path, 'w') as z:
            for file in os.listdir(temp_dir): z.write(os.path.join(temp_dir, file), arcname=file)
        shutil.rmtree(temp_dir)

    def run_video_inference(self, video_path, output_path, lens_id):
        cap = cv2.VideoCapture(video_path)
        w, h = int(cap.get(3)), int(cap.get(4))
        fps = 20.0
        
        temp_id_hits, frame_cache, frame_idx = {}, {}, 0
        print(f"[INFO] Lens {lens_id} - PASS 1: Tracking...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            results = self.model.track(frame, 
                           persist=True, 
                           verbose=False, 
                           conf=0.20,     # Forces BoT-SORT to see horizon cars
                           iou=0.45,      # Helps separate crowded autos/bikes
                           imgsz=640, 
                           tracker="custom_track.yaml")[0]
            
            frame_data = []
            if results.boxes.id is not None:
                for box, t_id, cls_idx, conf in zip(results.boxes.xyxy.cpu().numpy().astype(int), results.boxes.id.int().cpu().tolist(), results.boxes.cls.int().cpu().tolist(), results.boxes.conf.cpu().tolist()):
                    if box[1] < self.SKY_Y_LIMIT: continue 
                    label = self.model.names[cls_idx]
                    if label in self.excluded_classes: continue
                    
                    req_conf = self.get_dynamic_threshold(box[3], label)
                    is_est = temp_id_hits.get(t_id, 0) > 3
                    final_thresh = req_conf if label == 'auto' else (req_conf * 0.8 if is_est else req_conf)
                    
                    if conf >= final_thresh:
                        frame_data.append({'box': box, 'id': t_id, 'conf': round(conf, 3), 'lbl': label})
                        temp_id_hits[t_id] = temp_id_hits.get(t_id, 0) + 1
            
            frame_cache[frame_idx] = frame_data
            frame_idx += 1
            if frame_idx % 50 == 0: print(f"  Frame {frame_idx}...", end='\r')
        
        cap.release()
        frame_cache = self.heal_broken_tracks(frame_cache)

        # Voting Logic
        master_id_map = {}
        for f in frame_cache:
            for det in frame_cache[f]:
                tid = det['id']
                if tid not in master_id_map: master_id_map[tid] = {'votes': [], 'confs': []}
                master_id_map[tid]['votes'].append(det['lbl'])
                master_id_map[tid]['confs'].append(det['conf'])

        for tid in master_id_map:
            counts = Counter(master_id_map[tid]['votes'])
            final_lbl = counts.most_common(1)[0][0]
            # Car & Heavy Bias Logic
            if final_lbl in ['motorbike', 'auto'] and 'car' in counts and (counts['car']/len(master_id_map[tid]['votes']) > 0.20): final_lbl = 'car'
            if 'bus' in counts or 'truck' in counts:
                 if (counts.get('bus',0)+counts.get('truck',0))/len(master_id_map[tid]['votes']) > 0.3:
                     final_lbl = 'bus' if counts.get('bus',0) > counts.get('truck',0) else 'truck'
            master_id_map[tid] = {'label': final_lbl, 'avg_conf': np.mean(master_id_map[tid]['confs'])}

        # PASS 2: RENDER
        print(f"\n[INFO] Lens {lens_id} - PASS 2: Rendering...")
        cap = cv2.VideoCapture(video_path)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        live_counts, counted_ids = {n: 0 for n in self.colors.keys()}, set()
        
        full_meta = {"lens": lens_id, "counts": {}, "detections": []}
        
        for f_idx in range(frame_idx):
            ret, frame = cap.read()
            if not ret: break
            frame_objects = []
            for det in frame_cache.get(f_idx, []):
                tid = det['id']
                if tid in master_id_map:
                    lbl = master_id_map[tid]['label']
                    if tid not in counted_ids:
                        live_counts[lbl] = live_counts.get(lbl, 0) + 1
                        counted_ids.add(tid)
                    
                    box = det['box']
                    color = self.colors.get(lbl, (255, 255, 255))
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(frame, f"{tid} {lbl}", (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    frame_objects.append({"id": tid, "label": lbl, "box": box.tolist()})
            
            full_meta["detections"].append({"frame": f_idx, "objects": frame_objects})
            out.write(frame)

        cap.release(); out.release()
        full_meta["counts"] = live_counts
        
        # Save JSON & CVAT
        with open(output_path.replace(".mp4", ".json"), "w") as f: json.dump(full_meta, f, indent=4)
        self.save_for_cvat(frame_cache, master_id_map, w, h, output_path.replace(".mp4", "_cvat.zip"))
        
        return live_counts

if __name__ == "__main__":
    BASE_DIR = r"C:\IITM\Results"
    MODEL = r"C:\IITM\Vehicle_detection_360\runs\detect\balanced_v11_6classes\weights\best.pt"
    detector = StreetViewDetector(MODEL)

    for i in range(1, 7):
        lens_folder = os.path.join(BASE_DIR, f"lens {i}")
        video_file = os.path.join(lens_folder, f"1080p_lens{i}.mp4") # Adjust filename if necessary
        
        if os.path.exists(video_file):
            print(f"\n" + "="*50 + f"\nPROCESSING LENS {i}\n" + "="*50)
            output = os.path.join(lens_folder, f"final_lens{i}_detected.mp4")
            detector.run_video_inference(video_file, output, lens_id=i)
        else:
            print(f"[SKIP] Video not found for Lens {i}")
            '''