import os
import cv2
import numpy as np
import json
from ultralytics import YOLO
from collections import Counter, defaultdict

class PanoramicDirectDetector:
    def __init__(self, model_path):
        print(f"[INFO] Loading Model: {model_path}...")
        self.model = YOLO(model_path)
        self.colors = {'auto': (0, 165, 255), 'bus': (255, 0, 0), 'car': (0, 255, 0), 'motorbike': (255, 255, 0), 'truck': (0, 0, 255)}
        self.excluded_classes = ['tractor', 'rickshaw', 'e-rickshaw', 'cart', 'person', 'cycle']
        self.CONF_FAR = 0.10        
        self.CONF_CLOSE = 0.30

    def setup_resolution(self, w, h):
        scale_x, scale_y = w / 3328, h / 1664
        self.SKY_Y_LIMIT = int(800 * scale_y)
        self.VEHICLE_Y_LIMIT = int(1150 * scale_y)
        self.VIP_Y_LIMIT = int(950 * scale_y)
        
        # Note: If your video is shifted 25% to the left, you may also need to shift 
        # the X-coordinates of this EGO_POLYGON in the future to match the new hood position.
        poly = [[0, 1021], [78, 1088], [341, 1165], [419, 1094], [612, 1121], [679, 1206], [875, 1223], [921, 1290], [1083, 1229], [1616, 1215], [1930, 1306], [2064, 1113], [2206, 1065], [2340, 1111], [2807, 1021], [3267, 1009], [3287, 1013], [3328, 1664], [0, 1664]]
        self.EGO_POLYGON = np.array([[int(x * scale_x), int(y * scale_y)] for x, y in poly], np.int32)

    def get_dynamic_threshold(self, bbox_bottom_y, label):
        if label in ['bus', 'truck']: return 0.15 if bbox_bottom_y > self.VIP_Y_LIMIT else 0.40  
        y_clamped = max(self.SKY_Y_LIMIT, min(bbox_bottom_y, self.VEHICLE_Y_LIMIT))
        ratio = 1.0 if self.VEHICLE_Y_LIMIT == self.SKY_Y_LIMIT else ((y_clamped - self.SKY_Y_LIMIT) / (self.VEHICLE_Y_LIMIT - self.SKY_Y_LIMIT)) ** 2 
        base_thresh = self.CONF_FAR + (self.CONF_CLOSE - self.CONF_FAR) * ratio
        if label == 'auto': return max(base_thresh, 0.60) 
        if label == 'motorbike': return max(base_thresh, 0.20)
        return base_thresh

    def determine_movement_hybrid(self, prev_gray, curr_gray, bbox, frame_w, frame_h, history_boxes):
        x1, y1, x2, y2 = map(int, bbox)
        center_x = (x1 + x2) / 2
        
        # New Zone Mapping (Front centered at 25%, Back centered at 75%)
        is_front = (frame_w * 0.10 < center_x < frame_w * 0.40)
        is_back = (frame_w * 0.60 < center_x < frame_w * 0.90)

        if is_front or is_back:
            if len(history_boxes) < 15: return "UNKNOWN"
            past_box = history_boxes[-15]
            past_area = (past_box[2] - past_box[0]) * (past_box[3] - past_box[1])
            curr_area = (x2 - x1) * (y2 - y1)
            if past_area == 0: return "UNKNOWN"
            growth_rate = curr_area / past_area
            
            if 1.05 < growth_rate < 1.40: return "PARKED"
            elif growth_rate >= 1.40: return "MOVING (ONCOMING)"
            return "MOVING"
        else:
            # Sides are now in the middle (40% to 60%) and extreme edges (0-10% and 90-100%)
            veh_roi = prev_gray[max(0, y1+5):min(frame_h, y2-5), max(0, x1+5):min(frame_w, x2-5)]
            if veh_roi.size == 0: return "UNKNOWN"
            veh_pts = cv2.goodFeaturesToTrack(veh_roi, maxCorners=15, qualityLevel=0.2, minDistance=5)
            
            bg_y1, bg_y2, bg_x1, bg_x2 = min(y2 + 5, frame_h - 1), min(y2 + 60, frame_h), max(0, x1 - 20), min(frame_w, x2 + 20)
            bg_roi = prev_gray[bg_y1:bg_y2, bg_x1:bg_x2]
            if bg_roi.size == 0: return "UNKNOWN"
            bg_pts = cv2.goodFeaturesToTrack(bg_roi, maxCorners=15, qualityLevel=0.2, minDistance=5)
            
            if veh_pts is None or bg_pts is None: return "UNKNOWN"
            veh_pts += np.array([max(0, x1+5), max(0, y1+5)], dtype=np.float32)
            bg_pts += np.array([bg_x1, bg_y1], dtype=np.float32)

            veh_next, veh_st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, veh_pts, None)
            bg_next, bg_st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, bg_pts, None)
            if not (veh_st == 1).any() or not (bg_st == 1).any(): return "UNKNOWN"

            veh_flow = (veh_next[veh_st == 1] - veh_pts[veh_st == 1]).mean(axis=0)
            bg_flow = (bg_next[bg_st == 1] - bg_pts[bg_st == 1]).mean(axis=0)

            return "PARKED" if np.linalg.norm(veh_flow - bg_flow) < 1.5 else "MOVING"

    def heal_broken_tracks(self, frame_cache, max_gap_frames=50, max_dist_pixels=150):
        print(f"[INFO] Running Track Healing...")
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
                if dist < max_dist_pixels and dist < min_dist: min_dist, best_match = dist, id_a
            
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

    def run_video_inference(self, video_path, output_path):
        if not os.path.exists(video_path): return {}
        cap = cv2.VideoCapture(video_path)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        max_frames = int(2 * 60 * fps)
        self.setup_resolution(w, h)
        
        full_metadata = {"source_file": video_path, "resolution": f"{w}x{h}", "processed_fps": fps, "summary_counts": {}, "detections": []}
        temp_id_hits, frame_cache, box_history_map = {}, {}, defaultdict(list)
        prev_gray = None
        
        print(f"[INFO] PASS 1: Tracking, Spatial Filtering & Movement Computation...")
        frame_idx = 0
        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret: break
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = self.model.track(frame, persist=True, verbose=False, conf=0.25, iou=0.45, imgsz=1920, tracker="FINAL_custom_track.yaml")[0]
            temp_dets = []
            
            if results.boxes.id is not None:
                for box, t_id, cls_idx, conf in zip(results.boxes.xyxy.cpu().numpy().astype(int), results.boxes.id.int().cpu().tolist(), results.boxes.cls.int().cpu().tolist(), results.boxes.conf.cpu().tolist()):
                    if box[3] < self.SKY_Y_LIMIT or cv2.pointPolygonTest(self.EGO_POLYGON, (int((box[0] + box[2]) / 2), int(box[3])), False) >= 0: continue
                    raw_label = self.model.names[cls_idx]
                    if raw_label in self.excluded_classes: continue
                    req_conf = self.get_dynamic_threshold(box[3], raw_label)
                    final_thresh = req_conf if raw_label == 'auto' else (req_conf * 0.80 if temp_id_hits.get(t_id, 0) > 3 else req_conf)
                    if conf >= final_thresh: temp_dets.append({'box': box, 'id': t_id, 'conf': round(conf, 3), 'lbl': raw_label})

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
                    
                    box_history_map[det1['id']].append(det1['box'])
                    det1['status_vote'] = "UNKNOWN" if prev_gray is None else self.determine_movement_hybrid(prev_gray, curr_gray, det1['box'], w, h, box_history_map[det1['id']])

            frame_cache[frame_idx] = valid_dets
            prev_gray = curr_gray
            frame_idx += 1
            if frame_idx % 20 == 0: print(f"   Analyzing frame {frame_idx}/{max_frames}...", end='\r')
        
        cap.release()
        print() 
        frame_cache = self.heal_broken_tracks(frame_cache)

        history_map, conf_map, status_votes_map = defaultdict(list), defaultdict(list), defaultdict(list)
        for f in frame_cache:
            for det in frame_cache[f]:
                tid = det['id']
                history_map[tid].append(det['lbl'])
                conf_map[tid].append(det['conf'])
                if det.get('status_vote') and det['status_vote'] != "UNKNOWN": status_votes_map[tid].append(det['status_vote'])

        master_id_map = {}
        for t_id, history in history_map.items():
            counts = Counter(history)
            final_label = counts.most_common(1)[0][0]
            if final_label in ['motorbike', 'auto'] and counts.get('car', 0) / len(history) > 0.20: final_label = 'car'
            if 'bus' in counts or 'truck' in counts:
                 if (counts.get('bus',0)+counts.get('truck',0))/len(history)>0.3: final_label = 'bus' if counts.get('bus',0)>counts.get('truck',0) else 'truck'
            
            final_status, votes = "MOVING", status_votes_map.get(t_id, [])
            if votes:
                vote_counts = Counter(votes)
                if vote_counts.get('PARKED', 0) > vote_counts.get('MOVING', 0) and vote_counts.get('PARKED', 0) > vote_counts.get('MOVING (ONCOMING)', 0): final_status = "PARKED"
                elif vote_counts.get('MOVING (ONCOMING)', 0) > vote_counts.get('MOVING', 0): final_status = "MOVING (ONCOMING)"
            master_id_map[t_id] = {'label': final_label, 'avg_conf': round(sum(conf_map[t_id])/len(conf_map[t_id]), 3), 'status': final_status}

        print(f"[INFO] PASS 2: Rendering Fixed Output...")
        cap, out = cv2.VideoCapture(video_path), cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        live_counts, counted_ids, frame_idx = {name: 0 for name in ['car', 'auto', 'bus', 'truck', 'motorbike']}, set(), 0
        
        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret: break
            frame_log = {"frame": frame_idx, "objects": []}
            
            for det in frame_cache.get(frame_idx, []):
                t_id = det['id']
                if t_id in master_id_map:
                    label, avg_conf, final_status = master_id_map[t_id]['label'], master_id_map[t_id]['avg_conf'], master_id_map[t_id]['status']
                    if t_id not in counted_ids and label in live_counts:
                        live_counts[label] += 1
                        counted_ids.add(t_id)
                    frame_log["objects"].append({"id": t_id, "label": label, "status": final_status, "bbox": det['box'].tolist(), "avg_conf": avg_conf})
                    color = (0, 0, 255) if final_status == "PARKED" else self.colors.get(label, (255, 255, 255))
                    cv2.rectangle(frame, (det['box'][0], det['box'][1]), (det['box'][2], det['box'][3]), color, 3) 
                    cv2.putText(frame, f"{t_id} {label} ({final_status}) {det['conf']:.2f}", (det['box'][0], det['box'][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.polylines(frame, [self.EGO_POLYGON], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.rectangle(frame, (10, 10), (250, 230), (0,0,0), -1) 
            cv2.putText(frame, "LIVE COUNTER", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            for i, (v_type, count) in enumerate(live_counts.items()): cv2.putText(frame, f"{v_type.upper()}: {count}", (20, 80 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            full_metadata["detections"].append(frame_log)
            out.write(frame)
            frame_idx += 1
        
        cap.release(); out.release()
        with open("FINAL_CV_Hybrid_detection.json", "w") as f: json.dump(full_metadata, f, indent=4)
        print(f"[INFO] Metadata saved.")
        return live_counts

if __name__ == "__main__":
    SOURCE = r"C:\IITM\Results\Final_SIFT_Shifted_1080p.mp4" 
    MODEL = r"C:\IITM\Vehicle_detection_360\runs\detect\FINAL_balanced_v11_6classes\weights\best.pt"
    detector = PanoramicDirectDetector(model_path=MODEL)
    results = detector.run_video_inference(SOURCE, SOURCE.replace(".mp4", "_DynamicDetect_Parked.mp4"))