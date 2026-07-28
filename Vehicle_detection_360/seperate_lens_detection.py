import os
import cv2
import numpy as np
import json
import datetime
from ultralytics import YOLO

class StreetViewDetector:
    def __init__(self, model_path='yolov8n.pt', intrinsics_path=None):
        # Load model once
        self.model = YOLO(model_path)
        self.fov = 90
        self.window_size = (640, 640)
        
        # Load Intrinsics
        if intrinsics_path and os.path.exists(intrinsics_path):
            with open(intrinsics_path, "r") as f:
                data = json.load(f)
            self.K = np.asarray(data["K"], dtype=np.float64)
            self.D = np.asarray(data.get("D", [0,0,0,0,0]), dtype=np.float64)
        else:
            self.K = np.eye(3)
            self.D = np.zeros(5)

    def _is_detection_valid(self, conf, box_h, img_h, cls):
        # Use a dictionary for O(1) lookup speed
        THRESHOLDS = {'car': 0.50, 'bus': 0.50, 'truck': 0.50, 'bicycle': 0.40, 'motorcycle': 0.40}
        base_thresh = THRESHOLDS.get(cls, 0.45)
        
        # Distance-based scaling
        if box_h / img_h < 0.05:
            base_thresh *= 0.5
        return conf >= max(0.20, base_thresh)

    def run_video_inference(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        # CHANGE 1: Store Dictionary of {ID: Max_Confidence} instead of just Set of IDs
        # This allows us to calculate the average confidence of unique vehicles later
        unique_objects = {
            'car': {}, 'bus': {}, 'truck': {}, 
            'motorcycle': {}, 'auto_rickshaw': {}
        }

        # Use .track() with persist=True
        results_generator = self.model.track(source=video_path, stream=True, persist=True, verbose=False, conf=0.20, tracker="bytetrack.yaml")

        for frame_idx, result in enumerate(results_generator):
            frame = result.orig_img 

            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().tolist()
                clss = result.boxes.cls.int().cpu().tolist()

                for box, track_id, conf, cls in zip(boxes, track_ids, confs, clss):
                    label = self.model.names[cls]
                    
                    if self._is_detection_valid(conf, abs(box[3]-box[1]), h, label):
                        if label in unique_objects:
                            # Update confidence if this detection is better than previous ones for this ID
                            current_best = unique_objects[label].get(track_id, 0.0)
                            if conf > current_best:
                                unique_objects[label][track_id] = conf
                        
                        # Draw Box & ID & Conf
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                        label_text = f"ID:{track_id} {label} {conf:.2f}"
                        cv2.putText(frame, label_text, (box[0], box[1]-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            out.write(frame)
            if frame_idx % 100 == 0:
                print(f"  Progress: {frame_idx}/{total_frames} frames processed...", end="\r")
        
        cap.release()
        out.release()
        
        # CHANGE 2: Calculate Counts AND Averages
        summary = {
            "counts": {},
            "avg_confidence": {},
            "metadata": {
                "source_file": os.path.basename(__file__),
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "resolution": f"{w}x{h}",
                "fps": round(fps, 2),
                "total_frames": total_frames
            }
        }

        for label, id_map in unique_objects.items():
            count = len(id_map)
            summary["counts"][label] = count
            
            if count > 0:
                avg_conf = sum(id_map.values()) / count
                summary["avg_confidence"][label] = round(avg_conf, 4)
            else:
                summary["avg_confidence"][label] = 0.0

        return summary

    # =========================================================================
    # TASK 2 LOGIC (Preserved)
    # =========================================================================
    # NOTE: The following methods are preserved for Task 2.
    # They are NOT used in the current Task 1 (Individual Lens Detection) because:
    # 1. We are processing lenses independently to get raw sensor counts.
    # 2. We are working with pre-undistorted 2D video, so we don't need to project 
    #    to spherical coordinates yet.
    # 3. De-duplication across overlapping FOVs will be handled in the next phase
    #    by mapping these detections to a shared Global 360-degree Map.
    # =========================================================================
    '''
    def _pixel_to_spherical(self, px, py, u_c, v_c):
        nx = (px / self.window_size[0] - 0.5) * 2.0
        ny = (py / self.window_size[1] - 0.5) * 2.0
        fov_rad = np.radians(self.fov)
        f = 1.0 / np.tan(fov_rad / 2.0)
        yaw_rad = np.radians(u_c) + np.arctan(nx / f)
        pitch_rad = np.radians(v_c) + np.arctan(-ny / np.sqrt(nx*nx + f*f))
        return np.degrees(yaw_rad), np.degrees(pitch_rad)

    def _spherical_to_equi_pixel(self, yaw, pitch, w, h):
        ex = int(((yaw + 180) / 360) * w) % w
        ey = int(((90 - pitch) / 180) * h)
        return ex, ey

    def _get_global_bbox(self, x1, y1, x2, y2, u_deg, v_deg, w, h):
        yaw1, pitch1 = self._pixel_to_spherical(x1, y1, u_deg, v_deg)
        yaw2, pitch2 = self._pixel_to_spherical(x2, y2, u_deg, v_deg)
        gx1, gy1 = self._spherical_to_equi_pixel(yaw1, pitch1, w, h)
        gx2, gy2 = self._spherical_to_equi_pixel(yaw2, pitch2, w, h)
        if gx1 > gx2: gx2 += w
        return [gx1, gy1, gx2, gy2]

    def _get_robust_center(self, box, img_w):
        gx1, gy1, gx2, gy2 = box
        center_x = (gx1 + gx2 + img_w) / 2 % img_w if gx1 > gx2 else (gx1 + gx2) / 2
        center_y = (gy1 + gy2) / 2
        return center_x, center_y

    def _calculate_iou(self, boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0

    def _merge_boxes(self, boxA, confA, boxB, confB):
        total_conf = confA + confB
        return [(boxA[i] * confA + boxB[i] * confB) / total_conf for i in range(4)]

    def _hybrid_nms(self, detections, img_w, iou_threshold=0.5):
        if not detections: return []
        VEHICLE_GROUP = {'car', 'truck', 'bus', 'train', 'motorcycle', 'auto_rickshaw'}
        dets = sorted(detections, key=lambda x: x['conf'], reverse=True)
        keep = []
        while dets:
            best = dets.pop(0)
            m_box, m_conf, m_count = best['global_box'], best['conf'], best.get('merge_count', 1)
            b_cx, b_cy = self._get_robust_center(m_box, img_w)
            bgx1, _, bgx2, _ = m_box
            b_width = (bgx2 + img_w) - bgx1 if bgx1 > bgx2 else bgx2 - bgx1
            d_thresh = max(5.0, b_width * 0.5)

            remaining = []
            for d in dets:
                should_merge = False
                is_veh = best['class'] in VEHICLE_GROUP and d['class'] in VEHICLE_GROUP
                if best['class'] == d['class'] or is_veh:
                    if best['class'] == 'person':
                        d_cx, d_cy = self._get_robust_center(d['global_box'], img_w)
                        dx = abs(b_cx - d_cx)
                        if dx > img_w / 2: dx = img_w - dx
                        if np.sqrt(dx**2 + abs(b_cy - d_cy)**2) < d_thresh: should_merge = True
                    elif self._calculate_iou(m_box, d['global_box']) > iou_threshold:
                        should_merge = True
                
                if should_merge:
                    m_box = self._merge_boxes(m_box, m_conf, d['global_box'], d['conf'])
                    m_conf += d['conf']
                    m_count += 1
                else:
                    remaining.append(d)
            best['global_box'], best['merge_count'] = m_box, m_count
            keep.append(best)
            dets = remaining
        return keep'''

if __name__ == "__main__":
    BASE_DIR = r"C:\IITM\CAMERA_Short_Clips"
    INTRINSICS_DIR = r"C:\IITM\Vehicle_detection_360\Intrinsics"
    MODEL_PATH = 'yolov8n.pt'
    
    final_report = {}

    for i in range(1, 7):
        lens_folder = f"lens {i}"
        lens_id = lens_folder.replace(" ", "").lower()
        lens_path = os.path.join(BASE_DIR, lens_folder)
        
        if not os.path.isdir(lens_path):
            continue

        pinhole_json = os.path.join(INTRINSICS_DIR, f"calibration_pinhole_{lens_id}.json")
        detector = StreetViewDetector(model_path=MODEL_PATH, intrinsics_path=pinhole_json)

        video_files = [f for f in os.listdir(lens_path) if f.startswith("undistorted_") and f.endswith(".mp4")]

        for v_file in video_files:
            input_path = os.path.join(lens_path, v_file)
            output_path = os.path.join(lens_path, v_file.replace("undistorted_", "detected_"))

            print(f"\n--- Processing {lens_folder} ---")
            lens_results = detector.run_video_inference(input_path, output_path)
            final_report[lens_folder] = lens_results

    # --- SAVE DETAILED JSON REPORT ---
    json_output_path = "vehicles_detected.json"
    try:
        with open(json_output_path, "w") as f:
            json.dump(final_report, f, indent=4)
        print(f"\n[INFO] Detailed report saved to {json_output_path}")
    except Exception as e:
        print(f"\n[ERROR] Could not save JSON report: {e}")

    # --- SUMMARY TABLE ---
    print("\n" + "="*85)
    print("TASK 1: FINAL DETECTION SUMMARY")
    print("="*85)
    print(f"{'Lens':<10} | {'Cars':<6} | {'Buses':<6} | {'Trucks':<6} | {'Autos':<6} | {'Bikes':<6} | {'Quality'}")
    print("-" * 85)
    for lens, data in final_report.items():
        c = data['counts']
        meta = data['metadata']
        quality_str = f"{meta['resolution']}@{int(meta['fps'])}fps"
        print(f"{lens:<10} | {c['car']:<6} | {c['bus']:<6} | {c['truck']:<6} | {c['auto_rickshaw']:<6} | {c['motorcycle']:<6} | {quality_str}")
    print("="*85)