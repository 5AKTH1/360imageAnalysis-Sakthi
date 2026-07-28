import cv2
import py360convert
import numpy as np
from ultralytics import YOLO
import time
import json
import os
import supervision as sv

class StreetViewDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.fov = 90
        self.window_size = (640, 640)
        
        # INCREASED THRESHOLD: Only track objects that survive the Distance Filter
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.4, # Stricter activation
            lost_track_buffer=60, 
            minimum_matching_threshold=0.8, 
            frame_rate=20
        )
        
        self.class_id_map = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
            4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 
        }

    # --- MATH HELPERS ---
    def _pixel_to_spherical(self, px, py, u_c, v_c):
        nx = (px / self.window_size[0] - 0.5) * 2.0
        ny = (py / self.window_size[1] - 0.5) * 2.0
        fov_rad = np.radians(self.fov)
        f = 1.0 / np.tan(fov_rad / 2.0)
        denom = np.sqrt(nx*nx + f*f)
        if denom == 0: denom = 1e-6
        yaw_rad = np.radians(u_c) + np.arctan(nx / f)
        pitch_rad = np.radians(v_c) + np.arctan(-ny / denom)
        return np.degrees(yaw_rad), np.degrees(pitch_rad)

    def _spherical_to_equi_pixel(self, yaw, pitch, w, h):
        if np.isnan(yaw) or np.isnan(pitch): return 0, 0
        ex = int(((yaw + 180) / 360) * w) % w
        ey = int(((90 - pitch) / 180) * h)
        return ex, ey

    def _get_global_bbox(self, x1, y1, x2, y2, u_deg, v_deg, w, h):
        yaw1, pitch1 = self._pixel_to_spherical(x1, y1, u_deg, v_deg)
        yaw2, pitch2 = self._pixel_to_spherical(x2, y2, u_deg, v_deg)
        gx1, gy1 = self._spherical_to_equi_pixel(yaw1, pitch1, w, h)
        gx2, gy2 = self._spherical_to_equi_pixel(yaw2, pitch2, w, h)
        if gx1 > gx2: gx2 += w
        gx1, gy1 = max(0, gx1), max(0, gy1)
        return [gx1, gy1, gx2, gy2]

    def _apply_distance_penalty(self, conf, box_h, img_h):
        """
        Logic:
        1. Calculate 'Relative Height' (0.0 to 1.0).
        2. If object is SMALL (<5% height), it is FAR.
        3. FAR objects get a confidence PENALTY.
        4. NEAR objects keep original confidence.
        """
        rel_height = box_h / img_h
        
        # Distance Factor: 1.0 (Close) to 0.5 (Far)
        # We linearly scale down the confidence for small objects
        if rel_height < 0.10: # If smaller than 10% of screen
            # Penalty increases as object gets smaller
            penalty_factor = max(0.5, rel_height * 10) 
            adjusted_conf = conf * penalty_factor
        else:
            adjusted_conf = conf

        return adjusted_conf

    def run_video_inference(self, video_path, output_path, steps=8):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return

        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if src_fps <= 0 or np.isnan(src_fps): src_fps = 30

        TARGET_FPS = 20
        TARGET_HEIGHT = 1080
        aspect_ratio = src_w / src_h if src_h > 0 else 1.77
        TARGET_WIDTH = int(TARGET_HEIGHT * aspect_ratio)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
        out = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, (TARGET_WIDTH, TARGET_HEIGHT))

        unique_ids_seen = set()
        stats = {"source": video_path, "total_detections_raw": 0}

        start_time = time.time()
        frame_idx = 0
        frames_written = 0

        print(f"Processing... Target: {TARGET_WIDTH}x{TARGET_HEIGHT} @ {TARGET_FPS} FPS")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            should_process = int(frame_idx * TARGET_FPS / src_fps) > int((frame_idx - 1) * TARGET_FPS / src_fps)
            if frame_idx == 0: should_process = True
            if not should_process:
                frame_idx += 1
                continue

            resized_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
            vis_img = resized_frame.copy()
            
            raw_boxes = []
            raw_confidences = []
            raw_class_ids = []

            step_deg = 360 // steps
            for u_deg in range(-180, 180, step_deg):
                try:
                    persp = py360convert.e2p(resized_frame, fov_deg=self.fov, u_deg=u_deg, v_deg=0, out_hw=self.window_size)
                    results = self.model(persp, verbose=False)[0]

                    for box in results.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        cls_id = int(box.cls[0])
                        label = self.model.names[cls_id]
                        if label not in self.class_id_map.values(): continue

                        g_box = self._get_global_bbox(x1, y1, x2, y2, u_deg, 0, TARGET_WIDTH, TARGET_HEIGHT)
                        gx1, gy1, gx2, gy2 = g_box
                        obj_h = abs(gy2 - gy1)
                        
                        # --- NEW CONFIDENCE LOGIC ---
                        # 1. Apply Distance Penalty
                        final_conf = self._apply_distance_penalty(conf, obj_h, TARGET_HEIGHT)
                        
                        # 2. Strict Threshold Check (> 0.4)
                        if final_conf > 0.4:
                            
                            # Duplicate Pre-Filter
                            is_duplicate = False
                            center_x = (gx1 + gx2) / 2
                            if gx1 > gx2: center_x = (gx1 + gx2 + TARGET_WIDTH) / 2 % TARGET_WIDTH
                            
                            for existing_box in raw_boxes:
                                ex1, _, ex2, _ = existing_box
                                e_center = (ex1 + ex2) / 2
                                if abs(center_x - e_center) < (TARGET_WIDTH * 0.02): 
                                    is_duplicate = True
                                    break
                            
                            if not is_duplicate:
                                raw_boxes.append(g_box)
                                raw_confidences.append(final_conf) # Use Adjusted Conf
                                raw_class_ids.append(cls_id)
                except:
                    continue

            # TRACKER UPDATE
            if len(raw_boxes) > 0:
                np_boxes = np.array(raw_boxes, dtype=np.float32)
                np_conf = np.array(raw_confidences, dtype=np.float32)
                np_cls = np.array(raw_class_ids, dtype=int)

                detections = sv.Detections(
                    xyxy=np_boxes,
                    confidence=np_conf,
                    class_id=np_cls
                ).with_nms(threshold=0.5)
                
                tracked_detections = self.tracker.update_with_detections(detections)
            else:
                tracked_detections = sv.Detections.empty()

            # DRAWING (With ID Display)
            for i in range(len(tracked_detections)):
                box = tracked_detections.xyxy[i]
                tracker_id = int(tracked_detections.tracker_id[i])
                class_id = int(tracked_detections.class_id[i])
                
                if class_id in self.model.names:
                    label = self.model.names[class_id]
                else:
                    label = "Unk"

                if tracker_id not in unique_ids_seen:
                    unique_ids_seen.add(tracker_id)
                
                x1, y1, x2, y2 = map(int, box)
                draw_x2 = x2 % TARGET_WIDTH
                
                # Dynamic Color based on ID
                color = ((tracker_id * 50) % 255, (tracker_id * 80) % 255, (tracker_id * 110) % 255)
                
                if x1 < draw_x2:
                    cv2.rectangle(vis_img, (x1, y1), (draw_x2, y2), color, 2)
                    
                    # --- LARGE ID DISPLAY ---
                    caption = f"ID #{tracker_id} | {label}"
                    # Black Background for text
                    (w, h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(vis_img, (x1, y1 - 20), (x1 + w, y1), color, -1)
                    # White Text
                    cv2.putText(vis_img, caption, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            out.write(vis_img)
            frames_written += 1
            frame_idx += 1
            
            print(f"Frame {frames_written} | Tracks: {len(tracked_detections)} | Unique: {len(unique_ids_seen)}", end='\r')

        # Save simple count
        json_path = output_path.replace('.mp4', '_count.json')
        with open(json_path, 'w') as f:
            json.dump({"unique_count": len(unique_ids_seen)}, f)

        cap.release()
        out.release()
        print(f"\nDone! Unique Vehicles Counted: {len(unique_ids_seen)}")

if __name__ == "__main__":
    detector = StreetViewDetector()
    input_video = r"C:\IITM\CAMERA_Short_Clips\Final_clean_Panorama_Corrected.mp4"
    output_video = r"C:\IITM\CAMERA_Short_Clips\Final_Detected_Panorama.mp4"
    
    try:
        detector.run_video_inference(input_video, output_video, steps=8)
    except Exception as e:
        print(f"\nError: {e}")