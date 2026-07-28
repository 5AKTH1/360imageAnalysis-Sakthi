'''This code is used to detect vehicles using yolov8 in the most optimal way of a sample panaromic image of a busy road'''
import cv2
import py360convert
import numpy as np
from ultralytics import YOLO

class StreetViewDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.fov = 90
        self.window_size = (640, 640)

    def _pixel_to_spherical(self, px, py, u_c, v_c):
        """Maps local crop pixels to global Yaw/Pitch."""
        nx = (px / self.window_size[0] - 0.5) * 2.0
        ny = (py / self.window_size[1] - 0.5) * 2.0
        fov_rad = np.radians(self.fov)
        f = 1.0 / np.tan(fov_rad / 2.0)
        yaw_rad = np.radians(u_c) + np.arctan(nx / f)
        pitch_rad = np.radians(v_c) + np.arctan(-ny / np.sqrt(nx*nx + f*f))
        return np.degrees(yaw_rad), np.degrees(pitch_rad)

    def _spherical_to_equi_pixel(self, yaw, pitch, w, h):
        """Maps Yaw/Pitch back to final master image pixel coordinates."""
        ex = int(((yaw + 180) / 360) * w) % w
        ey = int(((90 - pitch) / 180) * h)
        return ex, ey

    def _get_global_bbox(self, x1, y1, x2, y2, u_deg, v_deg, w, h):
        """Converts local crop box to global equirectangular box."""
        yaw1, pitch1 = self._pixel_to_spherical(x1, y1, u_deg, v_deg)
        yaw2, pitch2 = self._pixel_to_spherical(x2, y2, u_deg, v_deg)
        gx1, gy1 = self._spherical_to_equi_pixel(yaw1, pitch1, w, h)
        gx2, gy2 = self._spherical_to_equi_pixel(yaw2, pitch2, w, h)
        if gx1 > gx2: gx2 += w
        return [gx1, gy1, gx2, gy2]

    def _get_robust_center(self, box, img_w):
        gx1, gy1, gx2, gy2 = box
        if gx1 > gx2:
            center_x = (gx1 + gx2 + img_w) / 2 % img_w
        else:
            center_x = (gx1 + gx2) / 2
        center_y = (gy1 + gy2) / 2
        return center_x, center_y

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        denominator = float(boxAArea + boxBArea - interArea)
        if denominator == 0: return 0
        return interArea / denominator

    def _merge_boxes(self, boxA, confA, boxB, confB):
        total_conf = confA + confB
        new_box = []
        for i in range(4):
            val = (boxA[i] * confA + boxB[i] * confB) / total_conf
            new_box.append(val)
        return new_box

    def _hybrid_nms(self, detections, img_w, iou_threshold=0.5):
        if not detections: return []
        
        VEHICLE_GROUP = {'car', 'truck', 'bus', 'train', 'motorcycle', 'auto_rickshaw'}
        
        # Sort by confidence
        dets = sorted(detections, key=lambda x: x['conf'], reverse=True)
        keep = []

        while dets:
            best = dets.pop(0)
            merged_best_box = best['global_box']
            merged_conf_sum = best['conf']
            merged_count = best.get('merge_count', 1)
            
            # Get center of the current 'best' candidate
            best_cx, best_cy = self._get_robust_center(merged_best_box, img_w)
            
            # --- NEW: Calculate Dynamic Width ---
            # We use the width of the box to determine the "Personal Space" threshold
            bgx1, _, bgx2, _ = merged_best_box
            if bgx1 > bgx2: best_width = (bgx2 + img_w) - bgx1 # Handle wrapping
            else:           best_width = bgx2 - bgx1
            
            # Threshold: 50% of the width. 
            # If centers are within half-a-shoulder-width, it's a duplicate.
            # If they are wider apart, it's a second person.
            dynamic_dist_thresh = max(5.0, best_width * 0.5)

            remaining = []
            for d in dets:
                should_merge = False
                
                same_class = (best['class'] == d['class'])
                best_is_vehicle = best['class'] in VEHICLE_GROUP
                d_is_vehicle = d['class'] in VEHICLE_GROUP
                
                if same_class or (best_is_vehicle and d_is_vehicle):
                    
                    # --- STRATEGY A: DYNAMIC DISTANCE (For People) ---
                    if best['class'] == 'person':
                        d_cx, d_cy = self._get_robust_center(d['global_box'], img_w)
                        
                        dx = abs(best_cx - d_cx)
                        dy = abs(best_cy - d_cy)
                        if dx > img_w / 2: dx = img_w - dx
                        
                        dist = np.sqrt(dx**2 + dy**2)
                        
                        # Compare against the Dynamic Threshold (0.5 * Width)
                        if dist < dynamic_dist_thresh:
                            should_merge = True

                    # --- STRATEGY B: IoU (For Vehicles) ---
                    else:
                        if self._calculate_iou(merged_best_box, d['global_box']) > iou_threshold:
                            should_merge = True
                
                if should_merge:
                    merged_best_box = self._merge_boxes(merged_best_box, merged_conf_sum, d['global_box'], d['conf'])
                    merged_conf_sum += d['conf']
                    merged_count += 1
                    
                    # Update center & width for next comparison (Refining the 'Average' Person)
                    best_cx, best_cy = self._get_robust_center(merged_best_box, img_w)
                    
                    bgx1, _, bgx2, _ = merged_best_box
                    if bgx1 > bgx2: best_width = (bgx2 + img_w) - bgx1
                    else:           best_width = bgx2 - bgx1
                    dynamic_dist_thresh = max(5.0, best_width * 0.5)
                else:
                    remaining.append(d)
            
            best['global_box'] = merged_best_box
            best['merge_count'] = merged_count
            keep.append(best)
            dets = remaining
        return keep

    def _is_detection_valid(self, conf, box_h, img_h, cls):
        """
        DYNAMIC THRESHOLDING:
        - Big objects: Require standard confidence (0.50).
        - Small objects (Distant): Accept LOWER confidence (0.25).
        """
        # Define base thresholds
        BASE_THRESHOLDS = {
            'car': 0.50, 'bus': 0.50, 'truck': 0.50,
            'person': 0.40, 'bicycle': 0.40
        }
        base_thresh = BASE_THRESHOLDS.get(cls, 0.45)

        # Check relative height (Object Height / Image Height)
        rel_height = box_h / img_h
        
        # LOGIC FLIP: If object is small (< 5% of image), relax the threshold
        if rel_height < 0.05:
            # Drop the required threshold by 50% for distant objects
            required_thresh = base_thresh * 0.5 
            # (Example: Car threshold drops from 0.50 -> 0.25)
        else:
            required_thresh = base_thresh

        # Hard floor to prevent total garbage (e.g. 0.1)
        required_thresh = max(0.20, required_thresh)

        return conf >= required_thresh

    def run_inference(self, img_path, steps=8, person_dist_px=80):
        img = cv2.imread(img_path)
        if img is None: raise ValueError(f"Check path: {img_path}")
        h, w = img.shape[:2]
        vis_img = img.copy()
        step_deg = 360 // steps
        all_detections = []

        print("Step 1: Sliding Window Detection...")
        for u_deg in range(-180, 180, step_deg):
            persp = py360convert.e2p(img, fov_deg=self.fov, u_deg=u_deg, v_deg=0, out_hw=self.window_size)
            results = self.model(persp, verbose=False)[0]

            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                label = self.model.names[int(box.cls[0])]
                
                g_box = self._get_global_bbox(x1, y1, x2, y2, u_deg, 0, w, h)
                gx1, gy1, gx2, gy2 = g_box
                obj_height = abs(gy2 - gy1)
                
                # --- NEW LOGIC: DISTANCE-AWARE FILTERING ---
                # We filter immediately based on the dynamic size/confidence rule
                if self._is_detection_valid(conf, obj_height, h, label):
                    all_detections.append({
                        'class': label, 
                        'conf': conf, 
                        'global_box': g_box,
                        'height': obj_height
                    })

        print(f"Found {len(all_detections)} valid candidates (including distant ones). Running NMS...")
        
        unique_detections = self._hybrid_nms(all_detections, img_w=w, 
                                             iou_threshold=0.5)
        
        print(f"Final Count: {len(unique_detections)}")

        # Draw
        for det in unique_detections:
            gx1, gy1, gx2, gy2 = map(int, det['global_box'])
            label = det['class']
            conf = det['conf']
            draw_x2 = gx2 % w
            center_x, center_y = map(int, self._get_robust_center(det['global_box'], w))
            
            cv2.circle(vis_img, (center_x, center_y), 8, (0, 255, 0), -1)
            cv2.putText(vis_img, f"{label} {conf:.2f}", (center_x + 10, center_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if gx1 < draw_x2:
                cv2.rectangle(vis_img, (gx1, gy1), (draw_x2, gy2), (0, 0, 255), 2)

        return vis_img

if __name__ == "__main__":
    detector = StreetViewDetector()
    input_path = r"C:\IITM\Sample Street View 360.jpg" 
    try:
        final_vis = detector.run_inference(input_path, person_dist_px=80)
        cv2.imwrite("final_output_boosted.jpg", final_vis)
        print("Done.")
    except Exception as e:
        print(e)