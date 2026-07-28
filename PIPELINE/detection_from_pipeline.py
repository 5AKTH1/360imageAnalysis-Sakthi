import os
import cv2
import numpy as np
import json
from ultralytics import YOLO

class PanoramicImageDetector:
    def __init__(self, model_path):
        print(f"[INFO] Loading Model: {model_path}...")
        self.model = YOLO(model_path)
        self.colors = {
            'auto': (0, 165, 255), 'bus': (255, 0, 0), 'car': (0, 255, 0),
            'motorbike': (255, 255, 0), 'truck': (0, 0, 255)
        }
        self.excluded_classes = ['tractor', 'rickshaw', 'e-rickshaw', 'cart', 'person', 'cycle']
        
        # --- FIXED CONFIGURATION FOR 1664p PANORAMAS ---
        self.SKY_Y_LIMIT = 800  
        self.VEHICLE_Y_LIMIT = 1150 
        
        # --- CUSTOM EGO-VEHICLE POLYGON ---
        self.EGO_POLYGON = np.array([
            [0, 1021], [78, 1088], [341, 1165], [419, 1094], 
            [612, 1121], [679, 1206], [875, 1223], [921, 1290], 
            [1083, 1229], [1616, 1215], [1930, 1306], [2064, 1113], 
            [2206, 1065], [2340, 1111], [2807, 1021], [3267, 1009], 
            [3287, 1013], 
            [3328, 1664], # Locked to bottom-right corner
            [0, 1664]     # Locked to bottom-left corner
        ], np.int32)        
        
        self.CONF_FAR = 0.10        
        self.CONF_CLOSE = 0.30    

    def get_dynamic_threshold(self, bbox_bottom_y, label):
        # --- CONDITIONAL VIP PASS ---
        # Only trust low-confidence trucks/buses if they are further down the image (on the road)
        # If their bottom is above 950, they are likely background/buildings, so we stay strict.
        if label in ['bus', 'truck']:
            if bbox_bottom_y > 950:
                return 0.15  # Trust them on the road
            else:
                return 0.40  # Be strict if they are high up (likely buildings)

        y_clamped = max(self.SKY_Y_LIMIT, min(bbox_bottom_y, self.VEHICLE_Y_LIMIT))
        
        if self.VEHICLE_Y_LIMIT == self.SKY_Y_LIMIT:
            ratio = 1.0
        else:
            linear_ratio = (y_clamped - self.SKY_Y_LIMIT) / (self.VEHICLE_Y_LIMIT - self.SKY_Y_LIMIT)
            ratio = linear_ratio ** 2 
            
        base_thresh = self.CONF_FAR + (self.CONF_CLOSE - self.CONF_FAR) * ratio
        
        if label == 'auto': return max(base_thresh, 0.6) 
        if label == 'motorbike': return max(base_thresh, 0.2)
        return base_thresh

    def process_single_image(self, image_path, output_path):
        if not os.path.exists(image_path): 
            print(f"[ERROR] Image not found: {image_path}")
            return {}

        print(f"[INFO] Processing image: {os.path.basename(image_path)}")
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[ERROR] Could not read image: {image_path}")
            return {}

        h, w = frame.shape[:2]
        
        full_metadata = {
            "source_file": image_path,
            "resolution": f"{w}x{h}",
            "summary_counts": {},
            "objects": [] 
        }
        
        all_boxes = []
        all_confs = []
        all_clss = []
        
        inference_scales = [2560, 3360]
        
        for size in inference_scales:
            results = self.model.predict(frame, 
                                         verbose=False, 
                                         conf=0.35,  
                                         iou=0.45, 
                                         imgsz=size)[0]
            
            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                clss = results.boxes.cls.int().cpu().tolist()
                confs = results.boxes.conf.cpu().tolist()

                for box, cls_idx, conf in zip(boxes, clss, confs):
                    # 1. Sky Filter 
                    if box[3] < self.SKY_Y_LIMIT: continue 
                    
                    # 2. Ego-Vehicle Polygon Mask
                    bottom_center = (int((box[0] + box[2]) / 2), int(box[3]))
                    if cv2.pointPolygonTest(self.EGO_POLYGON, bottom_center, False) >= 0:
                        continue
                    
                    raw_label = self.model.names[cls_idx]
                    if raw_label in self.excluded_classes: continue

                    # 3. Dynamic Exponential Thresholding
                    req_conf = self.get_dynamic_threshold(box[3], raw_label)
                    if conf < req_conf: continue
                    
                    all_boxes.append(box.tolist())
                    all_confs.append(conf)
                    all_clss.append(raw_label)

        live_counts = {name: 0 for name in ['car', 'auto', 'bus', 'truck', 'motorbike']}
        
        # --- PRE-NMS: IoA SWALLOWED BOX REJECTION ---
        # Only reject if the smaller box is almost entirely (85%+) covered by the big box
        valid_indices = []
        for i in range(len(all_boxes)):
            box1 = all_boxes[i]
            label1 = all_clss[i]
            
            is_swallowed = False
            for j in range(len(all_boxes)):
                if i == j: continue
                box2 = all_boxes[j]
                label2 = all_clss[j]
                
                # Big vehicles swallowing small vehicles
                if label2 in ['bus', 'truck'] and label1 not in ['bus', 'truck']:
                    
                    # Calculate Intersection Area
                    x_left = max(box1[0], box2[0])
                    y_top = max(box1[1], box2[1])
                    x_right = min(box1[2], box2[2])
                    y_bottom = min(box1[3], box2[3])
                    
                    if x_right > x_left and y_bottom > y_top:
                        intersection_area = (x_right - x_left) * (y_bottom - y_top)
                        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                        
                        # IoA (Intersection over Area)
                        ioa = intersection_area / box1_area
                        
                        # If 85% or more is inside the massive box, it's a false texture, delete it!
                        if ioa > 0.85:
                            print(f"[-] SWALLOWED REJECT: '{label1}' (IoA {ioa:.2f}) was inside a '{label2}'.")
                            is_swallowed = True
                            break
            
            if not is_swallowed:
                valid_indices.append(i)

        filtered_boxes = [all_boxes[i] for i in valid_indices]
        filtered_confs = [all_confs[i] for i in valid_indices]
        filtered_clss = [all_clss[i] for i in valid_indices]

        # --- NON-MAXIMUM SUPPRESSION (NMS) ---
        if len(filtered_boxes) > 0:
            cv_boxes = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in filtered_boxes]
            indices = cv2.dnn.NMSBoxes(cv_boxes, filtered_confs, score_threshold=0.0, nms_threshold=0.35)
            
            for i in indices:
                box = filtered_boxes[i]
                conf = filtered_confs[i]
                raw_label = filtered_clss[i]
                
                if raw_label in live_counts: 
                    live_counts[raw_label] += 1
                
                full_metadata["objects"].append({
                    "label": raw_label, 
                    "bbox": box,
                    "confidence": round(conf, 3)
                })

                color = self.colors.get(raw_label, (255, 255, 255))
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 3) 
                label_text = f"{raw_label} {conf:.2f}"
                cv2.putText(frame, label_text, (box[0], box[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
        # Draw the custom Ego-Vehicle exclusion zone
        cv2.polylines(frame, [self.EGO_POLYGON], isClosed=True, color=(0, 0, 255), thickness=2)

        # --- DRAW COUNTER OVERLAY ---
        cv2.rectangle(frame, (10, 10), (250, 230), (0,0,0), -1) 
        cv2.putText(frame, "VEHICLE COUNT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        for i, (v_type, count) in enumerate(live_counts.items()):
            text = f"{v_type.upper()}: {count}"
            cv2.putText(frame, text, (20, 80 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imwrite(output_path, frame)
        
        json_output = output_path.replace(".jpg", ".json").replace(".png", ".json")
        full_metadata["summary_counts"] = live_counts
        with open(json_output, "w") as f:
            json.dump(full_metadata, f, indent=4)
            
        print(f"[INFO] Saved detected image and metadata.")
        return live_counts

if __name__ == "__main__":
    SOURCE_IMAGE = r"C:\IITM\DATA_RETRIEVAL_TASK\Pipeline_2026-02-02_09-36-59_F43397\Final_Static_Stitch.jpg" 
    OUTPUT_IMAGE = SOURCE_IMAGE.replace(".jpg", "_detected2.jpg")
    MODEL = r"C:\IITM\Vehicle_detection_360\runs\detect\FINAL_balanced_v11_6classes\weights\best.pt"
    
    detector = PanoramicImageDetector(model_path=MODEL)
    results = detector.process_single_image(SOURCE_IMAGE, OUTPUT_IMAGE)
    
    print("\n" + "="*40)
    print(" FINAL TRAFFIC REPORT ")
    print("="*40)
    if results:
        for v_type, count in results.items():
            print(f"{v_type.upper():<15} : {count}")
    else:
        print("No results found or error reading image.")
    print("="*40)