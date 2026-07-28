import os
import cv2
import numpy as np
import json
import datetime
from ultralytics import YOLO
from collections import Counter

class StreetViewDetector:
    def __init__(self, model_path=r'C:\IITM\Vehicle_detection_360\runs\detect\vehicles_yolov11\weights\best.pt', intrinsics_path=None):
        print(f"[INFO] Loading Unified Model: {model_path}...")
        self.model = YOLO(model_path)
        
        self.colors = {
            'LCV': (0, 255, 255), 'auto': (0, 165, 255), 'bus': (255, 0, 0),
            'car': (0, 255, 0), 'cart': (128, 128, 128), 'cycle': (255, 0, 255),
            'e-rickshaw': (0, 255, 127), 'motorbike': (255, 255, 0),
            'person': (255, 255, 255), 'rickshaw': (0, 69, 255),
            'tractor': (42, 42, 165), 'truck': (0, 0, 255)
        }

    def run_video_inference(self, video_path, output_path):
        # --- NORMALIZING FPS TO 20 ---
        temp_input = video_path.replace(".mp4", "_20fps_temp.mp4")
        cap_orig = cv2.VideoCapture(video_path)
        
        orig_fps = cap_orig.get(cv2.CAP_PROP_FPS)
        w = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
        target_fps = 20.0
        
        print(f"[INFO] Normalizing {os.path.basename(video_path)} from {orig_fps}fps to {target_fps}fps...")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_temp = cv2.VideoWriter(temp_input, fourcc, target_fps, (w, h))
        
        while cap_orig.isOpened():
            ret, frame = cap_orig.read()
            if not ret: break
            out_temp.write(frame)
            
        cap_orig.release()
        out_temp.release()

        # --- START DETECTION ON 20FPS VIDEO ---
        cap = cv2.VideoCapture(temp_input)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (w, h))

        report_classes = [c for c in self.colors.keys() if c != 'LCV']
        unique_objects = {name: {} for name in report_classes}
        
        # Tracking Dictionaries
        id_to_class_map = {}
        id_frame_count = {} 
        id_class_history = {} # Tracks recent classifications for Majority Voting

        print(f"[INFO] Processing Detections with BoT-SORT + Voting...")
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1

            # Tracking with BoT-SORT
            results = self.model.track(
                frame, 
                persist=True, 
                verbose=False, 
                conf=0.20,        
                iou=0.45,         
                imgsz=640, 
                tracker="custom_track.yaml" 
            )[0]
            
            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                ids = results.boxes.id.int().cpu().tolist()
                confs = results.boxes.conf.cpu().tolist()
                clss = results.boxes.cls.int().cpu().tolist()

                for box, t_id, conf, cls in zip(boxes, ids, confs, clss):
                    id_frame_count[t_id] = id_frame_count.get(t_id, 0) + 1
                    
                    # --- MAJORITY VOTING LOGIC ---
                    raw_label = self.model.names[cls]
                    current_prediction = 'car' if raw_label == 'LCV' else raw_label
                    
                    if t_id not in id_class_history:
                        id_class_history[t_id] = []
                    
                    # Store the prediction in history
                    id_class_history[t_id].append(current_prediction)
                    
                    # Keep a sliding window of 30 frames
                    if len(id_class_history[t_id]) > 30:
                        id_class_history[t_id].pop(0)
                    
                    # Determine the label based on the most frequent class in recent history
                    label = Counter(id_class_history[t_id]).most_common(1)[0][0]
                    id_to_class_map[t_id] = label
                    
                    # Drawing
                    color = self.colors.get(label, (255, 255, 255))
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(frame, f"ID:{t_id} {label}", (box[0], box[1]-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # 15-frame minimum visibility for logging in summary
                    if id_frame_count[t_id] > 15:
                        if label in unique_objects:
                            if conf > unique_objects[label].get(t_id, 0.0):
                                unique_objects[label][t_id] = conf

            out.write(frame)
            if frame_idx % 50 == 0: print(f"   Progress: {frame_idx}/{total_frames} frames...", end="\r")
        
        cap.release()
        out.release()
        
        if os.path.exists(temp_input):
            os.remove(temp_input)

        return {
            "counts": {name: len(id_map) for name, id_map in unique_objects.items()},
            "metadata": {"fps": target_fps, "resolution": f"{w}x{h}"}
        }

if __name__ == "__main__":
    SOURCE_VIDEO = r"C:\IITM\Results\lens 1\source_1080p.mp4"
    UNIFIED_MODEL_PATH = r"C:\IITM\Vehicle_detection_360\runs\detect\vehicles_yolov11\weights\best.pt" 
    
    detector = StreetViewDetector(model_path=UNIFIED_MODEL_PATH)
    output_path = SOURCE_VIDEO.replace("source_1080p", "detected_12class_final_voted")
    
    print(f"\n--- Starting Optimized Inference (BoT-SORT + Voting) ---")
    results = detector.run_video_inference(SOURCE_VIDEO, output_path)

    with open("vehicles_detected_12class.json", "w") as f:
        json.dump(results, f, indent=4)

    c = results['counts']
    print("\n" + "="*90)
    print(f"{'Auto':<6} | {'Rick':<6} | {'E-Rik':<6} | {'Car':<6} | {'Bus':<6} | {'Truck':<6} | {'Bike':<6}")
    print("-" * 90)
    print(f"{c['auto']:<6} | {c['rickshaw']:<6} | {c['e-rickshaw']:<6} | {c['car']:<6} | {c['bus']:<6} | {c['truck']:<6} | {c['motorbike']:<6}")
    print("="*90)