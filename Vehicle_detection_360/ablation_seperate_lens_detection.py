import os
import cv2
import numpy as np
import json
import datetime
import time
import torch
from ultralytics import YOLO

class GPUAblationDetector:
    def __init__(self, model_path='yolov8n.pt', intrinsics_path=None):
        # FORCE GPU: device=0
        device = 0 if torch.cuda.is_available() else 'cpu'
        print(f"Loading YOLO on: {device} (CUDA Available: {torch.cuda.is_available()})")
        
        self.model = YOLO(model_path)
        self.device = device
        self.fov = 90
        
        if intrinsics_path and os.path.exists(intrinsics_path):
            with open(intrinsics_path, "r") as f:
                data = json.load(f)
            self.K = np.asarray(data["K"], dtype=np.float64)
        else:
            self.K = np.eye(3)

    def _is_detection_valid(self, conf, box_h, img_h, cls):
        THRESHOLDS = {'car': 0.50, 'bus': 0.50, 'truck': 0.50, 'bicycle': 0.40, 'motorcycle': 0.40}
        base_thresh = THRESHOLDS.get(cls, 0.45)
        if box_h / img_h < 0.05: base_thresh *= 0.5
        return conf >= max(0.20, base_thresh)

    def run_inference(self, source_path, output_path, target_fps, lens_id="lens1"):
        cap = cv2.VideoCapture(source_path)
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # FPS Skip Logic
        if target_fps > source_fps: target_fps = source_fps
        frame_step = max(1, int(round(source_fps / target_fps)))
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (w, h))

        unique_objects = {'car': {}, 'bus': {}, 'truck': {}, 'motorcycle': {}, 'auto_rickshaw': {}}
        
        start_time = time.time()
        processed_count = 0

        # Run Tracker on GPU
        results = self.model.track(
            source=source_path, 
            stream=True, 
            persist=True, 
            verbose=False, 
            conf=0.20, 
            tracker="bytetrack.yaml",
            device=self.device # EXPLICIT GPU CALL
        )

        for frame_idx, result in enumerate(results):
            # Simulate Low FPS by skipping frames
            if frame_idx % frame_step != 0:
                continue
            
            frame = result.orig_img
            
            # Note: Result is already on GPU, but boxes need moving to CPU for drawing
            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().tolist()
                clss = result.boxes.cls.int().cpu().tolist()

                for box, track_id, conf, cls in zip(boxes, track_ids, confs, clss):
                    label = self.model.names[cls]
                    
                    if self._is_detection_valid(conf, abs(box[3]-box[1]), h, label):
                        if label in unique_objects:
                            if track_id not in unique_objects[label]:
                                unique_objects[label][track_id] = {'max_conf': 0.0, 'frames_seen': 0}
                            
                            obj = unique_objects[label][track_id]
                            obj['frames_seen'] += 1
                            if conf > obj['max_conf']: obj['max_conf'] = conf
                        
                        # Drawing
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                        
                        font_scale = max(0.5, 1.2 * (w / 3840))
                        font_thick = max(1, int(3 * (w / 3840)))
                        
                        cv2.putText(frame, f"ID:{track_id} {label} {conf:.2f}", 
                                  (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thick)

            out.write(frame)
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"  Processed {processed_count} frames...", end="\r")

        cap.release()
        out.release()
        
        total_time = time.time() - start_time
        speed_fps = processed_count / total_time if total_time > 0 else 0
        
        # Build Stats
        stats = {
            "counts": {k: len(v) for k, v in unique_objects.items()},
            "avg_confidence": {},
            "avg_lifespan": {},
            "metadata": {
                "resolution": f"{w}x{h}",
                "simulated_fps": target_fps,
                "processing_speed": round(speed_fps, 2)
            }
        }
        
        for label, data in unique_objects.items():
            if data:
                stats["avg_confidence"][label] = round(sum(d['max_conf'] for d in data.values()) / len(data), 4)
                stats["avg_lifespan"][label] = round(sum(d['frames_seen'] for d in data.values()) / len(data), 1)
            else:
                stats["avg_confidence"][label] = 0.0
                stats["avg_lifespan"][label] = 0.0
                
        return stats

if __name__ == "__main__":
    BASE_DIR = r"C:\IITM\CAMERA_Short_Clips"
    LENS_DIR = os.path.join(BASE_DIR, "lens 1")
    INTRINSICS = r"C:\IITM\Vehicle_detection_360\Intrinsics\calibration_pinhole_lens1.json"
    
    # 1. Config Matrix
    # We map "Test Name" to "Source File"
    # Ensure you ran Step 1 so these files exist!
    CONFIGS = [
        ("4k", [f for f in os.listdir(LENS_DIR) if f.startswith("undistorted_")][0]), # Original
        ("1080p", "source_1080p.mp4"),
        ("720p", "source_720p.mp4"),
        ("480p", "source_480p.mp4")
    ]
    
    FPS_OPTS = [30, 20, 10, 5]
    
    detector = GPUAblationDetector(intrinsics_path=INTRINSICS)
    full_report = {}

    print(f"\n--- STARTING GPU ABLATION STUDY ---")
    
    for res_name, source_file in CONFIGS:
        source_path = os.path.join(LENS_DIR, source_file)
        if not os.path.exists(source_path):
            print(f"Skipping {res_name}: File not found ({source_file})")
            continue
            
        # Create output folder (e.g., lens 1/4k)
        out_folder = os.path.join(LENS_DIR, res_name)
        os.makedirs(out_folder, exist_ok=True)
        
        for fps in FPS_OPTS:
            # Skip 4K 30fps as requested
            if res_name == "4k" and fps == 30:
                print("Skipping 4k 30fps...")
                continue
                
            out_name = f"detected_{res_name}_{fps}fps_lens1.mp4"
            out_path = os.path.join(out_folder, out_name)
            
            print(f"\n> Processing {res_name} @ {fps} FPS (Source: {source_file})")
            
            # Run
            stats = detector.run_inference(source_path, out_path, fps)
            full_report[f"{res_name}_{fps}fps"] = stats

    # Save JSON
    with open(os.path.join(BASE_DIR, "ablation_study_gpu.json"), "w") as f:
        json.dump(full_report, f, indent=4)
    print("\n\nDone! JSON saved.")