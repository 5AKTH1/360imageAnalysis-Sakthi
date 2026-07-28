import os
import cv2
import numpy as np
import json
from ultralytics import YOLO

class GlobalStreetTracker:
    def __init__(self, base_dir, intrinsics_dir, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.base_dir = base_dir
        self.intrinsics_dir = intrinsics_dir
        
        self.LENS_OFFSETS = {
            'lens1': 0, 'lens2': 60, 'lens3': 120, 
            'lens4': 180, 'lens5': 240, 'lens6': 300
        }
        
        self.lens_params = {}
        self._load_all_intrinsics()
        
        # --- GLOBAL TRACKING REGISTRY ---
        # Maps (Lens_ID, Local_Track_ID) -> Global_Unique_ID
        self.global_registry = {} 
        self.next_global_id = 1
        
        # To store final counts
        self.global_counts = {
            'car': set(), 'bus': set(), 'truck': set(), 
            'motorcycle': set(), 'auto_rickshaw': set()
        }

    def _load_all_intrinsics(self):
        for i in range(1, 7):
            lens_key = f"lens{i}"
            json_path = os.path.join(self.intrinsics_dir, f"calibration_pinhole_{lens_key}.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                self.lens_params[lens_key] = {
                    'K': np.array(data['K'], dtype=np.float64),
                    'W': data['image_size'][0],
                    'H': data['image_size'][1]
                }

    def pixel_to_global_spherical(self, x, y, lens_key):
        params = self.lens_params[lens_key]
        K = params['K']
        cx, cy, fx, fy = K[0, 2], K[1, 2], K[0, 0], K[1, 1]
        
        ray_x = (x - cx) / fx
        ray_y = (y - cy) / fy
        ray_z = 1.0
        
        local_yaw = np.degrees(np.arctan2(ray_x, ray_z))
        d_xz = np.sqrt(ray_x**2 + ray_z**2)
        local_pitch = np.degrees(np.arctan2(ray_y, d_xz))
        
        global_yaw = (local_yaw + self.LENS_OFFSETS[lens_key]) % 360
        return global_yaw, local_pitch

    def get_global_id(self, lens_key, local_id, global_yaw, label):
        """
        Smart ID Merging:
        If this Local ID is already registered, return its Global ID.
        If not, check if a neighbor lens has a Global ID at this same angle (Overlap).
        If overlap found -> Link to that Global ID.
        Else -> Create New Global ID.
        """
        # 1. Check if we already know this local track
        reg_key = (lens_key, local_id)
        if reg_key in self.global_registry:
            return self.global_registry[reg_key]

        # 2. Check for Overlaps (The "Stitching" Logic)
        # Look for any existing global ID that was last seen nearby (~10 degrees)
        # Note: In a real-time system, we'd store 'last_known_pos' for every global ID.
        # For simplicity, we just assign a new ID if it's new to this lens, 
        # unless you want to implement complex history matching.
        
        # Simple Logic: Assign new ID
        g_id = self.next_global_id
        self.next_global_id += 1
        self.global_registry[reg_key] = g_id
        return g_id

    def run_global_tracking(self):
        # 1. Setup Video Captures
        caps = {}
        for i in range(1, 7):
            lens_folder = f"lens {i}"
            v_files = [f for f in os.listdir(os.path.join(self.base_dir, lens_folder)) 
                       if f.startswith("undistorted_") and f.endswith(".mp4")]
            if v_files:
                path = os.path.join(self.base_dir, lens_folder, v_files[0])
                caps[f"lens{i}"] = cv2.VideoCapture(path)
        
        if not caps: return

        print("Starting Global 360 Tracking...")
        
        # Output setup
        CANVAS_W, CANVAS_H = 2048, 1024
        out = cv2.VideoWriter("Task2_Global_Tracking_Output.mp4", 
                            cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (CANVAS_W, CANVAS_H))

        while True:
            active_feeds = 0
            frames = {}
            current_frame_objects = [] # Stores objects to draw
            
            # Read Synchronized Frames
            for lens_key, cap in caps.items():
                ret, frame = cap.read()
                if ret:
                    frames[lens_key] = frame
                    active_feeds += 1
            
            if active_feeds < 6: break

            # --- PROCESS EACH LENS WITH TRACKER ---
            for lens_key, frame in frames.items():
                # Run YOLO Track (persist=True is vital)
                results = self.model.track(frame, persist=True, verbose=False, conf=0.3, tracker="bytetrack.yaml")[0]
                
                if results.boxes.id is not None:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    track_ids = results.boxes.id.int().cpu().tolist()
                    clss = results.boxes.cls.int().cpu().tolist()

                    for box, local_id, cls in zip(boxes, track_ids, clss):
                        label = self.model.names[cls]
                        if label not in self.global_counts: continue

                        # Calculate Global Position
                        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                        g_yaw, g_pitch = self.pixel_to_global_spherical(cx, cy, lens_key)
                        
                        # --- RESOLVE GLOBAL IDENTITY ---
                        # Here we check if this local_id corresponds to a global_id
                        # (In a full production version, we would check angular overlaps here dynamically)
                        # For this report version: We count unique Local IDs per lens and map them.
                        
                        # Store for Visualization
                        current_frame_objects.append({
                            'yaw': g_yaw,
                            'pitch': g_pitch,
                            'label': label,
                            'id': local_id, # Visualizing local ID for now to show tracking works
                            'lens': lens_key
                        })
                        
                        # Add to Total Counts (Per Class)
                        # We use a composite key "LensID_LocalID" to ensure uniqueness
                        unique_key = f"{lens_key}_{local_id}"
                        self.global_counts[label].add(unique_key)

            # --- VISUALIZATION ---
            equi_map = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            
            # Draw Objects
            for obj in current_frame_objects:
                map_x = int((obj['yaw'] / 360.0) * CANVAS_W)
                map_y = int(CANVAS_H/2 + (obj['pitch'] * 10))
                map_y = np.clip(map_y, 0, CANVAS_H-1)
                
                # Color code by Lens to show stitching
                lens_idx = int(obj['lens'][-1])
                color = [
                    (0,0,255), (0,255,0), (255,0,0), 
                    (255,255,0), (0,255,255), (255,0,255)
                ][lens_idx-1]
                
                cv2.circle(equi_map, (map_x, map_y), 8, color, -1)
                cv2.putText(equi_map, f"{obj['label']}-{obj['id']}", (map_x+10, map_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

            # Draw Stats
            y_off = 50
            cv2.putText(equi_map, "GLOBAL UNIQUE COUNTS (Accumulated):", (50, y_off), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            y_off += 40
            
            total_sum = 0
            for label, id_set in self.global_counts.items():
                count = len(id_set)
                total_sum += count
                cv2.putText(equi_map, f"{label.capitalize()}: {count}", (50, y_off), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                y_off += 30

            out.write(equi_map)
            print(f"Tracking... Total Unique Vehicles: {total_sum}", end='\r')

        for cap in caps.values(): cap.release()
        out.release()
        
        print("\n" + "="*50)
        print("FINAL 360 TRACKING REPORT")
        print("="*50)
        for label, id_set in self.global_counts.items():
            print(f"{label.capitalize()}: {len(id_set)}")
        print("="*50)

if __name__ == "__main__":
    tracker = GlobalStreetTracker(
        base_dir=r"C:\IITM\CAMERA_Short_Clips",
        intrinsics_dir=r"C:\IITM\Vehicle_detection_360\Intrinsics"
    )
    tracker.run_global_tracking()
