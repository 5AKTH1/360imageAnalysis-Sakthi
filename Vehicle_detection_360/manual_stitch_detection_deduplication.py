import os
import cv2
import numpy as np
import json
from ultralytics import YOLO

class GlobalStreetTracker:
    def __init__(self, base_dir, intrinsics_dir, panorama_path, model_path='yolov8n.pt'):
        # 1. Initialize Model
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        
        # 2. Setup Paths
        self.base_dir = base_dir
        self.intrinsics_dir = intrinsics_dir
        self.panorama_path = panorama_path
        
        # 3. Lens Configuration (Counter-Clockwise Layout)
        # 1(Top-Left), 2(Left), 3(Back-Left), 4(Back-Right), 5(Right), 6(Top-Right)
        self.LENS_OFFSETS = {
            'lens1': -30,  'lens2': -90,  'lens3': -150, 
            'lens4': 150,  'lens5': 90,   'lens6': 30
        }
        
        # 4. Load Intrinsics
        self.lens_params = {}
        self._load_all_intrinsics()
        
        # 5. Tracking State
        self.global_registry = {}      # Maps { 'lens1_5': global_id_101 }
        self.next_global_id = 1
        
        # Re-ID Memory System
        # Stores: {gid: {'yaw': 120, 'label': 'car', 'last_seen_frame': 500}}
        self.track_history = {}        
        self.MEMORY_LIMIT = 60         # How long to remember a lost car (60 frames = ~2 sec)
        self.current_frame_idx = 0
        
        # Final Stats
        self.global_counts = {
            'car': set(), 'bus': set(), 'truck': set(), 
            'motorcycle': set(), 'auto_rickshaw': set()
        }

    def _load_all_intrinsics(self):
        """Loads the Pinhole K matrices for all 6 lenses."""
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
            else:
                print(f"[WARNING] Intrinsic file not found: {json_path}")

    def pixel_to_global_spherical(self, x, y, lens_key):
        """
        Converts 2D pixel (u,v) -> Global 3D Angles (Yaw, Pitch)
        """
        params = self.lens_params[lens_key]
        K = params['K']
        cx, cy, fx, fy = K[0, 2], K[1, 2], K[0, 0], K[1, 1]
        
        # 1. Back-project Pixel to Local 3D Ray
        ray_x = (x - cx) / fx
        ray_y = (y - cy) / fy
        ray_z = 1.0 # Forward vector
        
        # 2. Calculate Local Angles
        # Yaw = angle on horizontal plane (XZ)
        local_yaw = np.degrees(np.arctan2(ray_x, ray_z))
        
        # Pitch = angle of elevation
        d_xz = np.sqrt(ray_x**2 + ray_z**2)
        local_pitch = np.degrees(np.arctan2(ray_y, d_xz))
        
        # 3. Convert to Global Angles
        # Add the lens offset to Yaw
        global_yaw = (local_yaw + self.LENS_OFFSETS[lens_key]) % 360
        
        # Invert Pitch because image Y is Down, but world Y is Up
        global_pitch = -local_pitch 
        
        return global_yaw, global_pitch

    def resolve_identity(self, lens_key, local_id, current_yaw, label):
        """
        The 'Brain' of the tracker.
        Decides if a detection is a New Car, an Existing Active Car, or a Re-appearing Lost Car.
        """
        # 1. Check if this exact Local ID is already linked
        reg_key = f"{lens_key}_{local_id}"
        if reg_key in self.global_registry:
            g_id = self.global_registry[reg_key]
            # Refresh memory since we saw it
            if g_id in self.track_history:
                self.track_history[g_id]['yaw'] = current_yaw
                self.track_history[g_id]['last_seen_frame'] = self.current_frame_idx
            return g_id

        # 2. Search Memory for a match (Spatial & Temporal check)
        best_match_id = None
        min_dist = 1000
        
        for g_id, data in self.track_history.items():
            # Filter by Class
            if data['label'] != label: continue
            
            # Filter by Time (Don't resurrect ancient tracks)
            frames_since_seen = self.current_frame_idx - data['last_seen_frame']
            if frames_since_seen > self.MEMORY_LIMIT: continue 
            
            # Filter by Space (Is it near where we last saw it?)
            # Handle 359 -> 1 degree wrap-around logic
            diff = abs(data['yaw'] - current_yaw)
            diff = min(diff, 360 - diff) 
            
            # Threshold: 12 degrees (Generous buffer for overlap/motion)
            if diff < 12.0:
                if diff < min_dist:
                    min_dist = diff
                    best_match_id = g_id
        
        # 3. Match Found -> Link it
        if best_match_id is not None:
            self.global_registry[reg_key] = best_match_id
            # Update the global record with new position
            self.track_history[best_match_id]['yaw'] = current_yaw
            self.track_history[best_match_id]['last_seen_frame'] = self.current_frame_idx
            return best_match_id
        
        # 4. No Match -> Create New Global ID
        new_g_id = self.next_global_id
        self.next_global_id += 1
        
        self.global_registry[reg_key] = new_g_id
        self.track_history[new_g_id] = {
            'yaw': current_yaw, 
            'label': label, 
            'last_seen_frame': self.current_frame_idx
        }
        return new_g_id

    def run_global_tracking(self):
        # --- SETUP STREAMS ---
        caps = {}
        for i in range(1, 7):
            path = os.path.join(self.base_dir, f"lens {i}", f"undistorted_lens{i}.mp4")
            if os.path.exists(path):
                caps[f"lens{i}"] = cv2.VideoCapture(path)
            else:
                print(f"[ERROR] Could not find source video: {path}")
                return
        
        if not os.path.exists(self.panorama_path):
            print(f"[ERROR] Panorama video not found: {self.panorama_path}")
            return
            
        cap_pano = cv2.VideoCapture(self.panorama_path)
        
        # Setup Output
        CANVAS_W = int(cap_pano.get(cv2.CAP_PROP_FRAME_WIDTH))
        CANVAS_H = int(cap_pano.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = "Final_LateFusion_ReID_LargeFont.mp4"
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (CANVAS_W, CANVAS_H))
        
        print(f"Tracking Started. Resolution: {CANVAS_W}x{CANVAS_H}")
        print("Press Ctrl+C to stop safely.")

        try:
            while True:
                # 1. Read Panorama (Canvas)
                ret_p, frame_pano = cap_pano.read()
                if not ret_p: break
                
                # Use panorama as background
                equi_map = frame_pano 

                # 2. Read Source Feeds (Brain)
                frames = {}
                active_feeds = 0
                for k, cap in caps.items():
                    r, f = cap.read()
                    if r: 
                        frames[k] = f
                        active_feeds += 1
                
                if active_feeds < 6: 
                    print("End of source videos.")
                    break

                self.current_frame_idx += 1
                current_frame_objects = [] 

                # 3. DETECT & TRACK
                for lens_key, frame in frames.items():
                    # Run YOLO on high-quality source frame
                    results = self.model.track(frame, persist=True, verbose=False, conf=0.3, tracker="bytetrack.yaml")[0]
                    
                    if results.boxes.id is not None:
                        boxes = results.boxes.xyxy.cpu().numpy()
                        track_ids = results.boxes.id.int().cpu().tolist()
                        clss = results.boxes.cls.int().cpu().tolist()

                        for box, local_id, cls in zip(boxes, track_ids, clss):
                            label = self.model.names[cls]
                            if label not in self.global_counts: continue

                            # Get Box Center
                            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                            
                            # Calculate 360 Coordinates
                            g_yaw, g_pitch = self.pixel_to_global_spherical(cx, cy, lens_key)
                            
                            # RESOLVE IDENTITY (Merge Duplicates)
                            global_id = self.resolve_identity(lens_key, local_id, g_yaw, label)
                            
                            # Store for Drawing
                            current_frame_objects.append({
                                'yaw': g_yaw, 'pitch': g_pitch, 'label': label,
                                'id': global_id, 'lens': lens_key
                            })
                            
                            # Add to counts
                            self.global_counts[label].add(global_id)
                
                # 4. MEMORY CLEANUP (Every 100 frames)
                if self.current_frame_idx % 100 == 0:
                    keys_to_del = []
                    for gid, data in self.track_history.items():
                        if self.current_frame_idx - data['last_seen_frame'] > self.MEMORY_LIMIT:
                            keys_to_del.append(gid)
                    for k in keys_to_del: del self.track_history[k]

                # 5. VISUALIZE ON PANORAMA
                for obj in current_frame_objects:
                    # Map Yaw (X) - Linear
                    map_x = int((obj['yaw'] / 360.0) * CANVAS_W)
                    
                    # Map Pitch (Y) - Equirectangular Projection
                    # v = H * (0.5 - pitch_rad / pi)
                    pitch_rad = np.radians(obj['pitch'])
                    map_y = int(CANVAS_H * (0.5 - (pitch_rad / np.pi)))
                    
                    # Safety Clip
                    map_y = np.clip(map_y, 0, CANVAS_H-1)
                    map_x = np.clip(map_x, 0, CANVAS_W-1)
                    
                    # Color by Source Lens (Visual Proof of Fusion)
                    lens_idx = int(obj['lens'][-1])
                    colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0)]
                    color = colors[lens_idx-1]
                    
                    # Draw larger circle
                    cv2.circle(equi_map, (map_x, map_y), 12, color, -1)
                    
                    # --- FONT SIZE FIX IS HERE ---
                    # Increased fontScale to 1.3 and thickness to 3
                    cv2.putText(equi_map, f"{obj['label']}-{obj['id']}", (map_x+15, map_y+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,255), 3)

                # 6. DRAW STATS OVERLAY
                cv2.rectangle(equi_map, (20, 20), (400, 250), (0,0,0), -1)
                cv2.putText(equi_map, "GLOBAL UNIQUE COUNTS", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                y_off = 90
                for label, id_set in self.global_counts.items():
                    cv2.putText(equi_map, f"{label.capitalize()}: {len(id_set)}", (40, y_off), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                    y_off += 35

                out.write(equi_map)
                print(f"Processed Frame {self.current_frame_idx} | Active Vehicles: {len(current_frame_objects)}", end='\r')

        except KeyboardInterrupt:
            print("\nStopping...")

        # Cleanup
        for c in caps.values(): c.release()
        cap_pano.release()
        out.release()
        
        print("\n" + "="*60)
        print(f"DONE! Video Saved: {out_path}")
        print("FINAL COUNTS:")
        for label, id_set in self.global_counts.items():
            print(f"  - {label.capitalize()}: {len(id_set)}")
        print("="*60)

if __name__ == "__main__":
    # --- UPDATE THESE PATHS ---
    BASE_DIR = r"C:\IITM\CAMERA_Short_Clips"
    INTRINSICS_DIR = r"C:\IITM\Vehicle_detection_360\Intrinsics"
    
    # This must match the output filename from your stitcher script
    PANO_PATH = os.path.join(BASE_DIR, "Final_Clean_Panorama_Corrected.mp4")
    
    tracker = GlobalStreetTracker(BASE_DIR, INTRINSICS_DIR, PANO_PATH)
    tracker.run_global_tracking()