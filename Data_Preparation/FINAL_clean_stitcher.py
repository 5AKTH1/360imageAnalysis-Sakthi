'''#FINAL SIFT-BASED STITCHER WITH OPTIMIZED GEOMETRY & ZOOM VALUES FOR ALL LENSES
import os
import cv2
import numpy as np
import json

class SIFTInsta360Stitcher:
    def __init__(self, base_dir, intrinsics_dir):
        self.base_dir = base_dir
        self.intrinsics_dir = intrinsics_dir
        
        # --- CONFIGURATION ---
        self.W = 3328
        self.H = 1664
        
        # --- 1. FINAL OPTIMIZED ZOOM VALUES ---
        self.LENS_ZOOMS = [
            1.078,
            1.127,
            1.105,
            1.044,
            1.118,
            1.08,
        ]
        
        # --- 2. FINAL OPTIMIZED GEOMETRY (Yaw, Pitch) ---
        self.LENS_ORIENTATIONS = [
            (-59.8471, 0.0000),   # Lens 1 
            (-120.0512, 0.0000),  # Lens 2
            (180.1485, 0.0000),   # Lens 3
            (119.6971, 0.0000),   # Lens 4
            (60.4603, 0.0000),    # Lens 5
            (-0.1470, 0.0000),    # Lens 6
        ]
        
        self.maps_x = []
        self.maps_y = []
        self.weights = []
        self.Ks = []
        self.img_dims = []

        self._load_intrinsics()
        print(f"Initializing Geometry ({self.W}x{self.H})...")
        self._init_geometry_and_weights()

    def _load_intrinsics(self):
        for i in range(1, 7):
            path = os.path.join(self.intrinsics_dir, f"calibration_pinhole_lens{i}.json")
            zoom_factor = self.LENS_ZOOMS[i-1]

            if not os.path.exists(path):
                print(f"Warning: {path} not found. Using generic.")
                K = np.array([[1000,0,960],[0,1000,540],[0,0,1]], dtype=np.float32)
                self.img_dims.append((1920,1080))
                K[0,0] *= zoom_factor
                K[1,1] *= zoom_factor
                self.Ks.append(K)
            else:
                with open(path, 'r') as f:
                    data = json.load(f)
                K = np.array(data['K'], dtype=np.float32)
                K[0,0] *= zoom_factor 
                K[1,1] *= zoom_factor 
                self.Ks.append(K)
                self.img_dims.append(tuple(data['image_size']))

    def _init_geometry_and_weights(self):
        self.maps_x = []
        self.maps_y = []
        self.weights = []

        u_grid, v_grid = np.meshgrid(np.arange(self.W), np.arange(self.H))
        
        theta = (u_grid / self.W) * 2 * np.pi
        theta = (theta + np.pi) % (2 * np.pi) 

        phi = (np.pi / 2) - ((v_grid / self.H) * np.pi)
        
        x_s = np.cos(phi) * np.sin(theta)
        y_s = np.sin(phi) 
        z_s = np.cos(phi) * np.cos(theta)
        
        rays_global = np.dstack((x_s, -y_s, z_s)) 

        for i in range(6):
            yaw_deg, pitch_deg = self.LENS_ORIENTATIONS[i]
            
            yaw = np.radians(yaw_deg)
            pitch = np.radians(pitch_deg)
            
            Ry = np.array([
                [np.cos(yaw), 0, -np.sin(yaw)],
                [0, 1, 0],
                [np.sin(yaw), 0, np.cos(yaw)]
            ], dtype=np.float32)
            
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)]
            ], dtype=np.float32)
            
            R_inv = (Ry @ Rx).T
            
            rays_local = rays_global.reshape(-1, 3) @ R_inv
            
            z_vals = rays_local[:, 2]
            z_vals[z_vals <= 0.01] = 0.01 
            
            K = self.Ks[i]
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            
            u_src = (fx * (rays_local[:, 0] / z_vals)) + cx
            v_src = (fy * (rays_local[:, 1] / z_vals)) + cy
            
            map_x = u_src.reshape(self.H, self.W).astype(np.float32)
            map_y = v_src.reshape(self.H, self.W).astype(np.float32)
            
            src_w, src_h = self.img_dims[i]
            valid_mask = (map_x >= 0) & (map_x < src_w) & \
                         (map_y >= 0) & (map_y < src_h) & \
                         (rays_local[:, 2].reshape(self.H, self.W) > 0.1)
            
            dist_x = np.abs(map_x - (src_w / 2)) / (src_w / 2)
            dist_y = np.abs(map_y - (src_h / 2)) / (src_h / 2)
            
            linear_weight = 1.0 - np.maximum(dist_x, dist_y)
            linear_weight = np.clip(linear_weight, 0, 1)
            
            weight = np.power(linear_weight, 2.0) 
            weight[~valid_mask] = 0
            
            self.maps_x.append(map_x)
            self.maps_y.append(map_y)
            self.weights.append(weight.astype(np.float32))

    def find_shift_with_sift(self, img1, img2):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
            return 0
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
                
        if len(good) < 4:
            return 0
            
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        
        dx = pts1[:, 0] - pts2[:, 0]
        dy = pts1[:, 1] - pts2[:, 1]
        
        valid_matches = np.abs(dy) < 10 
        if np.sum(valid_matches) < 3: return 0
        
        return np.median(dx[valid_matches])

    def check_alignment_only(self, caps):
        """Runs SIFT one last time just to PRINT errors (does not change anything)."""
        print("\n--- VERIFYING ALIGNMENT (NO CORRECTION) ---")
        
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret: return
            frames.append(frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        pixels_per_deg = self.W / 360.0
        
        warped = []
        for i in range(6):
            w = cv2.remap(frames[i], self.maps_x[i], self.maps_y[i], cv2.INTER_LINEAR)
            warped.append(w)
            
        max_error = 0.0
        
        for i in range(6):
            next_i = (i + 1) % 6
            mask_curr = cv2.cvtColor(warped[i], cv2.COLOR_BGR2GRAY) > 10
            mask_next = cv2.cvtColor(warped[next_i], cv2.COLOR_BGR2GRAY) > 10
            overlap = mask_curr & mask_next
            
            if np.sum(overlap) == 0: continue
            
            y, x = np.where(overlap)
            min_x, max_x = np.min(x), np.max(x)
            if (max_x - min_x) < 10: continue
            
            strip_curr = warped[i][:, min_x:max_x]
            strip_next = warped[next_i][:, min_x:max_x]
            
            shift_px = self.find_shift_with_sift(strip_curr, strip_next)
            shift_deg = shift_px / pixels_per_deg
            
            if abs(shift_deg) > max_error: max_error = abs(shift_deg)
            
            print(f"  Lens {i+1} -> {next_i+1}: Final Error {shift_px:.1f}px ({shift_deg:.3f}°)")
            
        print(f"  >>> Max Error: {max_error:.3f}° (Target < 0.1°)")
        print("-" * 40 + "\n")

    def process_video(self):
        caps = []
        for i in range(1, 7):
            path = os.path.join(self.base_dir, f"lens {i}")
            v_files = [f for f in os.listdir(path) if f.startswith("undistorted_") and f.endswith(".mp4")]
            if not v_files: return
            caps.append(cv2.VideoCapture(os.path.join(path, v_files[0])))
            
        # --- VERIFY ALIGNMENT ---
        # This will print the errors before stitching starts
        self.check_alignment_only(caps)
            
        out_path = os.path.join(self.base_dir, "Final_SIFT_Shifted.mp4")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (self.W, self.H))
        
        print(f"Stitching to -> {out_path}")
        
        frame_idx = 0
        try:
            while True:
                frames = []
                for cap in caps:
                    ret, frame = cap.read()
                    if not ret: 
                        frames = None
                        break
                    frames.append(frame)
                
                if frames is None: break
                
                total_img = np.zeros((self.H, self.W, 3), dtype=np.float32)
                total_weight = np.zeros((self.H, self.W), dtype=np.float32)
                
                for i in range(6):
                    warped = cv2.remap(frames[i], self.maps_x[i], self.maps_y[i], cv2.INTER_LINEAR)
                    w = self.weights[i]
                    w_3c = np.dstack([w, w, w])
                    
                    total_img += warped.astype(np.float32) * w_3c
                    total_weight += w
                
                total_weight[total_weight == 0] = 1.0
                total_weight_3c = np.dstack([total_weight, total_weight, total_weight])
                
                final_pano = total_img / total_weight_3c
                final_pano = np.clip(final_pano, 0, 255).astype(np.uint8)
                
                out.write(final_pano)
                
                frame_idx += 1
                if frame_idx % 10 == 0: print(f"   Processed {frame_idx} frames...", end='\r')
                    
        except KeyboardInterrupt: pass
        for cap in caps: cap.release()
        out.release()
        print("\nDone!")

if __name__ == "__main__":
    BASE_DIR = r"C:\IITM\Results"
    INTRINSICS_DIR = r"C:\IITM\Vehicle_detection_360\Intrinsics"
    stitcher = SIFTInsta360Stitcher(BASE_DIR, INTRINSICS_DIR)
    stitcher.process_video()

'''

# AUTOMATED SIFT-ERROR CORRECTING STITCHER
import os
import cv2
import numpy as np
import json

class SIFTInsta360Stitcher:
    def __init__(self, base_dir, intrinsics_dir):
        self.base_dir = base_dir
        self.intrinsics_dir = intrinsics_dir
        
        # --- CONFIGURATION ---
        self.W = 3328
        self.H = 1664
        
        # --- PER-LENS ZOOM TUNING ---
        # (Using the "God Tier" values from our previous optimization)
        self.LENS_ZOOMS = [
            1.078,
            1.127,
            1.105,
            1.044,
            1.118,
            1.08,
        ]
        
        # [-60, -120, 180, 120, 60, 0] (Lens 6 is Front)
        self.LENS_ORIENTATIONS = [
            (-60, 0),    # Lens 1 
            (-120, 0),   # Lens 2
            (180, 0),    # Lens 3
            (120, 0),    # Lens 4
            (60, 0),     # Lens 5
            (0, 0)       # Lens 6 (Front)
        ]
        
        self.maps_x = []
        self.maps_y = []
        self.weights = []
        self.Ks = []
        self.img_dims = []

        self._load_intrinsics()
        print(f"Initializing Geometry ({self.W}x{self.H})...")
        self._init_geometry_and_weights()

    def _load_intrinsics(self):
        for i in range(1, 7):
            path = os.path.join(self.intrinsics_dir, f"calibration_pinhole_lens{i}.json")
            zoom_factor = self.LENS_ZOOMS[i-1]

            if not os.path.exists(path):
                print(f"Warning: {path} not found. Using generic.")
                K = np.array([[1000,0,960],[0,1000,540],[0,0,1]], dtype=np.float32)
                self.img_dims.append((1920,1080))
                K[0,0] *= zoom_factor
                K[1,1] *= zoom_factor
                self.Ks.append(K)
            else:
                with open(path, 'r') as f:
                    data = json.load(f)
                K = np.array(data['K'], dtype=np.float32)
                print(f"Lens {i}: Applying Zoom Factor {zoom_factor}")
                K[0,0] *= zoom_factor # fx
                K[1,1] *= zoom_factor # fy
                self.Ks.append(K)
                self.img_dims.append(tuple(data['image_size']))

    def _init_geometry_and_weights(self):
        self.maps_x = []
        self.maps_y = []
        self.weights = []

        # 1. Create Grid
        u_grid, v_grid = np.meshgrid(np.arange(self.W), np.arange(self.H))
        
        # 2. Spherical Mapping
        theta = (u_grid / self.W) * 2 * np.pi
        theta = (theta + np.pi) % (2 * np.pi) 

        phi = (np.pi / 2) - ((v_grid / self.H) * np.pi)
        
        x_s = np.cos(phi) * np.sin(theta)
        y_s = np.sin(phi) 
        z_s = np.cos(phi) * np.cos(theta)
        
        rays_global = np.dstack((x_s, -y_s, z_s)) 

        for i in range(6):
            yaw_deg, pitch_deg = self.LENS_ORIENTATIONS[i]
            
            yaw = np.radians(yaw_deg)
            pitch = np.radians(pitch_deg)
            
            Ry = np.array([
                [np.cos(yaw), 0, -np.sin(yaw)],
                [0, 1, 0],
                [np.sin(yaw), 0, np.cos(yaw)]
            ], dtype=np.float32)
            
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)]
            ], dtype=np.float32)
            
            R_inv = (Ry @ Rx).T
            
            rays_local = rays_global.reshape(-1, 3) @ R_inv
            
            z_vals = rays_local[:, 2]
            z_vals[z_vals <= 0.01] = 0.01 
            
            K = self.Ks[i]
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            
            u_src = (fx * (rays_local[:, 0] / z_vals)) + cx
            v_src = (fy * (rays_local[:, 1] / z_vals)) + cy
            
            map_x = u_src.reshape(self.H, self.W).astype(np.float32)
            map_y = v_src.reshape(self.H, self.W).astype(np.float32)
            
            src_w, src_h = self.img_dims[i]
            valid_mask = (map_x >= 0) & (map_x < src_w) & \
                         (map_y >= 0) & (map_y < src_h) & \
                         (rays_local[:, 2].reshape(self.H, self.W) > 0.1)
            
            dist_x = np.abs(map_x - (src_w / 2)) / (src_w / 2)
            dist_y = np.abs(map_y - (src_h / 2)) / (src_h / 2)
            
            linear_weight = 1.0 - np.maximum(dist_x, dist_y)
            linear_weight = np.clip(linear_weight, 0, 1)
            
            weight = np.power(linear_weight, 2.0) 
            weight[~valid_mask] = 0
            
            self.maps_x.append(map_x)
            self.maps_y.append(map_y)
            self.weights.append(weight.astype(np.float32))

    def find_shift_with_sift(self, img1, img2):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
            return 0
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
                
        if len(good) < 4:
            return 0
            
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        
        dx = pts1[:, 0] - pts2[:, 0]
        dy = pts1[:, 1] - pts2[:, 1]
        
        valid_matches = np.abs(dy) < 10 
        if np.sum(valid_matches) < 3: return 0
        
        return np.median(dx[valid_matches])

    def calibrate_angles_with_sift(self, caps):
        # --- SAFE AUTO-CALIBRATION CONFIG ---
        MAX_ITERATIONS = 50       # Give it more steps since we are moving slower
        ERROR_THRESHOLD_DEG = 0.4 # Target accuracy
        LEARNING_RATE = 0.25      # Lower gain = More stable (prevents explosion)
        MAX_CORRECTION_STEP = 1.5 # Safety limit: Never jump more than 1.5 deg at once
        
        print(f"\n--- STARTING AUTO-CALIBRATION (Safe Mode) ---")
        
        # 1. Load frames once
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret: return
            frames.append(frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        pixels_per_deg = self.W / 360.0

        for iteration in range(MAX_ITERATIONS):
            print(f"\n[Iteration {iteration + 1}/{MAX_ITERATIONS}] Checking alignment...")
            
            # 2. Warp with CURRENT geometry
            warped = []
            for i in range(6):
                w = cv2.remap(frames[i], self.maps_x[i], self.maps_y[i], cv2.INTER_LINEAR)
                warped.append(w)
                
            new_orientations = list(self.LENS_ORIENTATIONS)
            max_error_deg = 0.0
            total_corrections = 0
            
            # 3. Check all seams
            for i in range(6):
                next_i = (i + 1) % 6
                
                mask_curr = cv2.cvtColor(warped[i], cv2.COLOR_BGR2GRAY) > 10
                mask_next = cv2.cvtColor(warped[next_i], cv2.COLOR_BGR2GRAY) > 10
                overlap = mask_curr & mask_next
                
                if np.sum(overlap) == 0: continue
                
                y, x = np.where(overlap)
                min_x, max_x = np.min(x), np.max(x)
                if (max_x - min_x) < 10: continue
                
                strip_curr = warped[i][:, min_x:max_x]
                strip_next = warped[next_i][:, min_x:max_x]
                
                # Find Shift
                shift_px = self.find_shift_with_sift(strip_curr, strip_next)
                shift_deg = shift_px / pixels_per_deg
                
                # Track Max Error
                if abs(shift_deg) > max_error_deg:
                    max_error_deg = abs(shift_deg)
                
                status = "OK" if abs(shift_deg) <= ERROR_THRESHOLD_DEG else "FIXING"
                # Only print significant errors to reduce clutter
                if abs(shift_deg) > 0.1:
                    print(f"  Lens {i+1} -> {next_i+1}: Error {shift_px:.1f}px ({shift_deg:.2f}°) [{status}]")
                
                # Apply Correction
                if abs(shift_deg) > 0.1: # Only correct if error is measurable
                    old_yaw, pitch = new_orientations[next_i]
                    
                    # --- THE FIX: FLIPPED SIGN & CLAMPING ---
                    # We flip the sign (+) because the previous (-) caused divergence.
                    correction = shift_deg * LEARNING_RATE
                    
                    # Safety Clamp
                    correction = np.clip(correction, -MAX_CORRECTION_STEP, MAX_CORRECTION_STEP)
                    
                    new_yaw = old_yaw + correction 
                    new_orientations[next_i] = (new_yaw, pitch)
                    total_corrections += 1

            # 4. Check Convergence
            print(f"  >>> Max Error: {max_error_deg:.3f}°")
            
            if max_error_deg <= ERROR_THRESHOLD_DEG:
                print(f"  >>> SUCCESS! All errors are within range [-{ERROR_THRESHOLD_DEG}, {ERROR_THRESHOLD_DEG}].")
                break
            
            if total_corrections == 0 and max_error_deg > ERROR_THRESHOLD_DEG:
                print("  >>> No clear SIFT matches found to improve further. Stopping.")
                break

            # 5. Apply Updates
            if iteration < MAX_ITERATIONS - 1:
                self.LENS_ORIENTATIONS = new_orientations
                self._init_geometry_and_weights()
            else:
                print("  >>> Reached Max Iterations.")

        # --- AFTER THE LOOP FINISHES, PRINT THE FINAL CONFIG ---
        print("\n" + "="*40)
        print("   FINAL OPTIMIZED CONFIGURATION")
        print("   (Copy these into your __init__)")
        print("="*40)
        
        print("self.LENS_ZOOMS = [")
        for z in self.LENS_ZOOMS:
            print(f"    {z},")
        print("]")
        
        print("\nself.LENS_ORIENTATIONS = [")
        for i, (yaw, pitch) in enumerate(self.LENS_ORIENTATIONS):
            print(f"    ({yaw:.4f}, {pitch:.4f}), # Lens {i+1}")
        print("]")
        print("="*40 + "\n")

    def check_drift_and_update(self, frames):
        """Pauses every 150 frames to nudge the alignment back to perfect."""
        pixels_per_deg = self.W / 360.0
        TARGET_MAX_ERROR = 0.40   # Don't stop until error is tiny
        MAX_ITERATIONS = 5        # Quick nudge, not a full calibration
        
        best_orientations = list(self.LENS_ORIENTATIONS)
        best_error = float('inf')
        
        for iteration in range(MAX_ITERATIONS):
            max_drift = 0.0
            new_orientations = list(self.LENS_ORIENTATIONS)
            
            warped = [cv2.remap(frames[i], self.maps_x[i], self.maps_y[i], cv2.INTER_LINEAR) for i in range(6)]
            
            for i in range(6):
                next_i = (i + 1) % 6
                mask_curr = cv2.cvtColor(warped[i], cv2.COLOR_BGR2GRAY) > 10
                mask_next = cv2.cvtColor(warped[next_i], cv2.COLOR_BGR2GRAY) > 10
                overlap = mask_curr & mask_next
                
                if np.sum(overlap) == 0: continue
                y, x = np.where(overlap)
                min_x, max_x = np.min(x), np.max(x)
                
                # Check middle strip for drift
                strip_curr = warped[i][:, min_x:max_x]
                strip_next = warped[next_i][:, min_x:max_x]
                
                shift_px = self.find_shift_with_sift(strip_curr, strip_next)
                shift_deg = shift_px / pixels_per_deg
                
                if abs(shift_deg) > 1.2: # Ignore massive jumps (Likely an occlusion/car passing)
                    continue
                
                if abs(shift_deg) > max_drift:
                    max_drift = abs(shift_deg)
                
                # Nudge the lens
                if abs(shift_deg) > 0.05:
                    old_yaw, pitch = new_orientations[next_i]
                    # Very small correction to maintain stability
                    correction = np.clip(shift_deg * 0.35, -0.5, 0.5)
                    new_orientations[next_i] = (old_yaw + correction, pitch)

            # Check if this iteration helped
            if max_drift < best_error:
                best_error = max_drift
                best_orientations = list(self.LENS_ORIENTATIONS)
            elif max_drift > best_error + 0.05:
                # If error got worse, rollback and stop
                self.LENS_ORIENTATIONS = best_orientations
                self._init_geometry_and_weights()
                break

            self.LENS_ORIENTATIONS = new_orientations
            self._init_geometry_and_weights()
            
            if max_drift <= TARGET_MAX_ERROR: break

    def process_video(self):
        caps = []
        for i in range(1, 7):
            path = self.base_dir
            v_files = f"undistorted_lens{i}.mp4"
            caps.append(cv2.VideoCapture(os.path.join(path, v_files)))
            
        # Initial Calibration (Frame 0)
        self.calibrate_angles_with_sift(caps)
            
        out_path = os.path.join(self.base_dir, "Final_Dynamic_Stitch.mp4")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (self.W, self.H))
        
        frame_idx = 0
        try:
            while True:
                frames = []
                for cap in caps:
                    ret, frame = cap.read()
                    if not ret: 
                        frames = None
                        break
                    frames.append(frame)
                
                if frames is None: break
                
                # --- NEW LOGIC: DRIFT CHECK EVERY 150 FRAMES (5 seconds) ---
                if frame_idx > 0 and frame_idx % 150 == 0:
                    print(f"\n   [Frame {frame_idx}] Pausing to check for thermal drift...")
                    self.check_drift_and_update(frames)
                    print(f"   [Frame {frame_idx}] Check complete. Resuming video processing...")
                
                # Perform the actual stitch using current optimized math
                total_img = np.zeros((self.H, self.W, 3), dtype=np.float32)
                total_weight = np.zeros((self.H, self.W), dtype=np.float32)
                
                for i in range(6):
                    warped = cv2.remap(frames[i], self.maps_x[i], self.maps_y[i], cv2.INTER_LINEAR)
                    w = self.weights[i]
                    total_img += warped.astype(np.float32) * np.dstack([w, w, w])
                    total_weight += w
                
                total_weight[total_weight == 0] = 1.0
                final_pano = np.clip(total_img / np.dstack([total_weight]*3), 0, 255).astype(np.uint8)
                out.write(final_pano)
                
                frame_idx += 1
                if frame_idx % 10 == 0: print(f"   Processed {frame_idx} frames...", end='\r')
                    
        except KeyboardInterrupt: pass
        for cap in caps: cap.release()
        out.release()
        print("\nDone!")
        
if __name__ == "__main__":
    BASE_DIR = r"C:\IITM\DATA_RETRIEVAL_TASK\ClipPipeline_2026-02-02_09-36-58_F43387_D2p0\UNDISTORTED"
    INTRINSICS_DIR = r"C:\IITM\DATA_RETRIEVAL_TASK\ClipPipeline_2026-02-02_09-36-58_F43387_D2p0\INTRINSICS"
    stitcher = SIFTInsta360Stitcher(BASE_DIR, INTRINSICS_DIR)
    stitcher.process_video()
'''
# LOCKED SIFT-INITIALIZED STITCHER
import os
import cv2
import numpy as np
import json

class SIFTInsta360Stitcher:
    def __init__(self, base_dir, intrinsics_dir):
        self.base_dir = base_dir
        self.intrinsics_dir = intrinsics_dir
        
        # --- CONFIGURATION ---
        self.W = 3328
        self.H = 1664
        
        # --- PER-LENS ZOOM TUNING ---
        self.LENS_ZOOMS = [1.078, 1.127, 1.105, 1.044, 1.118, 1.08]
        
        # Initial approximate orientations
        self.LENS_ORIENTATIONS = [
            (-60, 0), (-120, 0), (180, 0), 
            (120, 0), (60, 0), (0, 0)
        ]
        
        self.maps_x = []
        self.maps_y = []
        self.weights = []
        self.Ks = []
        self.img_dims = []

        self._load_intrinsics()
        print(f"Initializing Geometry ({self.W}x{self.H})...")
        self._init_geometry_and_weights()

    def _load_intrinsics(self):
        self.Ks = []
        self.img_dims = []
        for i in range(1, 7):
            path = os.path.join(self.intrinsics_dir, f"calibration_pinhole_lens{i}.json")
            zoom_factor = self.LENS_ZOOMS[i-1]

            if not os.path.exists(path):
                K = np.array([[1000, 0, 540], [0, 1000, 960], [0, 0, 1]], dtype=np.float32)
                self.img_dims.append((1080, 1920))
            else:
                with open(path, 'r') as f:
                    data = json.load(f)
                K = np.array(data['K'], dtype=np.float32)
                self.img_dims.append(tuple(data['image_size']))
            
            K[0,0] *= zoom_factor
            K[1,1] *= zoom_factor
            self.Ks.append(K)

    def _init_geometry_and_weights(self):
        self.maps_x, self.maps_y, self.weights = [], [], []
        u_grid, v_grid = np.meshgrid(np.arange(self.W), np.arange(self.H))
        theta = (u_grid / self.W) * 2 * np.pi
        theta = (theta + np.pi) % (2 * np.pi) 
        phi = (np.pi / 2) - ((v_grid / self.H) * np.pi)
        
        x_s = np.cos(phi) * np.sin(theta)
        y_s = np.sin(phi) 
        z_s = np.cos(phi) * np.cos(theta)
        rays_global = np.dstack((x_s, -y_s, z_s)) 

        for i in range(6):
            yaw_deg, pitch_deg = self.LENS_ORIENTATIONS[i]
            yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
            Ry = np.array([[np.cos(yaw), 0, -np.sin(yaw)],[0, 1, 0],[np.sin(yaw), 0, np.cos(yaw)]], dtype=np.float32)
            Rx = np.array([[1, 0, 0],[0, np.cos(pitch), -np.sin(pitch)],[0, np.sin(pitch), np.cos(pitch)]], dtype=np.float32)
            R_inv = (Ry @ Rx).T
            rays_local = rays_global.reshape(-1, 3) @ R_inv
            z_vals = np.maximum(rays_local[:, 2], 0.01)
            u_src = (self.Ks[i][0,0] * (rays_local[:, 0] / z_vals)) + self.Ks[i][0,2]
            v_src = (self.Ks[i][1,1] * (rays_local[:, 1] / z_vals)) + self.Ks[i][1,2]
            mx, my = u_src.reshape(self.H, self.W).astype(np.float32), v_src.reshape(self.H, self.W).astype(np.float32)
            src_w, src_h = self.img_dims[i]
            valid = (mx >= 0) & (mx < src_w) & (my >= 0) & (my < src_h) & (rays_local[:, 2].reshape(self.H, self.W) > 0.1)
            dist_x, dist_y = np.abs(mx - (src_w / 2)) / (src_w / 2), np.abs(my - (src_h / 2)) / (src_h / 2)
            weight = np.power(np.clip(1.0 - np.maximum(dist_x, dist_y), 0, 1), 2.0)
            weight[~valid] = 0
            self.maps_x.append(mx); self.maps_y.append(my); self.weights.append(weight.astype(np.float32))

    def find_shift_with_sift(self, img1, img2):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5: return 0
        matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 4: return 0
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        valid = np.abs(pts1[:, 1] - pts2[:, 1]) < 10 
        return np.median(pts1[valid, 0] - pts2[valid, 0]) if np.any(valid) else 0

    def calibrate_angles_with_sift(self, caps):
        MAX_ITERATIONS = 30  # Increased to give more room for convergence
        LEARNING_RATE = 0.25
        ERROR_TARGET = 0.40  # Your required max error
        pixels_per_deg = self.W / 360.0
        
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            frames.append(frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        print("\n--- INITIAL CALIBRATION (Target Error: <0.4°) ---")
        for iteration in range(MAX_ITERATIONS):
            warped = [cv2.remap(frames[i], self.maps_x[i], self.maps_y[i], cv2.INTER_LINEAR) for i in range(6)]
            new_orientations = list(self.LENS_ORIENTATIONS)
            max_error_deg = 0.0
            
            for i in range(6):
                next_i = (i + 1) % 6
                mask = (cv2.cvtColor(warped[i], cv2.COLOR_BGR2GRAY) > 10) & (cv2.cvtColor(warped[next_i], cv2.COLOR_BGR2GRAY) > 10)
                if np.sum(mask) < 200: continue
                y, x = np.where(mask)
                
                shift_px = self.find_shift_with_sift(warped[i][:, np.min(x):np.max(x)], warped[next_i][:, np.min(x):np.max(x)])
                shift_deg = shift_px / pixels_per_deg
                
                # Track the highest error seen in this iteration
                if abs(shift_deg) > max_error_deg:
                    max_error_deg = abs(shift_deg)
                
                # Apply correction if above target
                if abs(shift_deg) > 0.05:
                    new_orientations[next_i] = (new_orientations[next_i][0] + np.clip(shift_deg * LEARNING_RATE, -1.5, 1.5), 0)
            
            self.LENS_ORIENTATIONS = new_orientations
            self._init_geometry_and_weights()
            
            print(f"   Iter {iteration+1:2d} | Max Alignment Error: {max_error_deg:.4f}°")
            
            # EARLY EXIT: If we are under 0.4, we are done
            if max_error_deg < ERROR_TARGET and iteration > 5:
                print(f"--- SUCCESS: Convergence reached at {max_error_deg:.4f}° ---")
                break
                
        print("Initial Calibration Complete. Geometry Locked.")

    def process_video(self):
        caps = [cv2.VideoCapture(os.path.join(self.base_dir, f"lens {i}", f"undistorted_lens{i}.mp4")) for i in range(1, 7)]
        self.calibrate_angles_with_sift(caps)
        
        fps = caps[0].get(cv2.CAP_PROP_FPS) or 29.90
        self.CROP_TOP, self.CROP_BOTTOM = 350, 500
        cropped_H = self.H - self.CROP_TOP - self.CROP_BOTTOM
        
        out_path = os.path.join(self.base_dir, "Final_Locked_Stitch.mp4")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (self.W, cropped_H))
        
        print(f"Stitching to -> {out_path} at {fps:.2f} FPS")
        
        frame_idx = 0
        try:
            while True:
                frames = []
                for cap in caps:
                    ret, frame = cap.read()
                    if not ret: break
                    frames.append(frame)
                if len(frames) < 6: break
                
                total_img, total_w = np.zeros((self.H, self.W, 3), dtype=np.float32), np.zeros((self.H, self.W), dtype=np.float32)
                for i in range(6):
                    warped = cv2.remap(frames[i], self.maps_x[i], self.maps_y[i], cv2.INTER_LINEAR)
                    total_img += warped.astype(np.float32) * np.dstack([self.weights[i]]*3)
                    total_w += self.weights[i]
                
                final = (total_img / np.dstack([np.maximum(total_w, 1.0)]*3)).astype(np.uint8)
                out.write(final[self.CROP_TOP : self.H - self.CROP_BOTTOM, :])
                
                frame_idx += 1
                if frame_idx % 50 == 0: print(f"   Processed {frame_idx} frames... (Geometry Locked)", end='\r')
                    
        except KeyboardInterrupt: pass
        for cap in caps: cap.release()
        out.release()
        print(f"\nDone! Processed {frame_idx} frames.")

if __name__ == "__main__":
    stitcher = SIFTInsta360Stitcher(r"C:\IITM\Results", r"C:\IITM\Results\FINAL_Intrinsics")
    stitcher.process_video()
    '''