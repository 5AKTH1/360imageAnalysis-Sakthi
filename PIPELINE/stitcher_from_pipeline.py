import os
import cv2
import numpy as np
import json

class SIFTInsta360ImageStitcher:
    def __init__(self, intrinsics_dir):
        self.intrinsics_dir = intrinsics_dir
        
        # --- CONFIGURATION ---
        self.W = 3328
        self.H = 1664
        
        # --- PER-LENS ZOOM TUNING ---
        self.LENS_ZOOMS = [
            1.078, 1.127, 1.105, 1.044, 1.118, 1.08,
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
        print(f"[INFO] Initializing Geometry ({self.W}x{self.H})...")
        self._init_geometry_and_weights()

    def _load_intrinsics(self):
        for i in range(1, 7):
            path = os.path.join(self.intrinsics_dir, f"calibration_pinhole_lens{i}.json")
            zoom_factor = self.LENS_ZOOMS[i-1]

            if not os.path.exists(path):
                print(f"[WARNING] {path} not found. Using generic.")
                K = np.array([[1000,0,960],[0,1000,540],[0,0,1]], dtype=np.float32)
                self.img_dims.append((1920,1080))
                K[0,0] *= zoom_factor
                K[1,1] *= zoom_factor
                self.Ks.append(K)
            else:
                with open(path, 'r') as f:
                    data = json.load(f)
                K = np.array(data['K'], dtype=np.float32)
                print(f"[INFO] Lens {i}: Applying Zoom Factor {zoom_factor}")
                K[0,0] *= zoom_factor # fx
                K[1,1] *= zoom_factor # fy
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
                
        if len(good) < 4: return 0
            
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        
        dx = pts1[:, 0] - pts2[:, 0]
        dy = pts1[:, 1] - pts2[:, 1]
        
        valid_matches = np.abs(dy) < 10 
        if np.sum(valid_matches) < 3: return 0
        
        return np.median(dx[valid_matches])

    def calibrate_angles_with_sift(self, frames):
        # Taking frames directly instead of reading from VideoCapture
        MAX_ITERATIONS = 50       
        ERROR_THRESHOLD_DEG = 0.3
        LEARNING_RATE = 0.25      
        MAX_CORRECTION_STEP = 1.5 
        
        print(f"\n--- STARTING AUTO-CALIBRATION (Safe Mode) ---")
        pixels_per_deg = self.W / 360.0

        for iteration in range(MAX_ITERATIONS):
            warped = []
            for i in range(6):
                w = cv2.remap(frames[i], self.maps_x[i], self.maps_y[i], cv2.INTER_LINEAR)
                warped.append(w)
                
            new_orientations = list(self.LENS_ORIENTATIONS)
            max_error_deg = 0.0
            total_corrections = 0
            
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
                
                if abs(shift_deg) > max_error_deg:
                    max_error_deg = abs(shift_deg)
                
                if abs(shift_deg) > 0.1:
                    status = "OK" if abs(shift_deg) <= ERROR_THRESHOLD_DEG else "FIXING"
                    print(f"  Lens {i+1} -> {next_i+1}: Error {shift_px:.1f}px ({shift_deg:.2f}°) [{status}]")
                    
                    old_yaw, pitch = new_orientations[next_i]
                    correction = shift_deg * LEARNING_RATE
                    correction = np.clip(correction, -MAX_CORRECTION_STEP, MAX_CORRECTION_STEP)
                    new_yaw = old_yaw + correction 
                    new_orientations[next_i] = (new_yaw, pitch)
                    total_corrections += 1

            if max_error_deg <= ERROR_THRESHOLD_DEG:
                print(f"  >>> SUCCESS! All errors are within range.")
                break
            
            if total_corrections == 0 and max_error_deg > ERROR_THRESHOLD_DEG:
                print("  >>> No clear SIFT matches found to improve further. Stopping.")
                break

            if iteration < MAX_ITERATIONS - 1:
                self.LENS_ORIENTATIONS = new_orientations
                self._init_geometry_and_weights()
            else:
                print("  >>> Reached Max Iterations.")

    def stitch_images(self, image_paths, output_path):
        print("\n[INFO] Loading images for stitching...")
        frames = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"[ERROR] Could not read image: {path}")
                return
            frames.append(img)
            
        if len(frames) != 6:
            print(f"[ERROR] Expected 6 images, but got {len(frames)}.")
            return

        # 1. Run the one-time SIFT calibration to line up the seams
        self.calibrate_angles_with_sift(frames)
        
        print("\n[INFO] Blending images into final panorama...")
        total_img = np.zeros((self.H, self.W, 3), dtype=np.float32)
        total_weight = np.zeros((self.H, self.W), dtype=np.float32)
        
        # 2. Map and blend using the optimized geometry
        for i in range(6):
            warped = cv2.remap(frames[i], self.maps_x[i], self.maps_y[i], cv2.INTER_LINEAR)
            w = self.weights[i]
            total_img += warped.astype(np.float32) * np.dstack([w, w, w])
            total_weight += w
        
        # 3. Normalize and save
        total_weight[total_weight == 0] = 1.0
        final_pano = np.clip(total_img / np.dstack([total_weight]*3), 0, 255).astype(np.uint8)
        
        cv2.imwrite(output_path, final_pano)
        print(f"[DONE] Saved high-quality stitched panorama to: {output_path}")

if __name__ == "__main__":
    # --- DIRECTORIES ---
    # Update this to where your pinhole JSONs live
    INTRINSICS_DIR = r"C:\IITM\DATA_RETRIEVAL_TASK\Extraction_2026-01-22_16-30-07\INTRINSICS" 
    
    # Update this to the folder containing the 6 output JPEGs from the extraction step
    IMAGE_DIR = r"C:\IITM\DATA_RETRIEVAL_TASK\Extraction_2026-01-22_16-30-07" 
    
    # Grab all 6 undistorted images in order
    image_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) 
                   if f.startswith("undistorted_LENS") and f.endswith(".jpg")]
                   
    # Ensure they are processed in order 1 to 6
    image_paths.sort() 

    if len(image_paths) == 6:
        output_pano_path = os.path.join(IMAGE_DIR, "Final_Static_Stitch.jpg")
        
        stitcher = SIFTInsta360ImageStitcher(intrinsics_dir=INTRINSICS_DIR)
        stitcher.stitch_images(image_paths, output_pano_path)
    else:
        print(f"[ERROR] Found {len(image_paths)} undistorted images. Need exactly 6.")