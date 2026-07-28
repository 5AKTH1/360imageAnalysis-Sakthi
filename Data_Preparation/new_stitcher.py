import os
import json
import numpy as np
import cv2
import glob

class ClipToImageStitcher:
    def __init__(self, intrinsics_dir):
        self.intrinsics_dir = intrinsics_dir
        
        # --- CONFIGURATION ---
        self.W = 3328
        self.H = 1664
        
        # --- MANUAL CROP SETTINGS ---
        self.CROP_TOP = 331
        self.CROP_BOTTOM = 335
        
        self.LENS_ZOOMS = [
            1.2500, 1.2300, 1.1600, 1.2500, 1.2500, 1.2500,
        ]

        self.LENS_ORIENTATIONS = [
            (-78.9035, 0.0000), 
            (-142.0095, 0.0000), 
            (161.6066, 0.0000), 
            (113.4935, 0.0000), 
            (43.6631, 0.0000), 
            (-20.6499, 0.0000), 
        ]
        
        self.maps_x = []
        self.maps_y = []
        self.weights = []
        self.base_Ks = [] 
        self.Ks = []      
        self.img_dims = []

        self._load_intrinsics()
        print(f"Initializing Geometry ({self.W}x{self.H})...")
        self._init_geometry_and_weights()

    def _load_intrinsics(self):
        self.base_Ks = []
        for i in range(1, 7):
            path = os.path.join(self.intrinsics_dir, f"calibration_pinhole_lens{i}.json")

            if not os.path.exists(path):
                print(f"Warning: {path} not found. Using generic.")
                K = np.array([[1000,0,960],[0,1000,540],[0,0,1]], dtype=np.float32)
                self.img_dims.append((1920,1080))
                self.base_Ks.append(K)
            else:
                with open(path, 'r') as f:
                    data = json.load(f)
                K = np.array(data['K'], dtype=np.float32)
                self.base_Ks.append(K)
                self.img_dims.append(tuple(data['image_size']))

    def _init_geometry_and_weights(self):
        self.maps_x = []
        self.maps_y = []
        self.weights = []
        self.Ks = []
        
        for i in range(6):
            K = self.base_Ks[i].copy()
            K[0,0] *= self.LENS_ZOOMS[i]
            K[1,1] *= self.LENS_ZOOMS[i]
            self.Ks.append(K)

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
            return None
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
                
        if len(good) < 4:
            return None
            
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        
        dx = pts1[:, 0] - pts2[:, 0]
        dy = pts1[:, 1] - pts2[:, 1]
        
        valid_matches = np.abs(dy) < 15 
        if np.sum(valid_matches) < 3: return None
        
        return np.median(dx[valid_matches])

    def auto_tune_zooms(self, frames):
        print("\n--- STARTING AUTO-ZOOM TUNING ---")
        TEST_ZOOMS = []
        
        for i in range(6):
            best_zoom = self.LENS_ZOOMS[i]
            best_error = float('inf')

            for test_z in TEST_ZOOMS:
                self.LENS_ZOOMS[i] = test_z
                self._init_geometry_and_weights() 

                warped_i = cv2.remap(frames[i], self.maps_x[i], self.maps_y[i], cv2.INTER_LINEAR)
                prev_i = (i - 1) % 6
                next_i = (i + 1) % 6

                warped_prev = cv2.remap(frames[prev_i], self.maps_x[prev_i], self.maps_y[prev_i], cv2.INTER_LINEAR)
                warped_next = cv2.remap(frames[next_i], self.maps_x[next_i], self.maps_y[next_i], cv2.INTER_LINEAR)

                def get_seam_error(w1, w2):
                    mask1 = cv2.cvtColor(w1, cv2.COLOR_BGR2GRAY) > 10
                    mask2 = cv2.cvtColor(w2, cv2.COLOR_BGR2GRAY) > 10
                    overlap = mask1 & mask2
                    if np.sum(overlap) == 0: return float('inf')
                    
                    y, x = np.where(overlap)
                    min_x, max_x = np.min(x), np.max(x)
                    if (max_x - min_x) < 10: return float('inf')
                    
                    s1 = w1[:, min_x:max_x]
                    s2 = w2[:, min_x:max_x]
                    
                    shift = self.find_shift_with_sift(s1, s2)
                    if shift is None: return float('inf') 
                    return abs(shift)

                err_left = get_seam_error(warped_prev, warped_i)
                err_right = get_seam_error(warped_i, warped_next)
                total_err = err_left + err_right
                
                if total_err != float('inf'):
                    print(f"  Lens {i+1} @ {test_z:.2f}x -> Shift Error: {total_err:.1f}px")
                else:
                    print(f"  Lens {i+1} @ {test_z:.2f}x -> SIFT Failed")

                if total_err < best_error:
                    best_error = total_err
                    best_zoom = test_z

            self.LENS_ZOOMS[i] = best_zoom
            print(f"  >>> Locked Lens {i+1} Zoom at {best_zoom:.2f}x\n")

        self._init_geometry_and_weights()
        print("Final Auto-Tuned Zooms:", [f"{z:.2f}" for z in self.LENS_ZOOMS])

    def calibrate_angles_with_sift(self, frames):
        MAX_ITERATIONS = 50       
        ERROR_THRESHOLD_DEG = 0.4 
        LEARNING_RATE = 0.25      
        MAX_CORRECTION_STEP = 1.5 
        
        print(f"\n--- STARTING ANGLE CALIBRATION ---")
        pixels_per_deg = self.W / 360.0

        for iteration in range(MAX_ITERATIONS):
            print(f"\n[Iteration {iteration + 1}/{MAX_ITERATIONS}] Checking alignment...")
            
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
                
                if shift_px is None: continue
                    
                shift_deg = shift_px / pixels_per_deg
                
                if abs(shift_deg) > max_error_deg:
                    max_error_deg = abs(shift_deg)
                
                status = "OK" if abs(shift_deg) <= ERROR_THRESHOLD_DEG else "FIXING"
                if abs(shift_deg) > 0.1:
                    print(f"  Lens {i+1} -> {next_i+1}: Error {shift_px:.1f}px ({shift_deg:.2f}°) [{status}]")
                
                if abs(shift_deg) > 0.1: 
                    old_yaw, pitch = new_orientations[next_i]
                    correction = shift_deg * LEARNING_RATE
                    correction = np.clip(correction, -MAX_CORRECTION_STEP, MAX_CORRECTION_STEP)
                    new_yaw = old_yaw + correction 
                    new_orientations[next_i] = (new_yaw, pitch)
                    total_corrections += 1

            print(f"  >>> Max Error: {max_error_deg:.3f}°")
            
            if max_error_deg <= ERROR_THRESHOLD_DEG:
                print(f"  >>> SUCCESS! All errors are within range [-{ERROR_THRESHOLD_DEG}, {ERROR_THRESHOLD_DEG}].")
                break
            
            if total_corrections == 0 and max_error_deg > ERROR_THRESHOLD_DEG:
                print("  >>> No clear SIFT matches found to improve further. Stopping.")
                break

            if iteration < MAX_ITERATIONS - 1:
                self.LENS_ORIENTATIONS = new_orientations
                self._init_geometry_and_weights()
            else:
                print("  >>> Reached Max Iterations.")

        print("\n" + "="*40)
        print("   FINAL OPTIMIZED CONFIGURATION")
        print("="*40)
        print("self.LENS_ZOOMS = [")
        for z in self.LENS_ZOOMS: print(f"    {z:.4f},")
        print("]")
        print("\nself.LENS_ORIENTATIONS = [")
        for i, (yaw, pitch) in enumerate(self.LENS_ORIENTATIONS):
            print(f"    ({yaw:.4f}, {pitch:.4f}), # Lens {i+1}")
        print("]")
        print("="*40 + "\n")

    def stitch_frame_from_video(self, input_paths, output_path, frame_idx=0):
        """
        Reads a specific frame from 6 video clips, stitches them into a panorama,
        crops the black edges, and saves as a single .jpg image.
        """
        print(f"\n[INFO] Extracting Frame {frame_idx} from 6 video clips...")
        
        frames = []
        for p in input_paths:
            cap = cv2.VideoCapture(p)
            if not cap.isOpened():
                print(f"[ERROR] Could not open video stream: {p}")
                return False
                
            # Seek to the requested frame (default is 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                print(f"[ERROR] Could not read frame {frame_idx} from {p}")
                return False
                
            frames.append(frame)
            
        # 2. Tune and Calibrate using the extracted frames
        self.auto_tune_zooms(frames)
        self.calibrate_angles_with_sift(frames)
            
        print("\n[INFO] Blending frames into final panoramic image...")
        total_img = np.zeros((self.H, self.W, 3), dtype=np.float32)
        total_weight = np.zeros((self.H, self.W), dtype=np.float32)
        
        # 3. Warp and Blend
        for i in range(6):
            warped = cv2.remap(frames[i], self.maps_x[i], self.maps_y[i], cv2.INTER_LINEAR)
            w = self.weights[i]
            total_img += warped.astype(np.float32) * np.dstack([w, w, w])
            total_weight += w
        
        total_weight[total_weight == 0] = 1.0
        final_pano = np.clip(total_img / np.dstack([total_weight]*3), 0, 255).astype(np.uint8)
        
        # 4. Apply the Manual Crop
        cropped_pano = final_pano[self.CROP_TOP : self.H - self.CROP_BOTTOM, :]
        
        # 5. Save the output
        cv2.imwrite(output_path, cropped_pano)
        print(f"[DONE] Panorama successfully saved to {output_path}")
        return True


# ==========================================
# STANDALONE EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    
    BASE_DIR = r"C:\IITM\CAMERA_Short_Clips\March'26 Clips"
    INTRINSICS_DIR = r"C:\IITM\IMU-GPS"
    
    # Notice we are outputting a .jpg, not an .mp4
    OUTPUT_PANO_IMAGE = os.path.join(BASE_DIR, "final_stitched_frame.jpg")

    print(f"Scanning directory: {BASE_DIR}")
    print("Searching for undistorted video streams...")
    
    input_video_paths = []

    for i in range(6):
        # We search for .mp4 video clips as requested
        search_pattern = os.path.join(BASE_DIR, f"undistorted_origin_{i}_*.mp4")
        matches = glob.glob(search_pattern)

        if not matches:
            print(f"\n[ERROR] Could not find the undistorted video for origin_{i}")
            exit(1)
        
        input_video_paths.append(matches[0])

    print("\nFound all 6 required video streams in order:")
    for idx, path in enumerate(input_video_paths):
        print(f"  Lens {idx+1}: {os.path.basename(path)}")

    print(f"\nInitializing the Video-to-Image Stitcher...")
    stitcher = ClipToImageStitcher(intrinsics_dir=INTRINSICS_DIR)

    print(f"\nStarting the Stitching Process. Output will be saved to:")
    print(f"{OUTPUT_PANO_IMAGE}\n")
    
    # You can change frame_idx if you want a frame further into the video (e.g., frame_idx=30)
    success = stitcher.stitch_frame_from_video(input_paths=input_video_paths, output_path=OUTPUT_PANO_IMAGE, frame_idx=0)

    if success:
        print(f"\n[SUCCESS] Image extraction and stitching completed successfully!")
    else:
        print("\n[ERROR] Stitching failed. Please check your input files and paths.")
