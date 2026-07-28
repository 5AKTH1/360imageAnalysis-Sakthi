import os
import sys

# Completely silence OpenCV and FFmpeg backend warnings
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

import cv2
import json
import numpy as np
from typing import Tuple, Dict, Any, List, Optional

# Ensure the user has the 'omnidir' module installed
if not hasattr(cv2, 'omnidir'):
    print("\n[CRITICAL ERROR] The 'cv2.omnidir' module is missing!")
    print("Your calibration files use the Omnidirectional model, which requires OpenCV's contrib modules.")
    print("Please run this command in your terminal, and then restart the script:")
    print("    pip install opencv-contrib-python\n")
    sys.exit(1)

def load_lens_intrinsics_omnidir(json_path: str) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int]]:
    with open(json_path, "r", encoding="utf-8") as f:
        jd = json.load(f)
    
    if jd.get("model") != "omnidirectional":
        print(f"Warning: JSON model is '{jd.get('model')}', but we are parsing it as omnidirectional.")
        
    K = np.array(jd.get("K") or [], dtype=np.float64).reshape(3, 3)
    
    D_list = jd.get("D") or []
    D = np.array(D_list, dtype=np.float64).reshape(1, 4) if len(D_list) == 4 else np.zeros((1, 4), np.float64)
    
    xi = float(jd.get("xi", 1.0))
    
    img_sz = jd.get("image_size", {})
    w = int(img_sz.get("w", 2880))
    h = int(img_sz.get("h", 3840))
        
    return K, D, xi, (w, h)

def locate_video_file(camera_dir: str, lens_id: int) -> str:
    if not os.path.exists(camera_dir):
        raise FileNotFoundError(f"Directory not found: {camera_dir}")
    for root, _, files in os.walk(camera_dir):
        for f in files:
            if f.lower().endswith(".mp4"):
                full_path = os.path.join(root, f)
                if f"lens{lens_id}" in full_path.lower().replace(" ", "").replace("_", ""):
                    return full_path
    raise FileNotFoundError(f"Could not locate an MP4 container for Lens {lens_id}")

def locate_intrinsic_file(intrinsics_dir: str, lens_id: int) -> str:
    if not os.path.exists(intrinsics_dir):
        raise FileNotFoundError(f"Directory not found: {intrinsics_dir}")
    for root, _, files in os.walk(intrinsics_dir):
        for f in files:
            if f.lower().endswith(".json"):
                full_path = os.path.join(root, f)
                filename_clean = f.lower().replace(" ", "").replace("_", "")
                parent_clean = os.path.basename(root).lower().replace(" ", "").replace("_", "")
                if f"lens{lens_id}" in filename_clean or f"lens{lens_id}" in parent_clean:
                    return full_path
    raise FileNotFoundError(f"Could not locate JSON parameter file for Lens {lens_id}")

def standalone_detect_checkerboard(frame_bgr: np.ndarray, pattern_size: Tuple[int, int], downscale: int = 2) -> Tuple[bool, Optional[np.ndarray]]:
    if frame_bgr is None or frame_bgr.size == 0:
        return False, None
    
    gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    if downscale > 1:
        h, w = gray_full.shape[:2]
        gray_small = cv2.resize(gray_full, (w // downscale, h // downscale), interpolation=cv2.INTER_AREA)
    else:
        gray_small = gray_full

    cols, rows = pattern_size
    expected = cols * rows
    
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    
    for pat in [(cols, rows), (rows, cols)]:
        found, corners = cv2.findChessboardCorners(gray_small, pat, flags=flags)
        
        if found and corners is not None and len(corners) == expected:
            corners = corners.astype(np.float32).reshape(-1, 1, 2)
            
            if downscale > 1:
                corners *= downscale
            
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
            cv2.cornerSubPix(gray_full, corners, (5, 5), (-1, -1), crit)
            
            if np.isnan(corners).any() or np.isinf(corners).any():
                return False, None
                
            return True, corners
            
    return False, None

def extract_pairwise_points_omnidir(
    video_path_1: str,
    video_path_2: str,
    pattern_size: Tuple[int, int],
    square_size: float,
    resolution_expected: Tuple[int, int],
    min_pixel_motion: float = 10.0,  # Lowered to capture frames closer together
    frame_skip: int = 7,    
    downscale: int = 2,
    max_pairs: int = 50              # Increased to cast a wider net for the "Golden Frame"
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    
    sys.stderr = open(os.devnull, 'w')
    capture_1 = cv2.VideoCapture(video_path_1)
    capture_2 = cv2.VideoCapture(video_path_2)
    sys.stderr = sys.__stderr__
    
    object_points_collection = []
    image_points_1 = []
    image_points_2 = []
    
    cols, rows = pattern_size
    expected_corners = cols * rows
    last_tracked_corners = None
    frame_count = 0
    
    exp_w, exp_h = resolution_expected
    
    while capture_1.isOpened() and capture_2.isOpened():
        success_1, image_1 = capture_1.read()
        success_2, image_2 = capture_2.read()
        
        if not success_1 or not success_2:
            break
            
        frame_count += 1
        
        if frame_count % frame_skip != 0:
            continue
            
        # VERY IMPORTANT: Fix Landscape/Portrait orientation mismatch
        if image_1.shape[1] > image_1.shape[0] and exp_w < exp_h:
            image_1 = cv2.rotate(image_1, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image_2 = cv2.rotate(image_2, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        found_1, corners_1 = standalone_detect_checkerboard(image_1, pattern_size, downscale=downscale)
        if not found_1 or corners_1 is None:
            continue
            
        found_2, corners_2 = standalone_detect_checkerboard(image_2, pattern_size, downscale=downscale)
        if not found_2 or corners_2 is None:
            continue
            
        if last_tracked_corners is not None:
            spatial_delta = np.linalg.norm(corners_1.reshape(-1, 2) - last_tracked_corners.reshape(-1, 2), axis=1).mean()
            if spatial_delta < min_pixel_motion:
                continue
                
        grid_3d = np.zeros((expected_corners, 1, 3), np.float32)
        grid_3d[:, 0, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        grid_3d *= square_size
        
        object_points_collection.append(grid_3d)
        image_points_1.append(corners_1)
        image_points_2.append(corners_2)
        last_tracked_corners = corners_1
        
        print(f"      -> Collected valid frame pair {len(object_points_collection)}/{max_pairs} (Frame {frame_count})")
        
        if len(object_points_collection) >= max_pairs:
            print("      -> Reached optimum frame target. Stopping video scan early.")
            break
            
    capture_1.release()
    capture_2.release()
    return object_points_collection, image_points_1, image_points_2

def compute_pairwise_extrinsics_omnidir(
    object_pts: List[np.ndarray],
    img_pts_1: List[np.ndarray],
    img_pts_2: List[np.ndarray],
    k1: np.ndarray, d1: np.ndarray, xi1: float,
    k2: np.ndarray, d2: np.ndarray, xi2: float,
    known_baseline_cm: float
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    
    K_id = np.eye(3, dtype=np.float64)
    D_id = np.zeros((4, 1), dtype=np.float64)
    
    # Store tuples of (RMS, R, T, frame_index) to find the absolute best synced frame
    frame_results = []
    first_error = None
    
    for i in range(len(object_pts)):
        try:
            # 1. Safely undistort points into normalized space
            try:
                norm_pts_1 = cv2.omnidir.undistortPoints(img_pts_1[i], k1, d1, xi1, K_id)
                norm_pts_2 = cv2.omnidir.undistortPoints(img_pts_2[i], k2, d2, xi2, K_id)
            except Exception:
                xi1_arr = np.array([xi1], dtype=np.float64)
                xi2_arr = np.array([xi2], dtype=np.float64)
                norm_pts_1 = cv2.omnidir.undistortPoints(img_pts_1[i], k1, d1, xi1_arr, K_id)
                norm_pts_2 = cv2.omnidir.undistortPoints(img_pts_2[i], k2, d2, xi2_arr, K_id)
            
            norm_pts_1 = norm_pts_1.reshape(-1, 1, 2).astype(np.float32)
            norm_pts_2 = norm_pts_2.reshape(-1, 1, 2).astype(np.float32)
            
            # 2. Use IPPE solver (mathematically optimal for planar checkerboards)
            flags = getattr(cv2, "SOLVEPNP_IPPE", cv2.SOLVEPNP_ITERATIVE)
            success1, rvec1, tvec1 = cv2.solvePnP(object_pts[i], norm_pts_1, K_id, D_id, flags=flags)
            success2, rvec2, tvec2 = cv2.solvePnP(object_pts[i], norm_pts_2, K_id, D_id, flags=flags)
            
            if success1 and success2:
                # Sanity check: Checkerboard must be in front of the camera (Z > 0)
                if tvec1[2].item() <= 0 or tvec2[2].item() <= 0:
                    if first_error is None:
                        first_error = "Z-axis <= 0. Checkerboard was solved as being behind the camera."
                    continue
                    
                R1, _ = cv2.Rodrigues(rvec1)
                R2, _ = cv2.Rodrigues(rvec2)
                
                # Relative Extrinsics: Transform from Camera 1 to Camera 2
                R_rel = R2 @ R1.T
                T_rel = tvec2 - R_rel @ tvec1
                
                # 3. Calculate Independent Frame RMS
                # Project the 3D points directly into Lens 2 using THIS frame's transform
                pts_3d_cam1 = (R1 @ object_pts[i].reshape(-1, 3).T).T + tvec1.reshape(1, 3)
                pts_3d_cam2 = (R_rel @ pts_3d_cam1.T).T + T_rel.reshape(1, 3)
                pts_3d_cam2_expanded = pts_3d_cam2.reshape(-1, 1, 3)
                
                rvec_id = np.zeros((3, 1), dtype=np.float64)
                tvec_id = np.zeros((3, 1), dtype=np.float64)
                
                try:
                    proj_pts_2, _ = cv2.omnidir.projectPoints(pts_3d_cam2_expanded, rvec_id, tvec_id, k2, xi2, d2)
                except Exception:
                    xi2_arr = np.array([xi2], dtype=np.float64)
                    proj_pts_2, _ = cv2.omnidir.projectPoints(pts_3d_cam2_expanded, rvec_id, tvec_id, k2, xi2_arr, d2)
                    
                # Measure pixel error between physical corner and mathematical projection
                err = np.linalg.norm(proj_pts_2.reshape(-1, 2) - img_pts_2[i].reshape(-1, 2), axis=1)
                frame_rms = np.sqrt(np.mean(err ** 2))
                
                # Log success
                frame_results.append((frame_rms, R_rel, T_rel, i))
                
            else:
                if first_error is None:
                    first_error = "cv2.solvePnP returned False. Target plane may be degenerate."
        except Exception as e:
            if first_error is None:
                first_error = f"Exception: {str(e)}"
            continue
            
    if not frame_results:
        raise RuntimeError(f"Analytic geometry solver failed to find any valid poses. Root cause: {first_error}")
        
    # 4. Golden Frame Extraction
    # Sort all processed frames by their RMS error (lowest first)
    frame_results.sort(key=lambda x: x[0])
    
    best_rms, best_R, best_T, best_idx = frame_results[0]
    
    print(f"      -> 'Golden Frame' Selected: Frame idx {best_idx} out of {len(frame_results)} successful pairs.")
    print(f"      -> Best isolated RMS errors: {frame_results[0][0]:.3f}, {frame_results[1][0] if len(frame_results)>1 else 0:.3f}, {frame_results[2][0] if len(frame_results)>2 else 0:.3f}")
    
    # 5. Fix Scale Ambiguity
    # Extract the vector direction and enforce the known physical baseline in centimeters
    t_norm = np.linalg.norm(best_T)
    if t_norm > 0:
        best_T_cm = (best_T / t_norm) * known_baseline_cm
        print(f"      -> Scaling Translation Vector to strict physical baseline: {known_baseline_cm} cm")
    else:
        best_T_cm = best_T
            
    return best_R, best_T_cm, float(best_rms), 1 

def compose_global_transforms(pairwise_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    global_transforms = {
        1: {"R": np.eye(3).tolist(), "T": np.zeros((3, 1)).flatten().tolist()}
    }
    current_r = np.eye(3, dtype=np.float64)
    current_t = np.zeros((3, 1), dtype=np.float64)
    
    for origin_id in range(1, 6):
        target_id = origin_id + 1
        key = f"{origin_id}->{target_id}"
        if key not in pairwise_data:
            print(f"[-] Missing transformation link for {key}. Stopping global loop.")
            break
        step_r = np.array(pairwise_data[key]["R"], dtype=np.float64)
        step_t = np.array(pairwise_data[key]["T"], dtype=np.float64).reshape(3, 1)
        
        current_t = step_r @ current_t + step_t
        current_r = step_r @ current_r
        
        global_transforms[target_id] = {
            "R": current_r.tolist(),
            "T": current_t.flatten().tolist()
        }
    return global_transforms

def run_ring_calibration_pipeline(
    camera_dir: str,
    intrinsics_dir: str,
    output_dir: str,
    pattern_size: Tuple[int, int],
    square_size: float,
    known_baseline_cm: float
):
    os.makedirs(output_dir, exist_ok=True)
    pairwise_results = {}
    
    print("=" * 60)
    print("Omnidirectional Analytic Stereo Calibration Pipeline")
    print("=" * 60)
    
    for i in range(1, 7):
        j = i + 1 if i < 6 else 1
        pair_key = f"{i}->{j}"
        print(f"\n[+] Analyzing Lens Pair: {pair_key}")
        try:
            vid1 = locate_video_file(camera_dir, i)
            vid2 = locate_video_file(camera_dir, j)
            json_path_1 = locate_intrinsic_file(intrinsics_dir, i)
            json_path_2 = locate_intrinsic_file(intrinsics_dir, j)
        except FileNotFoundError as e:
            print(f"    [!] Missing file dependency: {e}")
            continue
            
        k1, d1, xi1, res1 = load_lens_intrinsics_omnidir(json_path_1)
        k2, d2, xi2, res2 = load_lens_intrinsics_omnidir(json_path_2)
        
        print("    - Scanning video timelines & aligning Portrait/Landscape dimensions...")
        obj_pts, img_pts_1, img_pts_2 = extract_pairwise_points_omnidir(
            vid1, vid2, pattern_size, square_size, resolution_expected=res1, frame_skip=15, downscale=2
        )
        
        if len(obj_pts) < 5:
            print(f"    [!] Warning: Insufficient overlapping features for pair. Found {len(obj_pts)}, need at least 5.")
            continue
            
        try:
            rot, trans, rms, valid_count = compute_pairwise_extrinsics_omnidir(
                obj_pts, img_pts_1, img_pts_2, k1, d1, xi1, k2, d2, xi2, known_baseline_cm
            )
            print(f"    [*] Omnidirectional Extrinsics Converged! RMS Error: {rms:.4f}")
            
            pairwise_results[pair_key] = {
                "R": rot.tolist(),
                "T": trans.flatten().tolist(),
                "rms": rms,
                "samples": valid_count
            }
        except RuntimeError as e:
            print(f"    [!] {e}")
            continue
        
    if not pairwise_results:
        print("\n[!] Critical: No camera lens pairs were successfully optimized.")
        return
        
    with open(os.path.join(output_dir, "extrinsics_pairwise.json"), "w", encoding="utf-8") as f:
        json.dump(pairwise_results, f, indent=2)
        
    print("\n[+] Linking relative matrices into global coordinate transform matrix space (centimeters)...")
    global_system_map = compose_global_transforms(pairwise_results)
    
    with open(os.path.join(output_dir, "extrinsics_global.json"), "w", encoding="utf-8") as f:
        json.dump(global_system_map, f, indent=2)
        
    print(f"\n[=] Success! Calibration completed. Results saved to: {output_dir}")

if __name__ == "__main__":
    run_ring_calibration_pipeline(
        camera_dir=r"G:\MUMMAS DATA COLLECTION\MUMMAS AQI BUILDING\15062026\CAMERA\run_20260615_140421_9024",
        intrinsics_dir=r"C:\viswak_MUMMAS_360degcamera\Insta360ImageAnalysis\INTRINSICS",
        output_dir=r"C:\viswak_MUMMAS_360degcamera\Insta360ImageAnalysis\CAMERA_EXTRINSICS\extrinsics_output",
        pattern_size=(16, 6),
        square_size=50.0,
        known_baseline_cm=7.15  # Insta360 Pro 2 physical baseline between adjacent lenses
    )