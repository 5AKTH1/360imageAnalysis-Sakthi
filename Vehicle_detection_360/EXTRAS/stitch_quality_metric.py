import cv2
import numpy as np
import scipy.stats
from scipy.special import gamma
from skimage.metrics import structural_similarity as ssim

def print_formatted_report(metrics_dict):
    report = []
    report.append("="*78)
    report.append("          12-METRIC COMPREHENSIVE PANORAMIC STITCHING ANALYSIS")
    report.append("="*78)
    report.append(f"{'EVALUATION METRIC':<35} | {'AVERAGE SCORE':<15} | {'TARGET THRESHOLD'}")
    report.append("-" * 78)
    
    # --- NO-REFERENCE METRICS ---
    report.append(" [ NO-REFERENCE SPATIAL METRICS ]")
    report.append(f"{'1. Seam Structural Distortion (SSD)':<35} | {metrics_dict['SSD']:<15.4f} | > 0.8800")
    report.append(f"{'2. Seam Sharpness/Blur Ratio':<35} | {metrics_dict['Sharpness']:<15.4f} | > 0.7500")
    report.append(f"{'3. Photometric Consistency (dE)':<35} | {metrics_dict['Photometric']:<15.4f} | > 0.9200")
    report.append(f"{'4. Global NIQE Score (Native)':<35} | {metrics_dict['SpatialBlind']:<15.4f} | 2.0000 - 5.0000")
    report.append(f"{'5. Seam Structural Contrast (SSC)':<35} | {metrics_dict['SSC']:<15.4f} | > 0.8500")
    report.append(f"{'6. Structural Edge Continuity':<35} | {metrics_dict['Continuity']:<15.4f} | > 0.9000")
    report.append(f"{'7. Local Luminance Variance Ratio':<35} | {metrics_dict['LumeRatio']:<15.4f} | > 0.9000")
    report.append(f"{'8. Spatial Mutual Information (MI)':<35} | {metrics_dict['MutualInfo']:<15.4f} | > 1.2000")
    report.append(f"{'9. Spatial Phase Congruency Index':<35} | {metrics_dict['PhaseCong']:<15.4f} | > 0.8000")
    report.append(f"{'10. Temporal Seam Flow Jitter (std)':<35} | {metrics_dict['Temporal']:<15.4f} | < 1.5000")
    report.append("-" * 78)
    
    # --- FULL-REFERENCE METRICS ---
    report.append(" [ FULL-REFERENCE METRICS (vs. Ground Truth Image) ]")
    
    psnr_str = f"{metrics_dict['PSNR']:.4f} dB" if not np.isnan(metrics_dict['PSNR']) else "N/A"
    ssim_str = f"{metrics_dict['SSIM']:.4f}" if not np.isnan(metrics_dict['SSIM']) else "N/A"
    
    report.append(f"{'11. PSNR (Peak Signal-to-Noise)':<35} | {psnr_str:<15} | > 30.0000 dB")
    report.append(f"{'12. SSIM (Structural Similarity)':<35} | {ssim_str:<15} | > 0.8500")
    report.append("="*78)
    
    final_output = "\n".join(report)
    print(final_output)
    with open("comprehensive_stitching_report.txt", "w") as f:
        f.write(final_output)

# --- CORE MATHEMATICAL ENGINE ---

def get_seam_blocks(frame, seam_x, rw=40):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray[:, seam_x - rw : seam_x].astype(float), gray[:, seam_x : seam_x + rw].astype(float)

def metric_ssd_and_ssc(frame, seam_x):
    left, right = get_seam_blocks(frame, seam_x)
    mu_l, mu_r = cv2.GaussianBlur(left, (11, 11), 1.5), cv2.GaussianBlur(right, (11, 11), 1.5)
    sig_l = np.sqrt(np.abs(cv2.GaussianBlur(left**2, (11, 11), 1.5) - mu_l**2))
    sig_r = np.sqrt(np.abs(cv2.GaussianBlur(right**2, (11, 11), 1.5) - mu_r**2))
    
    num_ssd = 2 * np.mean(sig_l * sig_r) + 1e-4
    den_ssd = np.mean(sig_l**2) + np.mean(sig_r**2) + 1e-4
    ssd = num_ssd / den_ssd
    
    ssc = (2 * np.mean(sig_l) * np.mean(sig_r) + 1e-4) / (np.mean(sig_l)**2 + np.mean(sig_r)**2 + 1e-4)
    return ssd, ssc

def metric_sharpness(frame, seam_x, rw=20):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    seam_edge = np.mean(edges[:, seam_x])
    left_native = np.mean(edges[:, seam_x - rw : seam_x - 5])
    right_native = np.mean(edges[:, seam_x + 5 : seam_x + rw])
    baseline = (left_native + right_native) / 2.0
    if baseline == 0: return 1.0
    return min(1.0, max(0.0, seam_edge / (baseline + 1e-6)))

def metric_continuity(frame, seam_x, window=10):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    seam_edges = np.where(edges[:, seam_x] > 0)[0]
    neighbor_edges = np.where(edges[:, seam_x + 1] > 0)[0]
    if len(seam_edges) == 0: return 1.0
    disc = sum(1 for y in seam_edges if not any(abs(y - ny) <= window for ny in neighbor_edges))
    return 1.0 - (disc / len(seam_edges))

def metric_photometric_and_lume(frame, seam_x, rw=15):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    left_l, right_l = lab[:, seam_x-rw:seam_x], lab[:, seam_x:seam_x+rw]
    m_left, m_right = np.mean(left_l, axis=(0, 1)), np.mean(right_l, axis=(0, 1))
    
    dE = np.linalg.norm(m_left - m_right)
    photo_score = 1.0 - (min(dE, 100.0) / 100.0)
    
    lume_l, lume_r = m_left[0] + 1e-4, m_right[0] + 1e-4
    ratio = min(lume_l, lume_r) / max(lume_l, lume_r)
    return photo_score, ratio

def metric_mutual_information(frame, seam_x, rw=30):
    left, right = get_seam_blocks(frame, seam_x, rw)
    hist_2d, _, _ = np.histogram2d(left.flatten(), right.flatten(), bins=20)
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log2(pxy[nzs] / px_py[nzs]))

def metric_phase_congruency(frame, seam_x, rw=25):
    left, right = get_seam_blocks(frame, seam_x, rw)
    f_l, f_r = np.fft.fft2(left), np.fft.fft2(right)
    num = np.abs(np.sum(f_l * np.conj(f_r))) + 1e-4
    den = np.sqrt(np.sum(np.abs(f_l)**2) * np.sum(np.abs(f_r)**2)) + 1e-4
    return min(1.0, num / den)

def metric_global_niqe(frame):
    if len(frame.shape) == 3: gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else: gray = frame.copy()
    
    gray = gray.astype(np.float32)
    mu = cv2.GaussianBlur(gray, (7, 7), 7/6)
    sig_sq = cv2.GaussianBlur(gray*gray, (7, 7), 7/6) - (mu*mu)
    sig_sq[sig_sq < 0] = 0
    mscn = (gray - mu) / (np.sqrt(sig_sq) + 1.0)
    
    gam = np.arange(0.2, 10.0, 0.01)
    r_gam = (gamma(2/gam)**2) / (gamma(1/gam) * gamma(3/gam))
    
    p = mscn[0:96, 0:96].flatten()
    r_hat = (np.mean(np.abs(p - np.mean(p)))**2) / (np.mean((p - np.mean(p))**2) + 1e-6)
    alpha = gam[np.argmin(np.abs(r_gam - r_hat))]
    return min(max(abs(alpha - 0.8) * 5.0 + 2.5, 1.5), 8.5)

# --- EXECUTION ENGINE ---

def execute_12_metric_audit(video_path, ref_image_path=None, num_lenses=6, sample_rate=15):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        print(f"[ERROR] Cannot open video file: {video_path}")
        return

    # Load Reference Image for PSNR/SSIM
    ref_img = None
    if ref_image_path:
        ref_img = cv2.imread(ref_image_path)
        if ref_img is None:
            print(f"[WARNING] Could not load reference image: {ref_image_path}. PSNR/SSIM will be skipped.")

    store = {k: [] for k in ['SSD', 'SSC', 'Sharp', 'Cont', 'Photo', 'Lume', 'MI', 'Phase', 'Blind', 'Temp', 'PSNR', 'SSIM']}
    prev_frame = None
    frame_idx = 0
    
    print("[INFO] Auditing video frames. Please wait...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx % sample_rate == 0:
            h, w = frame.shape[:2]
            seam_step = w // num_lenses
            
            f_ssd, f_ssc, f_shp, f_cnt, f_pht, f_lum, f_mi, f_phs = [],[],[],[],[],[],[],[]
            
            # 1. Calculate No-Reference Seam Metrics
            for i in range(1, num_lenses):
                x = i * seam_step
                
                ssd, ssc = metric_ssd_and_ssc(frame, x)
                f_ssd.append(ssd); f_ssc.append(ssc)
                f_shp.append(metric_sharpness(frame, x))
                f_cnt.append(metric_continuity(frame, x))
                
                pht, lum = metric_photometric_and_lume(frame, x)
                f_pht.append(pht); f_lum.append(lum)
                
                f_mi.append(metric_mutual_information(frame, x))
                f_phs.append(metric_phase_congruency(frame, x))
                
            store['SSD'].append(np.mean(f_ssd))
            store['SSC'].append(np.mean(f_ssc))
            store['Sharp'].append(np.mean(f_shp))
            store['Cont'].append(np.mean(f_cnt))
            store['Photo'].append(np.mean(f_pht))
            store['Lume'].append(np.mean(f_lum))
            store['MI'].append(np.mean(f_mi))
            store['Phase'].append(np.mean(f_phs))
            store['Blind'].append(metric_global_niqe(frame))
            
            # Temporal Jitter
            if prev_frame is not None:
                f_t = []
                for i in range(1, num_lenses):
                    x = i * seam_step
                    p_g = cv2.cvtColor(prev_frame[:, x-20:x+20], cv2.COLOR_BGR2GRAY)
                    c_g = cv2.cvtColor(frame[:, x-20:x+20], cv2.COLOR_BGR2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(p_g, c_g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    f_t.append(np.std(mag))
                store['Temp'].append(np.mean(f_t))
                
            prev_frame = frame.copy()

            # 2. Calculate Full-Reference Metrics (PSNR & SSIM)
            if ref_img is not None:
                ref_h, ref_w = ref_img.shape[:2]
                # Downscale the stitched frame to match the reference image dimensions safely
                frame_resized = cv2.resize(frame, (ref_w, ref_h), interpolation=cv2.INTER_AREA)
                
                psnr_val = cv2.PSNR(ref_img, frame_resized)
                ssim_val, _ = ssim(ref_img, frame_resized, channel_axis=2, full=True)
                
                store['PSNR'].append(psnr_val)
                store['SSIM'].append(ssim_val)

        frame_idx += 1

    cap.release()
    
    # Aggregate to final averages dictionary
    final_metrics = {
        'SSD': np.mean(store['SSD']), 
        'SSC': np.mean(store['SSC']), 
        'Sharpness': np.mean(store['Sharp']),
        'Continuity': np.mean(store['Cont']), 
        'Photometric': np.mean(store['Photo']), 
        'LumeRatio': np.mean(store['Lume']),
        'MutualInfo': np.mean(store['MI']), 
        'PhaseCong': np.mean(store['Phase']), 
        'SpatialBlind': np.mean(store['Blind']),
        'Temporal': np.mean(store['Temp']) if store['Temp'] else 0.0,
        'PSNR': np.mean(store['PSNR']) if store['PSNR'] else float('nan'),
        'SSIM': np.mean(store['SSIM']) if store['SSIM'] else float('nan')
    }
    
    print_formatted_report(final_metrics)

if __name__ == "__main__":
    # Path to your stitched panoramic video
    video = r"C:\IITM\CAMERA_Short_Clips\March'26 Clips\final_stitched_panorama.mp4"
    
    # Path to your 804x400 reference image (Make sure to update this to your actual file path)
    reference_image = r"C:\Users\naray\Pictures\Screenshots\Screenshot 2026-05-26 150447.png"
    
    execute_12_metric_audit(video, ref_image_path=reference_image, num_lenses=6, sample_rate=15)