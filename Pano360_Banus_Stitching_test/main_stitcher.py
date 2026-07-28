import argparse
import os
import numpy as np
import cv2
from features import matching
from bundle_adj import traverse

def _add_weights(img):
    """Adds an alpha channel for feathering/blending."""
    img_f = img.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    y, x = np.indices((h, w))
    # Create a 'hat' function weight map to smooth edges
    wy = 0.5 - np.abs(y / h - 0.5)
    wx = 0.5 - np.abs(x / w - 0.5)
    weight = (wy * wx).astype(np.float32)
    return cv2.merge([img_f[:,:,0], img_f[:,:,1], img_f[:,:,2], weight])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Path to lens images folder")
    parser.add_argument('-o', '--out', type=str, default='panorama_result.jpg')
    args = parser.parse_args()

    # 1. Load Images
    files = sorted([f for f in os.listdir(args.path) if f.lower().endswith(('.jpg', '.png'))])
    imgs = [cv2.imread(os.path.join(args.path, f)) for f in files]
    if not imgs:
        print("No images found!")
        return

    # 2. Extract and Match Features
    print(f"Processing {len(imgs)} lenses...")
    kpts, matches = matching(imgs)
    if isinstance(matches, np.ndarray): 
        matches = matches.item()

    # 3. Solve for Camera Rotations (Bundle Adjustment)
    print("Calculating camera alignment...")
    regions = traverse(imgs, matches)

    # 4. Initialize Large Canvas
    # We create a canvas large enough to hold all warped images
    h, w = imgs[0].shape[:2]
    canvas_h, canvas_w = h * 2, w * 4 
    mosaic = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight_sum = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)

    # Offset to keep the panorama centered on the canvas
    offset_h, offset_w = canvas_h // 4, canvas_w // 4

    print("Warping and Blending...")
    for reg in regions:
        # Prepare image with weights for smooth blending
        weighted_img = _add_weights(reg.img)
        
        # Calculate Homography relative to the first lens (anchor)
        # Using the intrinsic and rotation matrices from bundle_adj.py
        H = reg.intr.dot(reg.rot).dot(np.linalg.inv(regions[0].intr))
        
        # Adjust Homography for the canvas offset
        T = np.array([[1, 0, offset_w], [0, 1, offset_h], [0, 0, 1]])
        H_final = T.dot(H)

        # Warp the lens image onto the global mosaic
        warped = cv2.warpPerspective(weighted_img, H_final, (canvas_w, canvas_h))
        
        # Accumulate pixels and weights for linear blending
        mosaic += warped[..., :3] * warped[..., [3]]
        weight_sum += warped[..., [3]]

    # 5. Finalize and Save
    # Normalize by weights to handle overlapping areas
    mosaic /= (weight_sum + 1e-8)
    final_res = (np.clip(mosaic, 0, 1) * 255).astype(np.uint8)

    # Optional: Auto-crop black borders
    gray = cv2.cvtColor(final_res, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    x, y, w_c, h_c = cv2.boundingRect(thresh)
    final_res = final_res[y:y+h_c, x:x+w_c]

    cv2.imwrite(args.out, final_res)
    print(f"Success! Panorama saved to {args.out}")

if __name__ == '__main__':
    main()