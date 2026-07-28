#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
undistort_omni_images.py

Batch-rectify fisheye (omnidirectional) images to pinhole images
using cv2.omnidir.undistortImage.

Expected directory structure:

  side_by_side/
    data/
      images/
        lens1/
          cam_*.png      # input fisheye images
        lens2/
          ...
      intrinsics/
        calibration_omni.json   # K, D, xi

Output, for each lens (e.g. lens1):

  data/images/lens1/undistorted/cam_*.png
      Rectified pinhole images.

  data/intrinsics/calibration_pinhole_lens1.json
      New pinhole intrinsics (K_new, D_new = 0).

Use K_new and D_new=0 for any further PnP / projection on these rectified images.
"""

import os
import sys
import json
import argparse
from typing import Tuple

import numpy as np
import cv2

HERE = os.path.abspath(os.path.dirname(__file__))
DATA_ROOT = os.path.join(HERE, "data")
IMAGES_ROOT = os.path.join(DATA_ROOT, "images")
INTRINSICS_ROOT = os.path.join(DATA_ROOT, "intrinsics")


def load_omni_intrinsics(json_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load omnidirectional intrinsics (K, D, xi) from JSON.

    JSON must contain:
      - either "K" or "camera_matrix"
      - either "D" or "dist_coeffs"
      - "xi" (scalar)
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Intrinsics JSON not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    if "K" in data:
        K = np.asarray(data["K"], dtype=np.float64)
    elif "camera_matrix" in data:
        K = np.asarray(data["camera_matrix"], dtype=np.float64)
    else:
        raise KeyError("Intrinsics JSON must contain 'K' or 'camera_matrix'.")

    if "D" in data:
        D = np.asarray(data["D"], dtype=np.float64)
    elif "dist_coeffs" in data:
        D = np.asarray(data["dist_coeffs"], dtype=np.float64)
    else:
        # Minimal default (4 is enough for omnidir)
        D = np.zeros((4,), dtype=np.float64)

    if "xi" not in data:
        raise KeyError("Intrinsics JSON must contain 'xi' for omni undistort.")
    xi = float(data["xi"])

    D = D.reshape(-1, 1)
    return K, D, xi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lens",
        default="lens1",
        help="Lens folder under data/images/ (e.g. lens1, lens2, ...).",
    )
    parser.add_argument(
        "--intrinsics",
        default=os.path.join(INTRINSICS_ROOT, "calibration_omni.json"),
        help="Path to omni intrinsics JSON (K, D, xi).",
    )
    parser.add_argument(
        "--out_subdir",
        default="undistorted",
        help="Sub-folder name inside lens directory for saving undistorted images.",
    )
    parser.add_argument(
        "--fov_scale",
        type=float,
        default=0.5,
        help=(
            "Scale factor for new focal length. "
            "K_new[0,0] and K_new[1,1] are multiplied by this. "
            "Smaller value = wider FOV. Default = 0.5."
        ),
    )
    args = parser.parse_args()

    lens = args.lens
    lens_dir = os.path.join(IMAGES_ROOT, lens)
    if not os.path.isdir(lens_dir):
        raise SystemExit(f"[ERROR] Lens directory not found: {lens_dir}")

    out_dir = os.path.join(lens_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Lens directory   : {lens_dir}")
    print(f"[INFO] Output directory : {out_dir}")

    # Load omni intrinsics
    K, D, xi = load_omni_intrinsics(args.intrinsics)
    print("[INTR] K =\n", K)
    print("[INTR] D =", D.ravel())
    print("[INTR] xi =", xi)

    # Wrap xi as 1-element array for cv2.omnidir.undistortImage
    xi_vec = np.array([xi], dtype=np.float64)

    # Collect input images
    image_files = [
        f for f in sorted(os.listdir(lens_dir))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
        and f.startswith("cam_")
    ]
    if not image_files:
        raise SystemExit(f"[ERROR] No cam_*.png/jpg images found in {lens_dir}")

    print(f"[INFO] Found {len(image_files)} input images.")

    # Use the first image to define output resolution and base K_new
    first_path = os.path.join(lens_dir, image_files[0])
    first_img = cv2.imread(first_path, cv2.IMREAD_COLOR)
    if first_img is None:
        raise SystemExit(f"[ERROR] Could not read first image: {first_path}")

    H, W = first_img.shape[:2]
    scale = float(args.fov_scale)

    K_new = K.copy()
    K_new[0, 0] *= scale
    K_new[1, 1] *= scale
    K_new[0, 2] = W / 2.0
    K_new[1, 2] = H / 2.0

    print("[INTR_NEW] K_new (pinhole, rectified) =\n", K_new)
    print("[INTR_NEW] Output size (W, H) =", (W, H))

    # Save new pinhole intrinsics
    pinhole_json = os.path.join(INTRINSICS_ROOT, f"calibration_pinhole_{lens}.json")
    pinhole_intr = {
        "K": K_new.tolist(),
        "D": [0.0, 0.0, 0.0, 0.0, 0.0],
        "model": "pinhole_from_omni",
        "source_omni_json": os.path.relpath(args.intrinsics, INTRINSICS_ROOT),
        "image_size": [int(W), int(H)],
    }
    with open(pinhole_json, "w") as f:
        json.dump(pinhole_intr, f, indent=2)

    print(f"[WRITE] Saved pinhole intrinsics to:\n  {pinhole_json}")

    # Process images
    processed = 0
    for fname in image_files:
        in_path = os.path.join(lens_dir, fname)
        img = cv2.imread(in_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Could not read image, skipping: {in_path}")
            continue

        h, w = img.shape[:2]
        if (h, w) != (H, W):
            print(
                f"[WARN] Image {fname} has size {(w, h)}, expected {(W, H)}; "
                "recomputing K_new for this size."
            )
            H, W = h, w
            K_new = K.copy()
            K_new[0, 0] *= scale
            K_new[1, 1] *= scale
            K_new[0, 2] = W / 2.0
            K_new[1, 2] = H / 2.0

        rectify_flag = cv2.omnidir.RECTIFY_PERSPECTIVE

        undistorted = cv2.omnidir.undistortImage(
            img,
            K,
            D,
            xi_vec,
            rectify_flag,
            Knew=K_new,
            new_size=(W, H),
        )

        if undistorted.dtype != np.uint8:
            print("[INFO] Converting undistorted image to uint8")
            min_val = undistorted.min()
            max_val = undistorted.max()
            if max_val > min_val:
                undistorted = ((undistorted - min_val) * 255.0 / (max_val - min_val)).astype(np.uint8)
            else:
                undistorted = np.zeros_like(undistorted, dtype=np.uint8)
        
        
        # write as .png format
        
        name, _ = os.path.splitext(fname)
        out_path = os.path.join(out_dir, f"{name}.png") 
        cv2.imwrite(out_path, undistorted)
        
        # out_path = os.path.join(out_dir, fname)
        # cv2.imwrite(out_path, undistorted)
        processed += 1
        print(f"[SAVE] {fname} -> {out_path}")
        
        
        
    print(f"\n[DONE] Undistorted {processed} images for lens '{lens}'.")
    print(f"       Use calibration_pinhole_{lens}.json for further extrinsic solving.")


if __name__ == "__main__":
    main()