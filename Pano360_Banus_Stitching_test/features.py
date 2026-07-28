import argparse
import logging
import math
import os
import time
from collections import defaultdict
import numpy as np
import scipy.ndimage as ndi
import cv2

DSIZE = 8
N_MIN_MATCH = 8

def gaussian_filter(img, sigma=1.0):
    ksz = max(int((sigma - 0.35) / 0.15), 1)
    ksz += not ksz % 2
    return cv2.GaussianBlur(img, (ksz, ksz), sigma, sigma)

def ssc(keypoints, im_size, n_points, tol=0.1):
    cols, rows = im_size
    def _high():
        exp1 = rows + cols + 2 * n_points
        exp2 = (4 * cols + 4 * n_points + 4 * rows * n_points + rows * rows + cols * cols - 2 * rows * cols + 4 * rows * cols * n_points)
        exp3 = math.sqrt(exp2)
        exp4 = n_points - 1
        sol1 = -round(float(exp1 + exp3) / exp4)
        sol2 = -round(float(exp1 - exp3) / exp4)
        return max(sol1, sol2)

    high = _high()
    low = math.floor(math.sqrt(len(keypoints) / n_points))
    prev_width, complete, k = -1, False, n_points
    k_min, k_max = round(k - (k * tol)), round(k + (k * tol))
    result = []
    while not complete:
        width = low + (high - low) / 2
        if (width == prev_width or low > high): break
        cgr = width / 2
        n_cell_cols = int(math.floor(cols / cgr))
        n_cell_rows = int(math.floor(rows / cgr))
        covered_vec = np.full((n_cell_rows+1, n_cell_cols+1), False)
        result = []
        for i, kpt in enumerate(keypoints):
            row = int(math.floor(kpt[1] / cgr))
            col = int(math.floor(kpt[0] / cgr))
            if not covered_vec[row][col]:
                result.append(i)
                row_min = int(max(row - math.floor(width / cgr), 0))
                row_max = int(min(row + math.floor(width / cgr), n_cell_rows))
                col_min = int(max(col - math.floor(width / cgr), 0))
                col_max = int(min(col + math.floor(width / cgr), n_cell_cols))
                covered_vec[row_min:row_max+1, col_min:col_max+1] = True
        if k_min <= len(result) <= k_max: complete = True
        elif len(result) < k_min: high = width - 1
        else: low = width + 1
        prev_width = width
    return [keypoints[res] for res in result]

def sift_detector():
    sift = cv2.SIFT_create() # Updated for OpenCV 4.x
    def _detect(img):
        kp_, des = sift.detectAndCompute(img, None)
        if des is None: return [], []
        des = np.sqrt(des/(des.sum(axis=1, keepdims=True) + 1e-7))
        return kp_, des
    return _detect

def flann_matching(des1, des2):
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    return [m for m, n in matches if m.distance < 0.7*n.distance]

def _match_hom(pt1, pt2, des1, des2):
    good = flann_matching(des1, des2)
    match = np.int32([(m.queryIdx, m.trainIdx) for m in good])
    if len(match) < N_MIN_MATCH: return None, None
    query_pts = np.float32([pt1[m] for m, _ in match])
    train_pts = np.float32([pt2[m] for _, m in match])
    hom, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC)
    if mask is None: return None, None
    mask = (mask != 0).squeeze()
    return match[mask, :], hom

def matching(imgs, detect=sift_detector()):
    kpts, descs = [], []
    start = time.time()
    for i, img in enumerate(imgs):
        kp_, des = detect(img)
        cent = np.array([img.shape[1], img.shape[0]]) / 2
        kpts.append(np.float32([kp.pt - cent for kp in kp_]))
        descs.append(des)
    matches, n_imgs = defaultdict(dict), len(imgs)
    for src in range(n_imgs):
        for dst in range(src+1, n_imgs):
            match, hom = _match_hom(kpts[src], kpts[dst], descs[src], descs[dst])
            if hom is None: continue
            matches[src][dst] = (match, hom)
            matches[dst][src] = (np.fliplr(match), np.linalg.inv(hom))
    # Fixed np.object to object for Python 3.11/Numpy 1.24+
    return np.array(kpts, dtype=object), np.array(matches, dtype=object)