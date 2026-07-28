import heapq
import logging
from dataclasses import dataclass
import numpy as np

@dataclass
class Image:
    img: np.ndarray
    rot: np.ndarray
    intr: np.ndarray
    range: tuple = (np.zeros(2), np.zeros(2))
    def hom(self): return self.rot.T.dot(np.linalg.inv(self.intr))
    def proj(self): return self.intr.dot(self.rot)

def intrinsics(focal, center=(0, 0)):
    if not isinstance(focal, (list, tuple)): focal = (focal,)*2
    return np.array([[focal[0], 0, center[0]], [0, focal[1], center[1]], [0, 0, 1]])

def rotation_to_mat(rad):
    ang = np.linalg.norm(rad)
    if ang < 1e-8: return np.eye(3)
    vec = rad / ang
    cross = np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])
    return np.eye(3) + cross*np.sin(ang) + (1-np.cos(ang))*cross.dot(cross)

def to_rotation(rot):
    uu_, _, vv_ = np.linalg.svd(rot)
    res = uu_.dot(vv_)
    if np.linalg.det(res) < 0: res *= -1
    return res

def _hom_to_from(cm1, cm2):
    return (cm1.intr.dot(cm1.rot)).dot(cm2.rot.T.dot(np.linalg.inv(cm2.intr)))

def traverse(imgs, matches, badjust="incr", use_straighten=True):
    # Find starting point based on best match scores
    idx, homs, scores = zip(*[(i, matches[i][j][1], len(matches[i][j][0])) 
                             for i in matches.keys() for j in matches[i].keys()])
    src = idx[np.argmax(scores)]
    
    # Estimate global focal
    focals = []
    for i in matches.keys():
        for j in matches[i].keys():
            h = matches[i][j][1].ravel()
            f2 = abs((h[0]*h[1]+h[3]*h[4])/-(h[6]*h[7]+1e-8))
            if f2 > 0: focals.append(np.sqrt(f2))
    intr = intrinsics(np.median(focals) if focals else 1000)

    # Simplified camera tracking for stability
    cameras = [None] * len(imgs)
    cameras[src] = Image(None, np.eye(3), intr)
    
    qq_ = [(-len(matches[src][j][0]), src, j) for j in matches[src].keys()]
    heapq.heapify(qq_)

    while qq_:
        _, s, d = heapq.heappop(qq_)
        if cameras[d] is not None: continue
        hom = matches[s][d][1]
        rot = to_rotation(np.linalg.inv(intr).dot(hom.dot(intr)))
        cameras[d] = Image(None, rot.dot(cameras[s].rot), intr)
        for n in matches[d].keys():
            heapq.heappush(qq_, (-len(matches[d][n][0]), d, n))

    for idx, img in enumerate(imgs):
        if cameras[idx] is not None: cameras[idx].img = img
    return [c for c in cameras if c is not None]