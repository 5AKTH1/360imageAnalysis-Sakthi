import numpy as np
import struct
import math
import dpkt

_cache = {}

def load_pcap(path, t_min=None, t_max=None):
    cache_key = f"{path}_{t_min}_{t_max}"
    if cache_key in _cache: return _cache[cache_key]
    
    omega = [math.radians(a) for a in [-15, 1, -13, 3, -11, 5, -9, 7, -7, 9, -5, 11, -3, 13, -1, 15]]
    cos_omega, sin_omega = np.cos(omega), np.sin(omega)
    chunks, points = [], []

    with open(path, "rb") as f:
        pcap = dpkt.pcap.Reader(f)
        for ts, buf in pcap:
            if t_min is not None and ts < t_min: continue
            if t_max is not None and ts > t_max: continue
            if len(buf) < 42: continue
            payload = buf[42:]
            if len(payload) != 1206: continue

            # THE FIX: Convert to relative time to save float32 precision
            rel_ts = ts - t_min if t_min is not None else ts

            for i in range(12):
                block_offset = i * 100
                if struct.unpack_from('<H', payload, block_offset)[0] != 0xEEFF: continue
                azimuth = struct.unpack_from('<H', payload, block_offset + 2)[0] / 100.0
                alpha = math.radians(azimuth)
                cos_alpha, sin_alpha = math.cos(alpha), math.sin(alpha)

                for firing in range(2):
                    for laser_id in range(16):
                        data_offset = block_offset + 4 + (firing * 48) + (laser_id * 3)
                        distance = struct.unpack_from('<H', payload, data_offset)[0] * 0.002
                        if distance > 1.0:
                            points.append((rel_ts, distance * cos_omega[laser_id] * sin_alpha, 
                                           distance * cos_omega[laser_id] * cos_alpha, 
                                           distance * sin_omega[laser_id]))

            if len(points) >= 500000:
                chunks.append(np.array(points, dtype=np.float32))
                points = []

    if points: chunks.append(np.array(points, dtype=np.float32))
    if not chunks: return None
    
    result = np.vstack(chunks)
    _cache[cache_key] = result
    return result

def load_both(l1_path, l2_path, t_min=None, t_max=None):
    pts1 = load_pcap(l1_path, t_min, t_max)
    pts2 = load_pcap(l2_path, t_min, t_max)

    if pts1 is None: pts1 = np.empty((0, 4), dtype=np.float32)
    if pts2 is None: pts2 = np.empty((0, 4), dtype=np.float32)

    if len(pts2) > 0:
        T_L2_to_L1 = np.array([
            [ 0.99975943,  0.02096891, -0.00643553, -0.1137049 ],
            [-0.02102737,  0.99973696, -0.00915438,  0.91938674],
            [ 0.00624188,  0.0092875 ,  0.99993742, -0.00164386],
            [ 0.0,         0.0,         0.0,         1.0       ]
        ], dtype=np.float32)

        R, t = T_L2_to_L1[:3, :3], T_L2_to_L1[:3, 3]
        for i in range(0, len(pts2), 10_000_000):
            end = min(i + 10_000_000, len(pts2))
            pts2[i:end, 1:4] = np.dot(pts2[i:end, 1:4], R.T) + t

    # Sort arrays individually so binary search works perfectly
    if len(pts1) > 0: pts1 = pts1[pts1[:, 0].argsort()]
    if len(pts2) > 0: pts2 = pts2[pts2[:, 0].argsort()]

    return pts1, pts2