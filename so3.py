"""회전 벡터 <-> 회전 행렬."""

import numpy as np


def exp_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def log_so3(R):
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    if th < 1e-9:
        return v
    if np.pi - th < 1e-6:  # 180° 근처: 대각 성분에서 축을 구한다
        k = np.sqrt(np.maximum((np.diag(R) + 1) / 2, 0))
        i = int(np.argmax(k))
        k = R[:, i] + np.eye(3)[i]
        return th * k / np.linalg.norm(k)
    return v * th / np.sin(th)
