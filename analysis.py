from __future__ import annotations

import cv2
import numpy as np

DIR_HORIZONTAL = 0
DIR_VERTICAL = 1
DIR_DIAG_POS = 2  # "/" edge orientation
DIR_DIAG_NEG = 3  # "\\" edge orientation


def _to_gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if gray.dtype == np.uint8:
        return gray.astype(np.float32) / 255.0

    gray_f = gray.astype(np.float32)
    max_val = np.max(gray_f) if gray_f.size else 1.0
    if max_val > 1.0:
        gray_f /= 255.0
    return gray_f


def compute_structure_tensor(
    gray: np.ndarray,
    sigma: float = 1.2,
    sobel_ksize: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray_f = gray.astype(np.float32)

    gx = cv2.Sobel(
        gray_f,
        cv2.CV_32F,
        1,
        0,
        ksize=sobel_ksize,
        borderType=cv2.BORDER_REFLECT101,
    )
    gy = cv2.Sobel(
        gray_f,
        cv2.CV_32F,
        0,
        1,
        ksize=sobel_ksize,
        borderType=cv2.BORDER_REFLECT101,
    )

    jxx = cv2.GaussianBlur(
        gx * gx,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    )
    jxy = cv2.GaussianBlur(
        gx * gy,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    )
    jyy = cv2.GaussianBlur(
        gy * gy,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    )

    return jxx, jxy, jyy


def tensor_orientation_and_coherence(
    jxx: np.ndarray,
    jxy: np.ndarray,
    jyy: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    two_jxy = 2.0 * jxy
    theta = 0.5 * np.arctan2(two_jxy, jxx - jyy)

    trace = jxx + jyy
    delta = np.sqrt((jxx - jyy) * (jxx - jyy) + two_jxy * two_jxy)

    lambda1 = 0.5 * (trace + delta)
    lambda2 = 0.5 * (trace - delta)
    coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + eps)

    return theta.astype(np.float32), coherence.astype(np.float32)


def quantize_orientation(theta: np.ndarray) -> np.ndarray:
    angle = np.mod(theta + np.pi, np.pi)
    direction = np.zeros_like(angle, dtype=np.uint8)

    pi = np.pi
    direction[(angle >= (pi / 8.0)) & (angle < (3.0 * pi / 8.0))] = DIR_DIAG_NEG
    direction[(angle >= (3.0 * pi / 8.0)) & (angle < (5.0 * pi / 8.0))] = DIR_VERTICAL
    direction[(angle >= (5.0 * pi / 8.0)) & (angle < (7.0 * pi / 8.0))] = DIR_DIAG_POS

    return direction


def analyze_local_structure(
    image: np.ndarray,
    sigma: float = 1.2,
    coherence_threshold: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gray = _to_gray_float(image)
    jxx, jxy, jyy = compute_structure_tensor(gray, sigma=sigma)
    theta, coherence = tensor_orientation_and_coherence(jxx, jxy, jyy)
    direction_map = quantize_orientation(theta)

    strong_edge_mask = coherence >= coherence_threshold
    return direction_map, strong_edge_mask, theta, coherence