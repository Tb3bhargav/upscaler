from __future__ import annotations

import argparse
import math

import cv2
import numpy as np
from numba import njit, prange

from analysis import (
    DIR_DIAG_NEG,
    DIR_DIAG_POS,
    DIR_HORIZONTAL,
    DIR_VERTICAL,
    analyze_local_structure,
)
from kernels import cubic_kernel, lanczos_kernel


@njit(cache=True)
def _clamp_index(idx: int, size: int) -> int:
    if idx < 0:
        return 0
    if idx >= size:
        return size - 1
    return idx


@njit(cache=True, fastmath=True)
def _bilinear_sample(img: np.ndarray, y: float, x: float) -> float:
    h, w = img.shape

    if y < 0.0:
        y = 0.0
    elif y > (h - 1):
        y = h - 1.0

    if x < 0.0:
        x = 0.0
    elif x > (w - 1):
        x = w - 1.0

    y0 = int(math.floor(y))
    x0 = int(math.floor(x))
    y1 = _clamp_index(y0 + 1, h)
    x1 = _clamp_index(x0 + 1, w)

    wy = y - y0
    wx = x - x0

    v00 = img[y0, x0]
    v01 = img[y0, x1]
    v10 = img[y1, x0]
    v11 = img[y1, x1]

    top = v00 * (1.0 - wx) + v01 * wx
    bottom = v10 * (1.0 - wx) + v11 * wx
    return top * (1.0 - wy) + bottom * wy


@njit(cache=True, fastmath=True)
def _directional_cubic(img: np.ndarray, src_y: float, src_x: float, direction: int) -> float:
    if direction == DIR_HORIZONTAL:
        mode = 0
        c = src_y
        t = src_x
    elif direction == DIR_VERTICAL:
        mode = 1
        c = src_x
        t = src_y
    elif direction == DIR_DIAG_POS:
        mode = 2
        c = src_x + src_y
        t = 0.5 * (src_x - src_y)
    else:
        mode = 3
        c = src_x - src_y
        t = 0.5 * (src_x + src_y)

    t0 = int(math.floor(t))
    acc = 0.0
    wsum = 0.0

    for offset in range(-1, 3):
        tk = t0 + offset
        weight = cubic_kernel(t - tk)

        if mode == 0:
            sy = c
            sx = tk
        elif mode == 1:
            sy = tk
            sx = c
        elif mode == 2:
            sx = tk + 0.5 * c
            sy = 0.5 * c - tk
        else:
            sx = tk + 0.5 * c
            sy = tk - 0.5 * c

        sample = _bilinear_sample(img, sy, sx)
        acc += sample * weight
        wsum += weight

    if abs(wsum) < 1e-8:
        return _bilinear_sample(img, src_y, src_x)
    return acc / wsum


@njit(cache=True, fastmath=True)
def _lanczos6_2d(img: np.ndarray, src_y: float, src_x: float) -> float:
    a = 6
    h, w = img.shape

    iy = int(math.floor(src_y))
    ix = int(math.floor(src_x))

    y_start = iy - a + 1
    y_end = iy + a
    x_start = ix - a + 1
    x_end = ix + a

    acc = 0.0
    wsum = 0.0

    for yy in range(y_start, y_end + 1):
        wy = lanczos_kernel(src_y - yy, a)
        if abs(wy) < 1e-12:
            continue

        yyc = _clamp_index(yy, h)

        for xx in range(x_start, x_end + 1):
            wx = lanczos_kernel(src_x - xx, a)
            if abs(wx) < 1e-12:
                continue

            wxy = wy * wx
            xxc = _clamp_index(xx, w)
            acc += img[yyc, xxc] * wxy
            wsum += wxy

    if abs(wsum) < 1e-8:
        return _bilinear_sample(img, src_y, src_x)
    return acc / wsum


@njit(cache=True, fastmath=True, parallel=True)
def _dcci_interpolate_channel(
    channel: np.ndarray,
    direction_map: np.ndarray,
    strong_edges: np.ndarray,
    scale: int,
) -> np.ndarray:
    h, w = channel.shape
    out_h = h * scale
    out_w = w * scale
    out = np.empty((out_h, out_w), dtype=np.float32)

    for y in prange(out_h):
        src_y = (y + 0.5) / scale - 0.5
        cy = _clamp_index(int(math.floor(src_y + 0.5)), h)

        for x in range(out_w):
            src_x = (x + 0.5) / scale - 0.5
            cx = _clamp_index(int(math.floor(src_x + 0.5)), w)

            if strong_edges[cy, cx]:
                out[y, x] = _directional_cubic(
                    channel,
                    src_y,
                    src_x,
                    int(direction_map[cy, cx]),
                )
            else:
                out[y, x] = _lanczos6_2d(channel, src_y, src_x)

    return out


def _to_float_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0

    image_f = image.astype(np.float32)
    max_val = np.max(image_f) if image_f.size else 1.0
    if max_val > 1.0:
        # ponytail: unknown integer depth — normalize by actual max, not hard-coded 255.
        image_f /= max_val
    return image_f


def dcci_interpolate(
    image: np.ndarray,
    direction_map: np.ndarray,
    strong_edges: np.ndarray,
    scale: int,
) -> np.ndarray:
    image_f = np.ascontiguousarray(image.astype(np.float32))
    direction_map_u8 = np.ascontiguousarray(direction_map.astype(np.uint8))
    strong_edges_b = np.ascontiguousarray(strong_edges.astype(np.bool_))

    if image_f.ndim == 2:
        return _dcci_interpolate_channel(image_f, direction_map_u8, strong_edges_b, scale)

    out_h = image_f.shape[0] * scale
    out_w = image_f.shape[1] * scale
    out = np.empty((out_h, out_w, image_f.shape[2]), dtype=np.float32)
    for channel_idx in range(image_f.shape[2]):
        out[:, :, channel_idx] = _dcci_interpolate_channel(
            image_f[:, :, channel_idx],
            direction_map_u8,
            strong_edges_b,
            scale,
        )
    return out


def ibp_refine(
    low_res: np.ndarray,
    direction_map: np.ndarray,
    strong_edges: np.ndarray,
    scale: int,
    iterations: int = 3,
) -> np.ndarray:
    if iterations != 3:
        raise ValueError("IBP loop is fixed to exactly 3 iterations in this architecture.")

    high_res = dcci_interpolate(low_res, direction_map, strong_edges, scale)

    low_h, low_w = low_res.shape[:2]
    for _ in range(iterations):
        low_prime = cv2.resize(high_res, (low_w, low_h), interpolation=cv2.INTER_AREA)
        residual = low_res - low_prime
        residual_up = dcci_interpolate(residual, direction_map, strong_edges, scale)
        high_res = np.clip(high_res + residual_up, 0.0, 1.0)

    return high_res


def guided_refine(
    high_res: np.ndarray,
    low_res: np.ndarray,
    radius: int = 6,
    eps: float = 1e-3,
) -> np.ndarray:
    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "guidedFilter"):
        raise ImportError(
            "cv2.ximgproc.guidedFilter is unavailable. Install opencv-contrib-python."
        )

    out_h, out_w = high_res.shape[:2]
    guide_hr = cv2.resize(low_res, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

    refined = cv2.ximgproc.guidedFilter(
        guide=guide_hr.astype(np.float32),
        src=high_res.astype(np.float32),
        radius=radius,
        eps=eps,
        dDepth=-1,
    )
    return np.clip(refined, 0.0, 1.0)


def guided_refine_with_fallback(
    high_res: np.ndarray,
    low_res: np.ndarray,
    radius: int = 6,
    eps: float = 1e-3,
) -> np.ndarray:
    # opencv-contrib is optional; without it just skip the guided pass.
    try:
        return guided_refine(high_res, low_res, radius=radius, eps=eps)
    except ImportError:
        return np.clip(high_res, 0.0, 1.0)


def upscale_ibp_dcci(
    image: np.ndarray,
    scale: int = 2,
    tensor_sigma: float = 1.2,
    edge_threshold: float = 0.25,
    guided_radius: int = 6,
    guided_eps: float = 1e-3,
    ibp_iterations: int = 3,
) -> np.ndarray:
    if scale < 2:
        raise ValueError("Scale must be >= 2.")

    alpha_channel = None
    work_image = image
    if image.ndim == 3 and image.shape[2] == 4:
        work_image = image[:, :, :3]
        alpha_channel = image[:, :, 3]

    low_res = _to_float_image(work_image)
    direction_map, strong_edges, _, _ = analyze_local_structure(
        work_image,
        sigma=tensor_sigma,
        coherence_threshold=edge_threshold,
    )

    high_res = ibp_refine(
        low_res,
        direction_map,
        strong_edges,
        scale=scale,
        iterations=ibp_iterations,
    )
    high_res = guided_refine_with_fallback(high_res, low_res, radius=guided_radius, eps=guided_eps)

    output = np.clip(np.round(high_res * 255.0), 0, 255).astype(np.uint8)

    if alpha_channel is not None:
        alpha_up = cv2.resize(
            alpha_channel,
            (output.shape[1], output.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        output = np.dstack((output, alpha_up))

    return output


def upscale_file(
    input_path: str,
    output_path: str,
    scale: int,
    tensor_sigma: float,
    edge_threshold: float,
    guided_radius: int,
    guided_eps: float,
    ibp_iterations: int,
) -> None:
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not open input image: {input_path}")

    upscaled = upscale_ibp_dcci(
        image,
        scale=scale,
        tensor_sigma=tensor_sigma,
        edge_threshold=edge_threshold,
        guided_radius=guided_radius,
        guided_eps=guided_eps,
        ibp_iterations=ibp_iterations,
    )

    if not cv2.imwrite(output_path, upscaled):
        raise IOError(f"Failed to write output image: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="High-fidelity non-AI image upscaler: DCCI + IBP + Guided Filter",
    )
    parser.add_argument("input", type=str, help="Input image path")
    parser.add_argument("output", type=str, help="Output image path")
    parser.add_argument("--scale", type=int, default=2, help="Upscale factor (integer >= 2)")
    parser.add_argument(
        "--tensor-sigma",
        type=float,
        default=1.2,
        help="Gaussian sigma used to smooth structure tensor terms",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.25,
        help="Coherence threshold for classifying strong directional edges",
    )
    parser.add_argument(
        "--guided-radius",
        type=int,
        default=6,
        help="Guided filter radius",
    )
    parser.add_argument(
        "--guided-eps",
        type=float,
        default=1e-3,
        help="Guided filter epsilon regularization",
    )
    parser.add_argument(
        "--ibp-iterations",
        type=int,
        default=3,
        choices=[3],
        help="Fixed to 3 in this IBP architecture",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    upscale_file(
        input_path=args.input,
        output_path=args.output,
        scale=args.scale,
        tensor_sigma=args.tensor_sigma,
        edge_threshold=args.edge_threshold,
        guided_radius=args.guided_radius,
        guided_eps=args.guided_eps,
        ibp_iterations=args.ibp_iterations,
    )


if __name__ == "__main__":
    main()