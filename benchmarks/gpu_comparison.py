#!/usr/bin/env python3
"""
GPU acceleration benchmark: compares CPU (NumPy), Partial CuPy, Full CuPy, and PyTorch
implementations of the WAMOS per-frame processing pipeline.

Usage:
    LD_LIBRARY_PATH=/scratch/tpw/WAMOS_tpw/.cuda_compat:/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    python benchmarks/gpu_comparison.py /path/to/POLAR

Measures per-step timing and memory for each backend across multiple iterations.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# ── Ensure project is importable ──
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wamos_tpw.config import Config
from wamos_tpw.polarfile import PolarFile

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# GPU availability checks
# ─────────────────────────────────────────────────────────────────────────────


def _check_cupy():
    try:
        import cupy as cp

        cp.ones(1)  # trigger JIT compile
        return True
    except Exception:
        return False


def _check_torch():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


HAS_CUPY = _check_cupy()
HAS_TORCH = _check_torch()


# ─────────────────────────────────────────────────────────────────────────────
# 1. CPU baseline — uses the existing pipeline exactly as-is
# ─────────────────────────────────────────────────────────────────────────────


def run_cpu(frame, config):
    """Run existing CPU pipeline and return per-step timings."""
    import cv2
    from numpy.polynomial import Polynomial
    from scipy import ndimage

    from wamos_tpw.dewind import Dewind
    from wamos_tpw.pps import PPS
    from wamos_tpw.range import Range
    from wamos_tpw.shadow import Shadow
    from wamos_tpw.theta import Theta

    timings = {}
    t0 = time.perf_counter()

    PPS(frame)
    timings["PPS"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    theta = Theta(frame)
    timings["Theta"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rng = Range(frame)
    timings["Range"] = time.perf_counter() - t0

    # --- Destreak ---
    t0 = time.perf_counter()
    kernel = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)
    kAdjacent = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    kAdjacent = kAdjacent / kAdjacent.sum()

    intensity = frame.intensity.astype(np.float32)
    pad = 1
    padded = np.vstack([intensity[-pad:], intensity, intensity[:pad]])
    a_padded = cv2.filter2D(padded, -1, kernel, borderType=cv2.BORDER_REFLECT)
    b_padded = cv2.filter2D(padded, -1, kAdjacent, borderType=cv2.BORDER_REFLECT)
    a = a_padded[pad:-pad]
    b = b_padded[pad:-pad]
    timings["Destreak.convolve"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    sigma = np.std(a)
    thres_center = 3.0 * sigma
    thres_adjacent = thres_center / 2
    qAdjacent = a <= -thres_adjacent
    qCenter = a >= thres_center
    q = qCenter & np.roll(qAdjacent, +1, axis=0) & np.roll(qAdjacent, -1, axis=0)
    timings["Destreak.threshold"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    qAny = np.any(q, axis=1)
    if qAny.any():
        qStreaks = q[qAny, :]
        kernelH = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
        [labels, _] = ndimage.label(~qStreaks, structure=kernelH)
        cnt = np.bincount(labels.ravel())
        RL = cnt[labels]
        qShort = (RL <= 4) & ~qStreaks
        qStreaks |= qShort
        [labels, _] = ndimage.label(qStreaks, structure=kernelH)
        cnt = np.bincount(labels.ravel())
        RL = cnt[labels]
        qStreaks &= RL >= 10
        q[qAny] = qStreaks
    timings["Destreak.label"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    destreaked = intensity.copy()
    destreaked[q] = b[q]
    timings["Destreak.replace"] = time.perf_counter() - t0

    # --- Shadow ---
    t0 = time.perf_counter()
    shadow = Shadow(destreaked, theta)
    timings["Shadow"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    masked = shadow.mask(destreaked)
    timings["MaskShadow"] = time.perf_counter() - t0

    # --- Deramp ---
    t0 = time.perf_counter()
    mu = np.nanmean(masked, axis=0)
    q_nan = np.isnan(mu)
    slant = rng.slant_range
    x = 1.0 / slant
    p = Polynomial.fit(x[~q_nan], mu[~q_nan], deg=4)
    py = p(x)
    masked -= py[np.newaxis, :]
    timings["Deramp"] = time.perf_counter() - t0

    # --- Dewind ---
    t0 = time.perf_counter()
    dewind = Dewind(masked, theta, copy=False)
    timings["Dewind"] = time.perf_counter() - t0

    # --- Grid projection (standalone benchmark) ---
    t0 = time.perf_counter()
    earth_bearing_rad = np.deg2rad(theta.theta % 360)
    sin_b = np.sin(earth_bearing_rad)
    cos_b = np.cos(earth_bearing_rad)
    gr = rng.ground_range
    x_coords = np.outer(sin_b, gr)
    y_coords = np.outer(cos_b, gr)
    n_x = n_y = 512
    spacing = float(gr[-1]) * 2 / n_x
    inv_sp = 1.0 / spacing
    x_origin = -float(gr[-1])
    y_origin = -float(gr[-1])
    x_idx = ((x_coords - x_origin) * inv_sp).astype(np.int32)
    y_idx = ((y_coords - y_origin) * inv_sp).astype(np.int32)
    x_flat = x_idx.ravel()
    y_flat = y_idx.ravel()
    vals = dewind.intensity.ravel()
    valid = (x_flat >= 0) & (x_flat < n_x) & (y_flat >= 0) & (y_flat < n_y) & ~np.isnan(vals)
    if np.sum(valid) > 0:
        lin = y_flat[valid] * n_x + x_flat[valid]
        gs = n_x * n_y
        _ = np.bincount(lin, weights=vals[valid], minlength=gs)
        _ = np.bincount(lin, minlength=gs)
    timings["GridProjection"] = time.perf_counter() - t0

    return timings


# ─────────────────────────────────────────────────────────────────────────────
# 2. Partial CuPy — GPU for Destreak (convolutions) + Deramp only
# ─────────────────────────────────────────────────────────────────────────────


def run_partial_cupy(frame, config):
    """Destreak convolutions + Deramp on GPU; everything else on CPU."""
    import cupy as cp
    import cupyx.scipy.ndimage as cpx_ndimage
    from numpy.polynomial import Polynomial
    from scipy import ndimage

    from wamos_tpw.dewind import Dewind
    from wamos_tpw.pps import PPS
    from wamos_tpw.range import Range
    from wamos_tpw.shadow import Shadow
    from wamos_tpw.theta import Theta

    timings = {}
    t0 = time.perf_counter()

    PPS(frame)
    timings["PPS"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    theta = Theta(frame)
    timings["Theta"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rng = Range(frame)
    timings["Range"] = time.perf_counter() - t0

    # --- Destreak (GPU convolutions) ---
    t0 = time.perf_counter()
    kernel = cp.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=cp.float32)
    kAdjacent = cp.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=cp.float32)
    kAdjacent = kAdjacent / kAdjacent.sum()

    intensity_np = frame.intensity.astype(np.float32)
    intensity_gpu = cp.asarray(intensity_np)
    pad = 1
    padded = cp.vstack([intensity_gpu[-pad:], intensity_gpu, intensity_gpu[:pad]])
    a_padded = cpx_ndimage.convolve(padded, kernel, mode="reflect")
    b_padded = cpx_ndimage.convolve(padded, kAdjacent, mode="reflect")
    a_gpu = a_padded[pad:-pad]
    b_gpu = b_padded[pad:-pad]
    # Sync to get accurate timing
    cp.cuda.Device(0).synchronize()
    timings["Destreak.convolve"] = time.perf_counter() - t0

    # Threshold + label on CPU (transfer back)
    t0 = time.perf_counter()
    a = cp.asnumpy(a_gpu)
    b = cp.asnumpy(b_gpu)
    sigma = np.std(a)
    thres_center = 3.0 * sigma
    thres_adjacent = thres_center / 2
    qAdjacent = a <= -thres_adjacent
    qCenter = a >= thres_center
    q = qCenter & np.roll(qAdjacent, +1, axis=0) & np.roll(qAdjacent, -1, axis=0)
    timings["Destreak.threshold"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    qAny = np.any(q, axis=1)
    if qAny.any():
        qStreaks = q[qAny, :]
        kernelH = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
        [labels, _] = ndimage.label(~qStreaks, structure=kernelH)
        cnt = np.bincount(labels.ravel())
        RL = cnt[labels]
        qShort = (RL <= 4) & ~qStreaks
        qStreaks |= qShort
        [labels, _] = ndimage.label(qStreaks, structure=kernelH)
        cnt = np.bincount(labels.ravel())
        RL = cnt[labels]
        qStreaks &= RL >= 10
        q[qAny] = qStreaks
    timings["Destreak.label"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    destreaked = intensity_np.copy()
    destreaked[q] = b[q]
    timings["Destreak.replace"] = time.perf_counter() - t0

    # --- Shadow (CPU) ---
    t0 = time.perf_counter()
    shadow = Shadow(destreaked, theta)
    timings["Shadow"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    masked = shadow.mask(destreaked)
    timings["MaskShadow"] = time.perf_counter() - t0

    # --- Deramp (GPU) ---
    t0 = time.perf_counter()
    masked_gpu = cp.asarray(masked)
    mu = cp.nanmean(masked_gpu, axis=0)
    q_nan = cp.isnan(mu)
    slant = rng.slant_range
    x_np = 1.0 / slant
    # Polynomial fit on CPU (small array)
    mu_cpu = cp.asnumpy(mu)
    q_nan_cpu = cp.asnumpy(q_nan)
    p = Polynomial.fit(x_np[~q_nan_cpu], mu_cpu[~q_nan_cpu], deg=4)
    py = cp.asarray(p(x_np).astype(np.float32))
    masked_gpu -= py[cp.newaxis, :]
    masked[:] = cp.asnumpy(masked_gpu)
    cp.cuda.Device(0).synchronize()
    timings["Deramp"] = time.perf_counter() - t0

    # --- Dewind (CPU) ---
    t0 = time.perf_counter()
    dewind = Dewind(masked, theta, copy=False)
    timings["Dewind"] = time.perf_counter() - t0

    # --- Grid projection (CPU, same as baseline) ---
    t0 = time.perf_counter()
    earth_bearing_rad = np.deg2rad(theta.theta % 360)
    sin_b = np.sin(earth_bearing_rad)
    cos_b = np.cos(earth_bearing_rad)
    gr = rng.ground_range
    x_coords = np.outer(sin_b, gr)
    y_coords = np.outer(cos_b, gr)
    n_x = n_y = 512
    spacing = float(gr[-1]) * 2 / n_x
    inv_sp = 1.0 / spacing
    x_origin = -float(gr[-1])
    y_origin = -float(gr[-1])
    x_idx = ((x_coords - x_origin) * inv_sp).astype(np.int32)
    y_idx = ((y_coords - y_origin) * inv_sp).astype(np.int32)
    x_flat = x_idx.ravel()
    y_flat = y_idx.ravel()
    vals = dewind.intensity.ravel()
    valid = (x_flat >= 0) & (x_flat < n_x) & (y_flat >= 0) & (y_flat < n_y) & ~np.isnan(vals)
    if np.sum(valid) > 0:
        lin = y_flat[valid] * n_x + x_flat[valid]
        gs = n_x * n_y
        _ = np.bincount(lin, weights=vals[valid], minlength=gs)
        _ = np.bincount(lin, minlength=gs)
    timings["GridProjection"] = time.perf_counter() - t0

    return timings


# ─────────────────────────────────────────────────────────────────────────────
# 3. Full CuPy — GPU for Destreak + Deramp + Grid projection + bearing trig
# ─────────────────────────────────────────────────────────────────────────────


def run_full_cupy(frame, config):
    """All heavy computation on GPU via CuPy."""
    import cupy as cp
    import cupyx.scipy.ndimage as cpx_ndimage
    from numpy.polynomial import Polynomial

    from wamos_tpw.dewind import Dewind
    from wamos_tpw.pps import PPS
    from wamos_tpw.range import Range
    from wamos_tpw.shadow import Shadow
    from wamos_tpw.theta import Theta

    timings = {}
    t0 = time.perf_counter()

    PPS(frame)
    timings["PPS"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    theta = Theta(frame)
    timings["Theta"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rng = Range(frame)
    timings["Range"] = time.perf_counter() - t0

    # --- Destreak (fully GPU) ---
    t0 = time.perf_counter()
    kernel = cp.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=cp.float32)
    kAdjacent_k = cp.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=cp.float32)
    kAdjacent_k = kAdjacent_k / kAdjacent_k.sum()

    intensity_gpu = cp.asarray(frame.intensity.astype(np.float32))
    pad = 1
    padded = cp.vstack([intensity_gpu[-pad:], intensity_gpu, intensity_gpu[:pad]])
    a_padded = cpx_ndimage.convolve(padded, kernel, mode="reflect")
    b_padded = cpx_ndimage.convolve(padded, kAdjacent_k, mode="reflect")
    a_gpu = a_padded[pad:-pad]
    b_gpu = b_padded[pad:-pad]
    cp.cuda.Device(0).synchronize()
    timings["Destreak.convolve"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    sigma = float(cp.std(a_gpu))
    thres_center = 3.0 * sigma
    thres_adjacent = thres_center / 2
    qAdjacent_g = a_gpu <= -thres_adjacent
    qCenter_g = a_gpu >= thres_center
    q_gpu = qCenter_g & cp.roll(qAdjacent_g, +1, axis=0) & cp.roll(qAdjacent_g, -1, axis=0)
    cp.cuda.Device(0).synchronize()
    timings["Destreak.threshold"] = time.perf_counter() - t0

    # Label on GPU via cupyx
    t0 = time.perf_counter()
    qAny_gpu = cp.any(q_gpu, axis=1)
    qAny_np = cp.asnumpy(qAny_gpu)
    if qAny_np.any():
        qStreaks_g = q_gpu[qAny_gpu, :]
        kernelH_g = cp.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=cp.int32)
        # cupyx.scipy.ndimage.label works with int/float, convert bool
        labels_g, _ = cpx_ndimage.label((~qStreaks_g).astype(cp.int32), structure=kernelH_g)
        cnt_g = cp.bincount(labels_g.ravel())
        RL_g = cnt_g[labels_g]
        qShort_g = (RL_g <= 4) & ~qStreaks_g
        qStreaks_g = qStreaks_g | qShort_g
        labels_g, _ = cpx_ndimage.label(qStreaks_g.astype(cp.int32), structure=kernelH_g)
        cnt_g = cp.bincount(labels_g.ravel())
        RL_g = cnt_g[labels_g]
        qStreaks_g = qStreaks_g & (RL_g >= 10)
        q_gpu[qAny_gpu] = qStreaks_g
    cp.cuda.Device(0).synchronize()
    timings["Destreak.label"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    destreaked_gpu = intensity_gpu.copy()
    destreaked_gpu[q_gpu] = b_gpu[q_gpu]
    cp.cuda.Device(0).synchronize()
    timings["Destreak.replace"] = time.perf_counter() - t0

    # --- Shadow (CPU — config-driven, fast) ---
    t0 = time.perf_counter()
    destreaked_np = cp.asnumpy(destreaked_gpu)
    shadow = Shadow(destreaked_np, theta)
    timings["Shadow"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    masked_np = shadow.mask(destreaked_np)
    timings["MaskShadow"] = time.perf_counter() - t0

    # --- Deramp (GPU) ---
    t0 = time.perf_counter()
    masked_gpu = cp.asarray(masked_np)
    mu = cp.nanmean(masked_gpu, axis=0)
    q_nan = cp.isnan(mu)
    slant = rng.slant_range
    x_np = 1.0 / slant
    mu_cpu = cp.asnumpy(mu)
    q_nan_cpu = cp.asnumpy(q_nan)
    p = Polynomial.fit(x_np[~q_nan_cpu], mu_cpu[~q_nan_cpu], deg=4)
    py = cp.asarray(p(x_np).astype(np.float32))
    masked_gpu -= py[cp.newaxis, :]
    cp.cuda.Device(0).synchronize()
    timings["Deramp"] = time.perf_counter() - t0

    # --- Dewind (CPU — uses theta lookup, fast) ---
    t0 = time.perf_counter()
    dewind_input = cp.asnumpy(masked_gpu)
    dewind = Dewind(dewind_input, theta, copy=False)
    timings["Dewind"] = time.perf_counter() - t0

    # --- Grid projection (GPU) ---
    t0 = time.perf_counter()
    theta_gpu = cp.asarray(theta.theta.astype(np.float32))
    earth_bearing_rad = cp.deg2rad(theta_gpu % 360)
    sin_b = cp.sin(earth_bearing_rad)
    cos_b = cp.cos(earth_bearing_rad)
    gr = cp.asarray(rng.ground_range.astype(np.float32))
    x_coords = cp.outer(sin_b, gr)
    y_coords = cp.outer(cos_b, gr)
    n_x = n_y = 512
    spacing = float(gr[-1]) * 2 / n_x
    inv_sp = 1.0 / spacing
    x_origin = -float(gr[-1])
    y_origin = -float(gr[-1])
    x_idx = ((x_coords - x_origin) * inv_sp).astype(cp.int32)
    y_idx = ((y_coords - y_origin) * inv_sp).astype(cp.int32)
    x_flat = x_idx.ravel()
    y_flat = y_idx.ravel()
    vals = cp.asarray(dewind.intensity.astype(np.float32)).ravel()
    valid = (x_flat >= 0) & (x_flat < n_x) & (y_flat >= 0) & (y_flat < n_y) & ~cp.isnan(vals)
    if int(cp.sum(valid)) > 0:
        lin = y_flat[valid] * n_x + x_flat[valid]
        gs = n_x * n_y
        frame_sum = cp.zeros(gs, dtype=cp.float64)
        frame_cnt = cp.zeros(gs, dtype=cp.int32)
        # scatter_add for atomic accumulation
        cp.add.at(frame_sum, lin, vals[valid].astype(cp.float64))
        cp.add.at(frame_cnt, lin, 1)
    cp.cuda.Device(0).synchronize()
    timings["GridProjection"] = time.perf_counter() - t0

    return timings


# ─────────────────────────────────────────────────────────────────────────────
# 4. PyTorch — GPU for Destreak + Deramp + Grid projection
# ─────────────────────────────────────────────────────────────────────────────


def run_pytorch(frame, config):
    """All heavy computation on GPU via PyTorch."""

    import torch
    import torch.nn.functional as F
    from numpy.polynomial import Polynomial
    from scipy import ndimage

    from wamos_tpw.dewind import Dewind
    from wamos_tpw.pps import PPS
    from wamos_tpw.range import Range
    from wamos_tpw.shadow import Shadow
    from wamos_tpw.theta import Theta

    device = torch.device("cuda")

    timings = {}
    t0 = time.perf_counter()

    PPS(frame)
    timings["PPS"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    theta = Theta(frame)
    timings["Theta"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rng = Range(frame)
    timings["Range"] = time.perf_counter() - t0

    # --- Destreak (PyTorch conv2d) ---
    t0 = time.perf_counter()
    kernel_np = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)
    kAdj_np = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    kAdj_np = kAdj_np / kAdj_np.sum()

    intensity_np = frame.intensity.astype(np.float32)
    pad = 1
    padded_np = np.vstack([intensity_np[-pad:], intensity_np, intensity_np[:pad]])

    # PyTorch conv2d expects (N, C, H, W) format
    padded_t = torch.from_numpy(padded_np).unsqueeze(0).unsqueeze(0).to(device)
    kernel_t = torch.from_numpy(kernel_np).unsqueeze(0).unsqueeze(0).to(device)
    kAdj_t = torch.from_numpy(kAdj_np).unsqueeze(0).unsqueeze(0).to(device)

    # F.conv2d with padding=1 for reflect-like border (pad order: left, right, top, bottom)
    a_t = F.conv2d(F.pad(padded_t, (1, 1, 1, 1), mode="reflect"), kernel_t)
    b_t = F.conv2d(F.pad(padded_t, (1, 1, 1, 1), mode="reflect"), kAdj_t)

    a_padded_t = a_t.squeeze()
    b_padded_t = b_t.squeeze()
    a_out = a_padded_t[pad:-pad]
    b_out = b_padded_t[pad:-pad]
    torch.cuda.synchronize()
    timings["Destreak.convolve"] = time.perf_counter() - t0

    # Threshold + label on CPU (PyTorch doesn't have connected component labeling)
    t0 = time.perf_counter()
    a = a_out.cpu().numpy()
    b = b_out.cpu().numpy()
    sigma = np.std(a)
    thres_center = 3.0 * sigma
    thres_adjacent = thres_center / 2
    qAdjacent = a <= -thres_adjacent
    qCenter = a >= thres_center
    q = qCenter & np.roll(qAdjacent, +1, axis=0) & np.roll(qAdjacent, -1, axis=0)
    timings["Destreak.threshold"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    qAny = np.any(q, axis=1)
    if qAny.any():
        qStreaks = q[qAny, :]
        kernelH = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
        [labels, _] = ndimage.label(~qStreaks, structure=kernelH)
        cnt = np.bincount(labels.ravel())
        RL = cnt[labels]
        qShort = (RL <= 4) & ~qStreaks
        qStreaks |= qShort
        [labels, _] = ndimage.label(qStreaks, structure=kernelH)
        cnt = np.bincount(labels.ravel())
        RL = cnt[labels]
        qStreaks &= RL >= 10
        q[qAny] = qStreaks
    timings["Destreak.label"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    destreaked = intensity_np.copy()
    destreaked[q] = b[q]
    timings["Destreak.replace"] = time.perf_counter() - t0

    # --- Shadow (CPU) ---
    t0 = time.perf_counter()
    shadow = Shadow(destreaked, theta)
    timings["Shadow"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    masked = shadow.mask(destreaked)
    timings["MaskShadow"] = time.perf_counter() - t0

    # --- Deramp (GPU) ---
    t0 = time.perf_counter()
    masked_t = torch.from_numpy(masked).to(device)
    mu_t = torch.nanmean(masked_t, dim=0)
    q_nan_t = torch.isnan(mu_t)
    slant = rng.slant_range
    x_np = 1.0 / slant
    mu_cpu = mu_t.cpu().numpy()
    q_nan_cpu = q_nan_t.cpu().numpy()
    p = Polynomial.fit(x_np[~q_nan_cpu], mu_cpu[~q_nan_cpu], deg=4)
    py_t = torch.from_numpy(p(x_np).astype(np.float32)).to(device)
    masked_t -= py_t.unsqueeze(0)
    torch.cuda.synchronize()
    timings["Deramp"] = time.perf_counter() - t0

    # --- Dewind (CPU) ---
    t0 = time.perf_counter()
    dewind_input = masked_t.cpu().numpy()
    dewind = Dewind(dewind_input, theta, copy=False)
    timings["Dewind"] = time.perf_counter() - t0

    # --- Grid projection (GPU) ---
    t0 = time.perf_counter()
    theta_t = torch.from_numpy(theta.theta.astype(np.float32)).to(device)
    earth_bearing_rad = torch.deg2rad(theta_t % 360)
    sin_b = torch.sin(earth_bearing_rad)
    cos_b = torch.cos(earth_bearing_rad)
    gr_t = torch.from_numpy(rng.ground_range.astype(np.float32)).to(device)
    x_coords = torch.outer(sin_b, gr_t)
    y_coords = torch.outer(cos_b, gr_t)
    n_x = n_y = 512
    spacing = float(gr_t[-1]) * 2 / n_x
    inv_sp = 1.0 / spacing
    x_origin = -float(gr_t[-1])
    y_origin = -float(gr_t[-1])
    x_idx = ((x_coords - x_origin) * inv_sp).to(torch.int64)
    y_idx = ((y_coords - y_origin) * inv_sp).to(torch.int64)
    x_flat = x_idx.ravel()
    y_flat = y_idx.ravel()
    vals = torch.from_numpy(dewind.intensity.astype(np.float32)).to(device).ravel()
    valid = (x_flat >= 0) & (x_flat < n_x) & (y_flat >= 0) & (y_flat < n_y) & ~torch.isnan(vals)
    if int(valid.sum()) > 0:
        lin = y_flat[valid] * n_x + x_flat[valid]
        gs = n_x * n_y
        frame_sum = torch.zeros(gs, dtype=torch.float64, device=device)
        frame_sum.scatter_add_(0, lin, vals[valid].to(torch.float64))
    torch.cuda.synchronize()
    timings["GridProjection"] = time.perf_counter() - t0

    return timings


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

BACKENDS = {
    "CPU (NumPy)": run_cpu,
    "Partial CuPy": run_partial_cupy,
    "Full CuPy": run_full_cupy,
    "PyTorch": run_pytorch,
}

STEP_ORDER = [
    "PPS",
    "Theta",
    "Range",
    "Destreak.convolve",
    "Destreak.threshold",
    "Destreak.label",
    "Destreak.replace",
    "Shadow",
    "MaskShadow",
    "Deramp",
    "Dewind",
    "GridProjection",
]


def get_memory_mb():
    """Get current RSS in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_benchmark(frame, config, backend_name, func, n_warmup=3, n_iter=20):
    """Run a single backend benchmark with warmup and multiple iterations."""
    print(f"\n{'=' * 60}")
    print(f"  Backend: {backend_name}")
    print(f"  Frame shape: {frame.shape}")
    print(f"  Warmup: {n_warmup}, Iterations: {n_iter}")
    print(f"{'=' * 60}")

    # Warmup
    for _i in range(n_warmup):
        _ = func(frame, config)
        gc.collect()

    # Timed iterations
    all_timings = []
    mem_before = get_memory_mb()

    for _i in range(n_iter):
        gc.collect()
        t_start = time.perf_counter()
        timings = func(frame, config)
        t_total = time.perf_counter() - t_start
        timings["TOTAL"] = t_total
        all_timings.append(timings)

    mem_after = get_memory_mb()

    # Aggregate
    result = {
        "backend": backend_name,
        "steps": {},
        "mem_before_mb": mem_before,
        "mem_after_mb": mem_after,
    }

    steps = STEP_ORDER + ["TOTAL"]
    for step in steps:
        vals = [t.get(step, 0.0) for t in all_timings]
        vals_ms = [v * 1000 for v in vals]
        result["steps"][step] = {
            "mean_ms": statistics.mean(vals_ms),
            "std_ms": statistics.stdev(vals_ms) if len(vals_ms) > 1 else 0.0,
            "min_ms": min(vals_ms),
            "max_ms": max(vals_ms),
            "median_ms": statistics.median(vals_ms),
        }

    return result


def print_results(results):
    """Print comparison table."""

    steps = STEP_ORDER + ["TOTAL"]

    # Header
    backends = [r["backend"] for r in results]
    col_w = 16
    step_w = 22
    header = f"{'Step':<{step_w}}"
    for b in backends:
        header += f"  {b:>{col_w}}"
    if len(results) > 1:
        header += f"  {'Speedup':>{col_w}}"

    print(f"\n{'=' * len(header)}")
    print("  PERFORMANCE COMPARISON  (median ms, lower is better)")
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))

    for step in steps:
        row = f"{step:<{step_w}}"
        vals = []
        for r in results:
            v = r["steps"].get(step, {}).get("median_ms", 0.0)
            vals.append(v)
            row += f"  {v:>{col_w}.3f}"

        if len(results) > 1 and vals[0] > 0:
            speedup = vals[0] / vals[-1] if vals[-1] > 0 else float("inf")
            row += f"  {speedup:>{col_w}.2f}x"

        if step == "TOTAL":
            print("-" * len(header))
        print(row)

    # Memory
    print(f"\n{'─' * 60}")
    print("  MEMORY (RSS MB)")
    print(f"{'─' * 60}")
    for r in results:
        print(
            f"  {r['backend']:<20s}  before={r['mem_before_mb']:.0f}  after={r['mem_after_mb']:.0f}  delta={r['mem_after_mb'] - r['mem_before_mb']:.0f}"
        )

    # Summary
    print(f"\n{'─' * 60}")
    print("  SUMMARY")
    print(f"{'─' * 60}")
    cpu_med = results[0]["steps"]["TOTAL"]["median_ms"]
    for r in results:
        med = r["steps"]["TOTAL"]["median_ms"]
        speedup = cpu_med / med if med > 0 else 0
        fps = 1000.0 / med if med > 0 else 0
        print(
            f"  {r['backend']:<20s}  median={med:8.2f} ms  speedup={speedup:5.2f}x  throughput={fps:.1f} fps"
        )


def print_detailed_stats(results):
    """Print detailed per-step statistics."""

    steps = STEP_ORDER + ["TOTAL"]

    for r in results:
        print(f"\n{'─' * 70}")
        print(f"  {r['backend']}  —  Detailed Statistics")
        print(f"{'─' * 70}")
        print(
            f"  {'Step':<22s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Median':>8s}  {'Max':>8s}"
        )
        print(f"  {'':─<22s}  {'':─>8s}  {'':─>8s}  {'':─>8s}  {'':─>8s}  {'':─>8s}")
        for step in steps:
            s = r["steps"].get(step, {})
            if step == "TOTAL":
                print(f"  {'':─<22s}  {'':─>8s}  {'':─>8s}  {'':─>8s}  {'':─>8s}  {'':─>8s}")
            print(
                f"  {step:<22s}  {s.get('mean_ms', 0):8.3f}  {s.get('std_ms', 0):8.3f}"
                f"  {s.get('min_ms', 0):8.3f}  {s.get('median_ms', 0):8.3f}  {s.get('max_ms', 0):8.3f}"
            )


def main():
    parser = argparse.ArgumentParser(description="GPU acceleration benchmark")
    parser.add_argument("polar_path", help="Path to POLAR directory or .pol file")
    parser.add_argument("--config", "-c", type=str, default=None, help="Config YAML")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations (default: 3)")
    parser.add_argument(
        "--iterations", "-n", type=int, default=20, help="Benchmark iterations (default: 20)"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        choices=["cpu", "partial_cupy", "full_cupy", "pytorch"],
        help="Backends to test (default: all available)",
    )
    parser.add_argument("--json", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    config = Config(args.config) if args.config else Config()

    # Find a polar file
    polar_path = Path(args.polar_path)
    if polar_path.is_file():
        filepath = polar_path
    else:
        files = sorted(polar_path.rglob("*.pol*"))
        if not files:
            print(f"ERROR: No .pol files found under {polar_path}")
            sys.exit(1)
        filepath = files[0]

    print(f"Loading: {filepath}")
    pf = PolarFile(str(filepath), config=config)
    frame = pf.frame()
    print(f"Frame shape: {frame.shape}  ({frame.shape[0]} bearings x {frame.shape[1]} distances)")

    # Select backends
    available = {"cpu": ("CPU (NumPy)", run_cpu)}
    if HAS_CUPY:
        available["partial_cupy"] = ("Partial CuPy", run_partial_cupy)
        available["full_cupy"] = ("Full CuPy", run_full_cupy)
    else:
        print("WARNING: CuPy not available, skipping CuPy backends")
    if HAS_TORCH:
        available["pytorch"] = ("PyTorch", run_pytorch)
    else:
        print("WARNING: PyTorch CUDA not available, skipping PyTorch backend")

    to_run = args.backends or list(available.keys())
    selected = [(available[k][0], available[k][1]) for k in to_run if k in available]

    if not selected:
        print("ERROR: No backends available")
        sys.exit(1)

    # Run benchmarks
    results = []
    for name, func in selected:
        r = run_benchmark(frame, config, name, func, n_warmup=args.warmup, n_iter=args.iterations)
        results.append(r)

    # Print results
    print_results(results)
    print_detailed_stats(results)

    # Save JSON
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
