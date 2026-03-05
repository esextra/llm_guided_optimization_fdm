# print_quality/utils/graph.py
from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np

try:
    from scipy.ndimage import binary_dilation  # optional but faster
except Exception:  # pragma: no cover
    binary_dilation = None


def connected_components(mask: np.ndarray, connectivity: int = 8) -> Tuple[np.ndarray, int]:
    """
    Label connected components in a binary mask.
    Returns (labels, num_labels). Label 0 is background.
    Implementation: straightforward BFS to avoid external deps; good for moderate sizes.
    """
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=np.int32)
    current = 0

    if connectivity == 4:
        neigh = [(-1,0),(1,0),(0,-1),(0,1)]
    else:
        neigh = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    for y in range(H):
        for x in range(W):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            current += 1
            labels[y, x] = current
            q = deque([(y, x)])
            while q:
                cy, cx = q.popleft()
                for dy, dx in neigh:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = current
                        q.append((ny, nx))
    return labels, current


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Binary dilation with square structuring element of side (2*radius+1).
    Uses scipy.ndimage if available; otherwise falls back to a simple sliding max filter.
    """
    if radius <= 0:
        return mask.copy()
    if binary_dilation is not None:
        from numpy import ones
        se = np.ones((2*radius+1, 2*radius+1), dtype=bool)
        return binary_dilation(mask, structure=se)
    # Fallback: iterative 1-pixel dilations
    out = mask.copy()
    for _ in range(radius):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        # 3x3 max filter
        acc = np.zeros_like(out, dtype=bool)
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                acc |= padded[1+dy:1+dy+out.shape[0], 1+dx:1+dx+out.shape[1]]
        out = acc
    return out


def overlap_fraction(mask_a: np.ndarray, mask_b: np.ndarray, dilate_radius_px: int = 0) -> float:
    """
    Jaccard-like overlap (intersection / area of A) after optionally dilating mask_b by a few pixels.
    Returns 0 if A has zero area.
    """
    if dilate_radius_px > 0:
        mask_b = _binary_dilate(mask_b, dilate_radius_px)
    A = mask_a.sum()
    if A == 0:
        return 0.0
    inter = (mask_a & mask_b).sum()
    return float(inter) / float(A)


def contact_length_pixels(perimeter_mask: np.ndarray, infill_mask: np.ndarray) -> int:
    """
    Approximate contact length between perimeter and infill by counting 4-neighbor adjacencies.
    Returns a length in pixel-edges; multiply by pixel size to get mm.
    """
    H, W = perimeter_mask.shape
    pm = perimeter_mask.astype(np.uint8)
    im = infill_mask.astype(np.uint8)
    # Right neighbor contacts
    right = (pm[:, :-1] & im[:, 1:]).sum()
    left  = (pm[:, 1:]  & im[:, :-1]).sum()
    up    = (pm[:-1, :] & im[1:, :]).sum()
    down  = (pm[1:, :]  & im[:-1, :]).sum()
    return int(right + left + up + down)
