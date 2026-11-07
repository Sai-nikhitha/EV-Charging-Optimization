"""Stage 2 - physical constraint enforcement (shared by baseline AND DRL).

Every policy only produces *requests* (kW per EV). `allocate_power` turns requests into
physically valid allocations:
  1. each request is clipped to [0, cap_i], where cap_i already encodes: EV max power,
     "not present" (0) and "energy still needed to reach target" (never overcharge);
  2. if the clipped requests exceed the station limit (= min(station capacity, solar + grid limit)),
     all allocations are scaled by the same factor (proportional, independent of EV order).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AllocationResult:
    allocated_kw: np.ndarray
    limit_kw: float
    limit_excess_kw: float   # how much the (EV-clipped) requests exceeded the station limit


def allocate_power(requests_kw, ev_caps_kw, total_limit_kw: float) -> AllocationResult:
    req = np.nan_to_num(np.asarray(requests_kw, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    req = np.clip(req, 0.0, None)
    caps = np.clip(np.asarray(ev_caps_kw, dtype=float), 0.0, None)
    limit = max(0.0, float(total_limit_kw))
    clipped = np.minimum(req, caps)
    total = float(clipped.sum())
    excess = max(0.0, total - limit)
    if total > limit and total > 0:
        clipped = clipped * (limit / total)
    return AllocationResult(clipped, limit, excess)


def weighted_waterfill(total_kw: float, weights, caps) -> np.ndarray:
    """Share `total_kw` in proportion to `weights`, never exceeding `caps`; leftovers from
    saturated items are redistributed. Result depends only on (weight, cap) values, not on order."""
    w = np.asarray(weights, dtype=float)
    c = np.clip(np.asarray(caps, dtype=float), 0.0, None)
    alloc = np.zeros_like(c)
    remaining = max(0.0, float(total_kw))
    open_ = (c > 1e-12) & (w > 0)
    while remaining > 1e-12 and open_.any():
        idx = np.where(open_)[0]
        share = remaining * w[idx] / w[idx].sum()
        give = np.minimum(share, c[idx] - alloc[idx])
        alloc[idx] += give
        remaining -= float(give.sum())
        saturated = (c[idx] - alloc[idx]) <= 1e-12
        if not saturated.any():
            break
        open_[idx[saturated]] = False
    return alloc
