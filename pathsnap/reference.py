"""Reference-only alignment. This module must never be imported by prediction."""

from dataclasses import dataclass

import numpy as np

from .geometry import RoadNetwork, project_segments


@dataclass(frozen=True)
class ReferenceAlignment:
    arc_ids: np.ndarray
    distance_m: np.ndarray
    route_positions: np.ndarray


def align_reference(network: RoadNetwork, route: np.ndarray) -> ReferenceAlignment:
    """Global monotone sequence alignment using squared geometric distance.

    D[t,j] = distance(t, route[j])**2 + min_{i<=j} D[t-1,i].
    Repeated visits retain distinct route positions. Forward skips are allowed
    because GPS sampling may miss short arcs. No forced first/last arc is used.
    These are proxy labels; the route does not supply physical observation labels.
    """
    n, m = len(network.gps), len(route)
    dtype = np.uint16 if m < 65536 else np.uint32
    back = np.empty((n, m), dtype=dtype)
    previous = np.zeros(m)
    indices = np.arange(m)
    starts, vectors = network.starts[route], network.vectors[route]
    for t, point in enumerate(network.gps):
        _, distance = project_segments(point, starts, vectors)
        prefix = np.minimum.accumulate(previous)
        records = np.r_[True, previous[1:] < prefix[:-1]]
        back[t] = np.maximum.accumulate(np.where(records, indices, 0))
        previous = distance**2 + prefix
        previous -= previous.min()  # preserves argmin while avoiding large accumulated cost
    positions = np.empty(n, dtype=np.int64)
    j = int(np.argmin(previous))
    for t in range(n - 1, -1, -1):
        positions[t] = j
        j = int(back[t, j])
    selected = route[positions]
    _, distance = project_segments(network.gps, network.starts[selected], network.vectors[selected])
    return ReferenceAlignment(selected, distance, positions)
