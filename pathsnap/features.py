"""Route-independent candidate generation and observation/candidate features."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .geometry import RoadNetwork, project_segments

FEATURE_NAMES = ["distance_m", "log_distance", "distance_above_nearest_m", "heading_cos",
                 "abs_heading_cos", "length_m", "log_length", "fraction", "speed_mps",
                 "log_previous_gap_s", "source_indegree", "target_outdegree", "stationary"]


@dataclass(frozen=True)
class Candidates:
    arc_ids: np.ndarray  # [observation, candidate], -1 is padding
    fraction: np.ndarray
    distance_m: np.ndarray
    features: np.ndarray  # [observation, candidate, feature]
    fallback: np.ndarray  # no edge within configured radius

    @property
    def valid(self) -> np.ndarray:
        return self.arc_ids >= 0


def motion(network: RoadNetwork) -> tuple[np.ndarray, np.ndarray]:
    """Offline local heading from a +/-3-observation window; zero heading at stops."""
    n = len(network.gps)
    ids = np.arange(n)
    lo, hi = np.maximum(0, ids - 3), np.minimum(n - 1, ids + 3)
    times = network.trip.track[:, 2]
    # Do not estimate motion across a multi-minute outage.
    gaps = np.diff(times)
    for i in range(n):
        if np.any(gaps[lo[i]:hi[i]] > 60):
            lo[i] = max(0, i - 1) if i > 0 and gaps[i - 1] <= 60 else i
            hi[i] = min(n - 1, i + 1) if i < n - 1 and gaps[i] <= 60 else i
    delta = network.gps[hi] - network.gps[lo]
    elapsed = times[hi] - times[lo]
    lengths = np.linalg.norm(delta, axis=1)
    unit = np.divide(delta, lengths[:, None], out=np.zeros_like(delta), where=lengths[:, None] > 0)
    speed = np.divide(lengths, elapsed, out=np.zeros(n), where=elapsed > 0)
    return unit, speed


def generate_candidates(network: RoadNetwork, count: int = 8, radius_m: float = 100,
                        spacing_m: float = 30) -> Candidates:
    if count < 1 or radius_m <= 0 or spacing_m <= 0:
        raise ValueError("Candidate count, radius and index spacing must be positive")
    lengths = network.lengths
    intervals = np.maximum(np.ceil(lengths / spacing_m).astype(int), 1)
    ids = np.repeat(np.arange(len(lengths)), intervals + 1)
    beginnings = np.repeat(np.cumsum(intervals + 1) - (intervals + 1), intervals + 1)
    fractions = (np.arange(len(ids)) - beginnings) / intervals[ids]
    samples = network.starts[ids] + fractions[:, None] * network.vectors[ids]
    tree = cKDTree(samples)
    # Every segment intersecting the radius has a sample within radius + spacing/2.
    balls = tree.query_ball_point(network.gps, radius_m + spacing_m / 2, workers=1)
    shape = (len(network.gps), count)
    arcs = np.full(shape, -1, dtype=np.int64)
    fraction = np.zeros(shape)
    distance = np.full(shape, np.inf)
    fallback = np.zeros(len(arcs), dtype=bool)
    for i, ball in enumerate(balls):
        edges = np.unique(ids[ball])
        if len(edges):
            f, d = project_segments(network.gps[i], network.starts[edges], network.vectors[edges])
            within = d <= radius_m
            edges, f, d = edges[within], f[within], d[within]
        if not len(edges):
            fallback[i] = True
            # Fallback is explicit and may be far away; never force in the reference edge.
            near = tree.query(network.gps[i], k=min(128, len(samples)))[1]
            edges = np.unique(ids[np.atleast_1d(near)])
            f, d = project_segments(network.gps[i], network.starts[edges], network.vectors[edges])
        order = np.lexsort((edges, d))[:count]
        k = len(order)
        arcs[i, :k], fraction[i, :k], distance[i, :k] = edges[order], f[order], d[order]
    safe = np.maximum(arcs, 0)
    heading, speed = motion(network)
    unit = np.divide(network.vectors[safe], lengths[safe, None],
                     out=np.zeros_like(network.vectors[safe]), where=lengths[safe, None] > 0)
    cosine = np.sum(unit * heading[:, None, :], axis=-1)
    finite_distance = np.where(arcs >= 0, distance, 0)
    gap = np.r_[1, np.diff(network.trip.track[:, 2])]
    broadcast = lambda values: np.broadcast_to(values[:, None], shape)
    features = np.stack([
        finite_distance, np.log1p(finite_distance), finite_distance - distance[:, :1],
        cosine, np.abs(cosine), lengths[safe], np.log1p(lengths[safe]), fraction,
        broadcast(speed), broadcast(np.log1p(gap)),
        network.indegree[network.trip.arcs[safe, 0]], network.outdegree[network.trip.arcs[safe, 1]],
        broadcast((speed < 0.5).astype(float)),
    ], axis=-1).astype(np.float32)
    features[arcs < 0] = 0
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite candidate feature values")
    return Candidates(arcs, fraction, distance, features, fallback)
