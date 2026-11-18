"""Local metre geometry and directed, bounded shortest-path routing."""

from collections import OrderedDict
from dataclasses import dataclass
from math import ceil

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .data import Trip

EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class LocalProjection:
    origin: np.ndarray
    scale: np.ndarray

    @classmethod
    def for_coordinates(cls, coordinates: np.ndarray) -> "LocalProjection":
        origin = np.median(coordinates, axis=0)
        scale = np.deg2rad(1.0) * EARTH_RADIUS_M * np.array([np.cos(np.deg2rad(origin[1])), 1.0])
        return cls(origin, scale)

    def forward(self, lonlat: np.ndarray) -> np.ndarray:
        return (lonlat - self.origin) * self.scale

    def inverse(self, xy: np.ndarray) -> np.ndarray:
        return xy / self.scale + self.origin


def project_segments(points: np.ndarray, starts: np.ndarray, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Broadcastable closest point on finite segments; return fraction and metres."""
    delta = points - starts
    denominator = np.sum(vectors * vectors, axis=-1)
    fraction = np.divide(np.sum(delta * vectors, axis=-1), denominator,
                         out=np.zeros_like(denominator, dtype=float), where=denominator > 0)
    # Geometric projection is constrained to the finite segment, not an infinite road line.
    fraction = np.clip(fraction, 0.0, 1.0)
    distance = np.linalg.norm(delta - fraction[..., None] * vectors, axis=-1)
    return fraction, distance


class RoadNetwork:
    def __init__(self, trip: Trip):
        self.trip = trip
        self.projection = LocalProjection.for_coordinates(trip.nodes)
        self.xy = self.projection.forward(trip.nodes)
        self.gps = self.projection.forward(trip.track[:, :2])
        self.starts = self.xy[trip.arcs[:, 0]]
        self.vectors = self.xy[trip.arcs[:, 1]] - self.starts
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        # Duplicate source/target pairs must take their minimum weight, never CSR's sum.
        order = np.lexsort((self.lengths, trip.arcs[:, 1], trip.arcs[:, 0]))
        pairs = trip.arcs[order]
        keep = np.r_[True, np.any(pairs[1:] != pairs[:-1], axis=1)]
        self.graph_arc_ids = order[keep]
        edges = trip.arcs[self.graph_arc_ids]
        self.graph = csr_matrix((self.lengths[self.graph_arc_ids], (edges[:, 0], edges[:, 1])),
                                shape=(len(trip.nodes), len(trip.nodes)))
        self.edge_lookup = {tuple(pair): int(arc) for pair, arc in zip(edges, self.graph_arc_ids)}
        self.outdegree = np.bincount(trip.arcs[:, 0], minlength=len(trip.nodes))
        self.indegree = np.bincount(trip.arcs[:, 1], minlength=len(trip.nodes))
        self._distance_cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()

    def distances_from(self, node: int, limit_m: float) -> np.ndarray:
        bucket = max(250, int(ceil(limit_m / 250) * 250))
        key = (int(node), bucket)
        if key not in self._distance_cache:
            self._distance_cache[key] = dijkstra(self.graph, directed=True, indices=int(node), limit=bucket)
            if len(self._distance_cache) > 96:
                self._distance_cache.popitem(last=False)
        self._distance_cache.move_to_end(key)
        return self._distance_cache[key]

    def transition_distances(self, left: np.ndarray, left_fraction: np.ndarray,
                             right: np.ndarray, right_fraction: np.ndarray, limit_m: float) -> np.ndarray:
        """Distance between projected positions on directed edges, including partial edges."""
        result = np.full((len(left), len(right)), np.inf)
        for i, arc in enumerate(left):
            if arc < 0:
                continue
            valid = right >= 0
            targets = right[valid]
            distance = self.distances_from(int(self.trip.arcs[arc, 1]), limit_m)
            via_nodes = ((1.0 - left_fraction[i]) * self.lengths[arc]
                         + distance[self.trip.arcs[targets, 0]]
                         + right_fraction[valid] * self.lengths[targets])
            result[i, valid] = via_nodes
            same = np.flatnonzero(right == arc)
            for j in same:
                forward = (right_fraction[j] - left_fraction[i]) * self.lengths[arc]
                # Up to 3 m backwards projection is treated as stationary GPS jitter.
                if forward >= -3.0:
                    result[i, j] = min(result[i, j], max(0.0, forward))
        result[result > limit_m] = np.inf
        return result

    def bridge(self, left: int, right: int, limit_m: float) -> list[int] | None:
        """Intermediate arcs only. None marks an actual break, never a straight connector."""
        if left == right:
            return []
        source, target = int(self.trip.arcs[left, 1]), int(self.trip.arcs[right, 0])
        if source == target:
            return []
        dist, predecessor = dijkstra(self.graph, directed=True, indices=source,
                                      limit=limit_m, return_predecessors=True)
        if not np.isfinite(dist[target]):
            return None
        nodes = [target]
        while nodes[-1] != source:
            parent = int(predecessor[nodes[-1]])
            if parent < 0:
                return None
            nodes.append(parent)
        nodes.reverse()
        return [self.edge_lookup[(u, v)] for u, v in zip(nodes[:-1], nodes[1:])]
