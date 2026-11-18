"""Opt-in improved variant: trajectory-context features and a re-weighted decoder.

The default (reported, frozen) predictor is untouched. This module adds features that
describe how well each candidate arc fits the neighbouring GPS points, not only the
current one. Inputs are still only .track, .nodes and .arcs.
"""

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .data import load_trip
from .decoder import DecoderConfig, decode, emission_probabilities, reconstruct_route
from .features import FEATURE_NAMES, Candidates, generate_candidates
from .geometry import RoadNetwork, project_segments
from .pipeline import Prediction

OFFSETS = (1, 2, 3, 5, 8)
CONTEXT_NAMES = (
    [f"ctx_mean_distance_pm{o}_m" for o in OFFSETS]
    + [f"ctx_max_distance_pm{o}_m" for o in OFFSETS]
    + ["ctx_mean_distance_rank", "ctx_mean_distance_above_best_m", "ctx_heading_cos_pm5",
       "distance_rank", "valid_candidates", "second_gap_m", "along_arc_progress_error_m"]
)
IMPROVED_FEATURE_NAMES = FEATURE_NAMES + CONTEXT_NAMES


def _segment_limits(times: np.ndarray, gap_s: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """First and last index of the gap-free run containing each observation."""
    n = len(times)
    breaks = np.r_[0, np.flatnonzero(np.diff(times) > gap_s) + 1, n]
    first, last = np.empty(n, dtype=int), np.empty(n, dtype=int)
    for a, b in zip(breaks[:-1], breaks[1:]):
        first[a:b], last[a:b] = a, b - 1
    return first, last


def context_features(network: RoadNetwork, candidates: Candidates) -> np.ndarray:
    """Per (observation, candidate) features from neighbouring observations, never crossing outages."""
    arcs = candidates.arc_ids
    valid = arcs >= 0
    safe = np.maximum(arcs, 0)
    n, k = arcs.shape
    starts, vectors = network.starts[safe], network.vectors[safe]
    first, last = _segment_limits(network.trip.track[:, 2])
    ids = np.arange(n)
    own = np.where(valid, candidates.distance_m, 0.0)

    def distances_at(offset: int) -> np.ndarray:
        j = np.clip(ids + offset, first, last)
        _, d = project_segments(network.gps[j][:, None, :], starts, vectors)
        return d

    means, maxima = [], []
    total, count = own.copy(), np.ones((n, 1))
    worst = own.copy()
    for offset in range(1, max(OFFSETS) + 1):
        for sign in (-1, 1):
            d = distances_at(sign * offset)
            total += d
            worst = np.maximum(worst, d)
        count += 2
        if offset in OFFSETS:
            means.append(total / count)
            maxima.append(worst.copy())
    window = means[OFFSETS.index(3)]
    big = np.where(valid, window, np.inf)
    rank_ctx = np.argsort(np.argsort(big, axis=1), axis=1).astype(float)
    above_best = window - big.min(axis=1, keepdims=True)

    # Heading of the arc against displacement over +/-5 observations inside one run.
    lo, hi = np.clip(ids - 5, first, last), np.clip(ids + 5, first, last)
    delta = network.gps[hi] - network.gps[lo]
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    heading = np.divide(delta, norm, out=np.zeros_like(delta), where=norm > 0)
    lengths = network.lengths[safe]
    unit = np.divide(vectors, lengths[..., None], out=np.zeros_like(vectors), where=lengths[..., None] > 0)
    heading_cos = np.sum(unit * heading[:, None, :], axis=-1)

    dist = np.where(valid, candidates.distance_m, np.inf)
    rank = np.argsort(np.argsort(dist, axis=1), axis=1).astype(float)
    n_valid = np.broadcast_to(valid.sum(axis=1, keepdims=True), (n, k)).astype(float)
    ordered = np.sort(dist, axis=1)
    second = np.where(np.isfinite(ordered[:, 1:2]), ordered[:, 1:2] - ordered[:, :1], 100.0)
    second = np.broadcast_to(second, (n, k))

    # Movement along the arc between the previous and next fix, against GPS travel.
    prev_j, next_j = np.clip(ids - 1, first, last), np.clip(ids + 1, first, last)
    f_prev, _ = project_segments(network.gps[prev_j][:, None, :], starts, vectors)
    f_next, _ = project_segments(network.gps[next_j][:, None, :], starts, vectors)
    along = (f_next - f_prev) * lengths
    travel = np.linalg.norm(network.gps[next_j] - network.gps[prev_j], axis=1)[:, None]
    progress_error = np.abs(along - travel)

    stacked = np.stack(means + maxima + [rank_ctx, above_best, heading_cos, rank, n_valid, second,
                                          progress_error], axis=-1)
    stacked[~valid] = 0
    stacked = np.where(np.isfinite(stacked), stacked, 0).astype(np.float32)
    return np.clip(stacked, -1e4, 1e4)


def improved_features(network: RoadNetwork, candidates: Candidates) -> np.ndarray:
    return np.concatenate([candidates.features, context_features(network, candidates)], axis=-1)


class ImprovedPipeline:
    """Same inputs, candidates and decoder as the default; extra features and an emission weight."""

    def __init__(self, model_path: Path):
        # joblib is executable serialization: load only the team's locally produced artifact.
        self.bundle = joblib.load(model_path)
        if self.bundle["feature_names"] != IMPROVED_FEATURE_NAMES:
            raise ValueError("Improved model feature schema differs from source; retrain the variant")
        self.config = self.bundle["config"]
        self.decoder_config = DecoderConfig(**self.config["decoder"])
        self.emission_power = float(self.bundle["emission_power"])

    def predict(self, trip_folder: Path) -> Prediction:
        total = perf_counter()
        network = RoadNetwork(load_trip(trip_folder))
        start = perf_counter()
        candidates = generate_candidates(network, self.config["candidate_count"],
                                           self.config["candidate_radius_m"], self.config["index_spacing_m"])
        features = improved_features(network, candidates)
        preparation = perf_counter() - start
        start = perf_counter()
        scores = np.zeros_like(candidates.distance_m)
        with threadpool_limits(limits=4):
            scores[candidates.valid] = self.bundle["model"].predict_proba(features[candidates.valid])[:, 1]
        model_time = perf_counter() - start
        emissions = emission_probabilities(scores, candidates.valid)
        classifier = candidates.arc_ids[np.arange(len(scores)), emissions.argmax(axis=1)]
        start = perf_counter()
        decoded = decode(network, candidates, weighted_scores(scores, self.emission_power), self.decoder_config)
        pieces = reconstruct_route(network, decoded, self.decoder_config)
        decoding = perf_counter() - start
        return Prediction(network, candidates, scores, classifier, decoded, pieces,
                          {"candidate_preparation_seconds": preparation, "model_seconds": model_time,
                           "decoding_and_bridging_seconds": decoding, "total_prediction_seconds": perf_counter() - total})


def weighted_scores(scores: np.ndarray, power: float) -> np.ndarray:
    """Raising scores to a power p weights log-emissions by p against transitions in Viterbi."""
    return np.power(np.clip(scores, 1e-12, 1.0), power)


def decoder_with(config: DecoderConfig, **changes) -> DecoderConfig:
    return replace(config, **changes)
