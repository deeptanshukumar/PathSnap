"""Directed network transition potentials and segmented log-space Viterbi."""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .features import Candidates
from .geometry import RoadNetwork


@dataclass(frozen=True)
class DecoderConfig:
    beta_m: float = 25.0
    max_speed_mps: float = 55.0
    gap_reset_s: float = 60.0
    routing_slack_m: float = 100.0
    normalize_transition_rows: bool = True

    def __post_init__(self):
        if min(self.beta_m, self.max_speed_mps, self.gap_reset_s, self.routing_slack_m) <= 0:
            raise ValueError("Decoder parameters must be positive")


@dataclass(frozen=True)
class Decoded:
    arc_ids: np.ndarray
    candidate_indices: np.ndarray
    break_before: np.ndarray
    break_reasons: dict[int, str]


def emission_probabilities(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Positive binary scores normalised over candidates; invalid padding has zero mass."""
    if scores.shape != valid.shape or np.any(valid.sum(axis=1) == 0):
        raise ValueError("Every observation needs at least one valid candidate")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Candidate scores must be finite probabilities")
    mass = np.where(valid, np.maximum(scores, 1e-12), 0)
    return mass / mass.sum(axis=1, keepdims=True)


def viterbi(emissions: np.ndarray, transitions: list[np.ndarray | None]) -> tuple[np.ndarray, np.ndarray]:
    """Generic recurrence, with explicit resets when no legal path survives.

    None indicates a known gap. A completely unreachable step also starts a new
    independent segment. Backtracking never links across either kind of reset.
    """
    n, k = emissions.shape
    if len(transitions) != n - 1:
        raise ValueError("Expected one transition matrix per consecutive observation pair")
    with np.errstate(divide="ignore"):
        log_emission = np.log(emissions)
    back = np.zeros((n, k), dtype=np.int64)
    resets = np.zeros(n, dtype=bool)
    resets[0] = True
    path = np.zeros(n, dtype=np.int64)
    delta = log_emission[0].copy()
    segment_start = 0

    def finish(end: int, scores: np.ndarray, start: int) -> None:
        path[end] = int(np.argmax(scores))
        for t in range(end, start, -1):
            path[t - 1] = back[t, path[t]]

    for t, transition in enumerate(transitions, start=1):
        if transition is None:
            next_delta = np.full(k, -np.inf)
        else:
            options = delta[:, None] + transition
            back[t] = np.argmax(options, axis=0)
            next_delta = log_emission[t] + np.max(options, axis=0)
        if not np.isfinite(next_delta).any():
            finish(t - 1, delta, segment_start)
            segment_start = t
            resets[t] = True
            delta = log_emission[t].copy()
        else:
            delta = next_delta - np.max(next_delta)
    finish(n - 1, delta, segment_start)
    return path, resets


def decode(network: RoadNetwork, candidates: Candidates, scores: np.ndarray,
           config: DecoderConfig) -> Decoded:
    emissions = emission_probabilities(scores, candidates.valid)
    transitions: list[np.ndarray | None] = []
    reasons = {}
    times = network.trip.track[:, 2]
    for t in range(1, len(times)):
        dt = times[t] - times[t - 1]
        if dt > config.gap_reset_s:
            transitions.append(None)
            reasons[t] = f"GPS gap {dt:g} s exceeds {config.gap_reset_s:g} s"
            continue
        limit = config.max_speed_mps * dt + config.routing_slack_m
        distances = network.transition_distances(candidates.arc_ids[t - 1], candidates.fraction[t - 1],
                                                  candidates.arc_ids[t], candidates.fraction[t], limit)
        gps_displacement = np.linalg.norm(network.gps[t] - network.gps[t - 1])
        potential = -np.abs(distances - gps_displacement) / config.beta_m
        if config.normalize_transition_rows:
            normalizer = logsumexp(potential, axis=1, keepdims=True)
            log_transition = np.full_like(potential, -np.inf)
            np.subtract(potential, normalizer, out=log_transition, where=np.isfinite(normalizer))
        else:
            # Newson/Krumm-style exponential compatibility has a global 1/beta
            # normaliser, constant for every path of the same length. Row-normalising
            # a truncated candidate lattice can introduce a state-dependent bias.
            log_transition = potential
        transitions.append(log_transition)
    path, breaks = viterbi(emissions, transitions)
    for t in np.flatnonzero(breaks)[1:]:
        reasons.setdefault(int(t), "No directed path survives within the speed and routing bound")
    return Decoded(candidates.arc_ids[np.arange(len(path)), path], path, breaks, reasons)


def reconstruct_route(network: RoadNetwork, decoded: Decoded, config: DecoderConfig) -> list[list[int]]:
    """Insert real shortest-path bridge arcs and retain separate route pieces at resets."""
    pieces = [[int(decoded.arc_ids[0])]]
    for i in range(1, len(decoded.arc_ids)):
        left, right = int(decoded.arc_ids[i - 1]), int(decoded.arc_ids[i])
        if decoded.break_before[i]:
            pieces.append([right])
        else:
            # Recompute the exact segment projections used for the states.
            # Equal arc IDs can still require a real network loop between them.
            left_point = network.gps[i - 1]
            right_point = network.gps[i]
            vector = network.vectors[left]
            denominator = float(np.dot(vector, vector))
            if left == right and denominator > 0:
                left_fraction = float(np.clip(np.dot(left_point - network.starts[left], vector) / denominator, 0, 1))
                right_fraction = float(np.clip(np.dot(right_point - network.starts[left], vector) / denominator, 0, 1))
                same_arc_loop = (right_fraction - left_fraction) * network.lengths[left] < -3.0
            else:
                same_arc_loop = False
            if left == right and not same_arc_loop:
                continue
            dt = network.trip.track[i, 2] - network.trip.track[i - 1, 2]
            bridge = network.bridge(left, right, config.max_speed_mps * dt + config.routing_slack_m,
                                    same_arc_loop=same_arc_loop)
            if bridge is None:
                pieces.append([right])
            else:
                pieces[-1].extend(bridge + [right])
    return pieces
