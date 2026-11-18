"""Read-only explanation of one finished prediction for the UI.

Everything here is derived from the Prediction object after inference completes. It never
reads reference routes and never changes candidates, scores or decoded arcs.
"""

from collections import Counter

import numpy as np

from .decoder import emission_probabilities
from .features import FEATURE_NAMES
from .pipeline import Prediction


def _latlon(lonlat: np.ndarray) -> list[list[float]]:
    return np.round(lonlat[:, ::-1], 6).tolist()


def _arc_chains(arc_ids: list[int], arcs: np.ndarray) -> list[list[int]]:
    """Split an arc list into node chains; a non-touching pair starts a new chain."""
    chains: list[list[int]] = []
    for arc in arc_ids:
        source, target = (int(v) for v in arcs[arc])
        if chains and chains[-1][-1] == source:
            chains[-1].append(target)
        else:
            chains.append([source, target])
    return chains


def build_trace(prediction: Prediction, config: dict, model_name: str, tuned: bool) -> dict:
    network = prediction.network
    trip = network.trip
    track = trip.track
    candidates = prediction.candidates
    valid = candidates.valid
    scores = prediction.scores
    emissions = emission_probabilities(scores, valid)
    decoded = prediction.decoded
    n = len(track)
    times = track[:, 2]
    gaps = np.diff(times)

    route_arcs = [arc for piece in prediction.route_pieces for arc in piece]
    route_lines = []
    for piece in prediction.route_pieces:
        for chain in _arc_chains(piece, trip.arcs):
            route_lines.append(_latlon(trip.nodes[chain]))
    classifier_unique = np.unique(prediction.classifier_arcs[prediction.classifier_arcs >= 0])
    classifier_segments = [_latlon(trip.nodes[trip.arcs[arc]]) for arc in classifier_unique.tolist()]

    classifier_index = emissions.argmax(axis=1)
    viterbi_index = decoded.candidate_indices
    changed = int(np.sum(decoded.arc_ids != prediction.classifier_arcs))
    nearest = np.where(valid, candidates.distance_m, np.inf).min(axis=1)
    top_probability = np.where(valid, scores, -1).max(axis=1)
    resets = [{"i": int(i), "reason": reason, "latlon": [round(float(track[i, 1]), 6), round(float(track[i, 0]), 6)]}
              for i, reason in sorted(decoded.break_reasons.items())]
    reason_counts = Counter("GPS gap" if reason.startswith("GPS gap") else "No directed path" for reason in decoded.break_reasons.values())

    # Local-metre geometry for the per-point explainer plot.
    used = np.unique(candidates.arc_ids[valid])
    segments = {int(arc): np.round(np.r_[network.starts[arc], network.starts[arc] + network.vectors[arc]], 1).tolist()
                for arc in used.tolist()}
    pad = lambda values, digits: np.round(np.where(valid, values, 0), digits).tolist()
    span = network.gps.max(axis=0) - network.gps.min(axis=0)
    centre = network.projection.origin

    return {
        "trip_id": trip.trip_id,
        "model": model_name, "tuned": tuned,
        "map": {
            "gps": _latlon(track[:, :2]),
            "route": route_lines,
            "classifier": classifier_segments,
            "resets": resets,
            "start": [round(float(track[0, 1]), 6), round(float(track[0, 0]), 6)],
            "finish": [round(float(track[-1, 1]), 6), round(float(track[-1, 0]), 6)],
        },
        "stages": {
            "input": {"observations": n, "duration_s": float(times[-1] - times[0]),
                      "median_step_s": float(np.median(gaps)) if len(gaps) else 0.0,
                      "gaps_over_60s": int(np.sum(gaps > 60)),
                      "road_nodes": int(len(trip.nodes)), "road_arcs": int(len(trip.arcs))},
            "projection": {"method": "local equirectangular, metres",
                           "origin_lonlat": np.round(centre, 6).tolist(),
                           "extent_km": np.round(span / 1000, 2).tolist()},
            "candidates": {"radius_m": config["candidate_radius_m"], "max_per_point": config["candidate_count"],
                           "pairs": int(valid.sum()), "mean_per_point": float(valid.sum(axis=1).mean()),
                           "fallback_points": int(candidates.fallback.sum()),
                           "nearest_median_m": float(np.median(nearest)),
                           "nearest_p95_m": float(np.quantile(nearest, 0.95))},
            "classifier": {"features": FEATURE_NAMES, "pairs_scored": int(valid.sum()),
                           "mean_top_probability": float(top_probability.mean()),
                           "points_top_above_half": int(np.sum(top_probability >= 0.5))},
            "viterbi": {"beta_m": config["decoder"]["beta_m"], "gap_reset_s": config["decoder"]["gap_reset_s"],
                        "max_speed_mps": config["decoder"]["max_speed_mps"],
                        "changed_from_classifier": changed, "resets": len(resets),
                        "reset_reasons": dict(reason_counts)},
            "route": {"pieces": len(prediction.route_pieces), "arcs": len(route_arcs),
                      "length_km": round(float(network.lengths[route_arcs].sum()) / 1000, 3) if route_arcs else 0.0},
            "timings": prediction.timings,
        },
        "points": {
            "t": np.round(times - times[0], 1).tolist(),
            "xy": np.round(network.gps, 1).tolist(),
            "arc": np.where(valid, candidates.arc_ids, -1).tolist(),
            "distance_m": pad(candidates.distance_m, 1),
            "heading_cos": pad(candidates.features[:, :, FEATURE_NAMES.index("heading_cos")], 3),
            "probability": pad(scores, 4),
            "emission": pad(emissions, 4),
            "classifier_pick": classifier_index.tolist(),
            "viterbi_pick": viterbi_index.tolist(),
            "reset_before": decoded.break_before.astype(int).tolist(),
        },
        "segments_xy": segments,
    }


def explain_point(prediction: Prediction, config: dict, i: int) -> dict:
    """Recompute the decoder's transition terms into observation i from its Viterbi predecessor.

    Uses the same network distance call and formula as decoder.decode; read-only.
    """
    network = prediction.network
    candidates = prediction.candidates
    n = len(network.gps)
    if not 0 <= i < n:
        raise ValueError(f"Observation must be between 0 and {n - 1}")
    decoder = config["decoder"]
    emissions = emission_probabilities(prediction.scores, candidates.valid)
    valid = candidates.valid[i]
    rows = [{"candidate": int(k), "arc": int(candidates.arc_ids[i, k]),
             "distance_m": round(float(candidates.distance_m[i, k]), 2),
             "features": {name: round(float(v), 4) for name, v in zip(FEATURE_NAMES, candidates.features[i, k])},
             "probability": round(float(prediction.scores[i, k]), 5),
             "emission": round(float(emissions[i, k]), 5)} for k in np.flatnonzero(valid)]
    result = {"i": i, "t": float(network.trip.track[i, 2] - network.trip.track[0, 2]),
              "classifier_pick": int(emissions[i].argmax()), "viterbi_pick": int(prediction.decoded.candidate_indices[i]),
              "candidates": rows, "transition": None}
    if i == 0 or prediction.decoded.break_before[i]:
        result["transition"] = {"reset": True, "reason": prediction.decoded.break_reasons.get(i, "Start of trip")}
        return result
    dt = float(network.trip.track[i, 2] - network.trip.track[i - 1, 2])
    previous = int(prediction.decoded.candidate_indices[i - 1])
    limit = decoder["max_speed_mps"] * dt + decoder["routing_slack_m"]
    distances = network.transition_distances(candidates.arc_ids[i - 1, previous:previous + 1],
                                             candidates.fraction[i - 1, previous:previous + 1],
                                             candidates.arc_ids[i], candidates.fraction[i], limit)[0]
    displacement = float(np.linalg.norm(network.gps[i] - network.gps[i - 1]))
    for row in rows:
        d = float(distances[row["candidate"]])
        row["route_distance_m"] = round(d, 2) if np.isfinite(d) else None
        row["log_transition"] = round(-abs(d - displacement) / decoder["beta_m"], 4) if np.isfinite(d) else None
        row["log_emission"] = round(float(np.log(max(row["emission"], 1e-12))), 4)
    result["transition"] = {"reset": False, "dt_s": dt, "gps_displacement_m": round(displacement, 2),
                            "previous_arc": int(candidates.arc_ids[i - 1, previous]), "beta_m": decoder["beta_m"],
                            "limit_m": round(limit, 1)}
    return result
