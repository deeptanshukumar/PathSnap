"""Single inference path used by evaluation and the API; no reference access."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .data import load_trip
from .decoder import Decoded, DecoderConfig, decode, emission_probabilities, reconstruct_route
from .features import FEATURE_NAMES, Candidates, generate_candidates
from .geometry import RoadNetwork


@dataclass(frozen=True)
class Prediction:
    network: RoadNetwork
    candidates: Candidates
    scores: np.ndarray
    classifier_arcs: np.ndarray
    decoded: Decoded
    route_pieces: list[list[int]]
    timings: dict[str, float]


class Pipeline:
    def __init__(self, model_path: Path):
        # joblib is executable serialization: load only the team's locally produced artifact.
        self.bundle = joblib.load(model_path)
        if self.bundle["feature_names"] != FEATURE_NAMES:
            raise ValueError("Model feature schema differs from source; retrain before serving")
        self.config = self.bundle["config"]
        self.decoder_config = DecoderConfig(**self.config["decoder"])

    def predict(self, trip_folder: Path) -> Prediction:
        total = perf_counter()
        network = RoadNetwork(load_trip(trip_folder))
        start = perf_counter()
        candidates = generate_candidates(network, self.config["candidate_count"],
                                           self.config["candidate_radius_m"], self.config["index_spacing_m"])
        preparation = perf_counter() - start
        start = perf_counter()
        scores = np.zeros_like(candidates.distance_m)
        with threadpool_limits(limits=4):
            scores[candidates.valid] = self.bundle["model"].predict_proba(candidates.features[candidates.valid])[:, 1]
        model_time = perf_counter() - start
        emissions = emission_probabilities(scores, candidates.valid)
        classifier = candidates.arc_ids[np.arange(len(scores)), emissions.argmax(axis=1)]
        start = perf_counter()
        decoded = decode(network, candidates, scores, self.decoder_config)
        pieces = reconstruct_route(network, decoded, self.decoder_config)
        decoding = perf_counter() - start
        return Prediction(network, candidates, scores, classifier, decoded, pieces,
                          {"candidate_preparation_seconds": preparation, "model_seconds": model_time,
                           "decoding_and_bridging_seconds": decoding, "total_prediction_seconds": perf_counter() - total})
