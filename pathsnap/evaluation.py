"""Development scoring reads reference data strictly after prediction completes."""

from pathlib import Path

import numpy as np

from .data import load_reference, trip_directories
from .experiments import write_json
from .maps import make_map
from .metrics import point_metrics, route_metrics
from .pipeline import Pipeline, Prediction
from .reference import align_reference


def score_prediction(prediction: Prediction, folder: Path) -> dict:
    network = prediction.network
    route = load_reference(folder, len(network.trip.arcs))
    alignment = align_reference(network, route)
    classifier = {**point_metrics(prediction.classifier_arcs, alignment.arc_ids),
                  **route_metrics(prediction.classifier_arcs, route, network.trip.arcs, network.lengths)}
    decoded = {**point_metrics(prediction.decoded.arc_ids, alignment.arc_ids),
               **route_metrics(prediction.decoded.arc_ids, route, network.trip.arcs, network.lengths)}
    bridged = [arc for piece in prediction.route_pieces for arc in piece]
    return {
        "label_convention": "proxy: monotone geometric alignment to reference arc sequence",
        "observations": len(network.gps), "classifier": classifier, "hmm": decoded,
        "bridged_route": route_metrics(bridged, route, network.trip.arcs, network.lengths),
        "proxy_candidate_miss_rate": float(np.mean(~np.any(prediction.candidates.arc_ids == alignment.arc_ids[:, None], axis=1))),
        "proxy_label_distance_median_m": float(np.median(alignment.distance_m)),
        "proxy_label_distance_p95_m": float(np.quantile(alignment.distance_m, 0.95)),
        "reference_endpoint_breaks": int(np.sum(network.trip.arcs[route[:-1], 1] != network.trip.arcs[route[1:], 0])),
        "decoder_resets": int(prediction.decoded.break_before.sum() - 1),
    }


def evaluate_decoder(data_root: Path, artifacts: Path, reports: Path) -> None:
    pipeline = Pipeline(artifacts / "models/final.joblib")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "maps").mkdir(exist_ok=True)
    rows = []
    for i, folder in enumerate(trip_directories(data_root / "student_test")):
        prediction = pipeline.predict(folder)
        metrics = score_prediction(prediction, folder)
        rows.append({"trip": folder.name, "timings": prediction.timings, **metrics})
        # Preserve reproducible predictions, never invented curves or hand-edited routes.
        np.savez_compressed(artifacts / f"prediction_{folder.name}.npz", classifier=prediction.classifier_arcs,
                            hmm=prediction.decoded.arc_ids, break_before=prediction.decoded.break_before)
        if i in [0, 1, 5]:
            route = load_reference(folder, len(prediction.network.trip.arcs))
            make_map(prediction, route).save(str(reports / "maps" / f"development_{folder.name}.html"))
        print(f"Decoded {folder.name}: classifier={metrics['classifier']['proxy_top1_accuracy']:.4f} HMM={metrics['hmm']['proxy_top1_accuracy']:.4f} resets={metrics['decoder_resets']} time={prediction.timings['total_prediction_seconds']:.2f}s", flush=True)
        write_json(artifacts / "decoder_comparison.json", rows)
