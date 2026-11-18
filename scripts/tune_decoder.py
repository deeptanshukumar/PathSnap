"""Explicit development-set decoder comparison; never reads evaluator data."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import joblib
import numpy as np

from pathsnap.data import load_reference, trip_directories
from pathsnap.decoder import DecoderConfig, decode
from pathsnap.experiments import write_json
from pathsnap.pipeline import Pipeline
from pathsnap.reference import align_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    model_file = args.artifacts / "models/final.joblib"
    pipeline = Pipeline(model_file)
    configurations = [(normalize, beta) for normalize in [True, False] for beta in [5., 25., 100.]]
    trials = [{"normalize_transition_rows": norm, "beta_m": beta, "trips": []} for norm, beta in configurations]
    for folder in trip_directories(args.data_root / "student_test"):
        prediction = pipeline.predict(folder)
        route = load_reference(folder, len(prediction.network.trip.arcs))
        alignment = align_reference(prediction.network, route)
        for trial in trials:
            config = replace(pipeline.decoder_config, normalize_transition_rows=trial["normalize_transition_rows"], beta_m=trial["beta_m"])
            result = decode(prediction.network, prediction.candidates, prediction.scores, config)
            trial["trips"].append({"trip": folder.name, "observations": len(alignment.arc_ids),
                                    "correct": int(np.sum(result.arc_ids == alignment.arc_ids)),
                                    "proxy_top1_accuracy": float(np.mean(result.arc_ids == alignment.arc_ids))})
        print("Decoder search:", folder.name, flush=True)
    for trial in trials:
        trial["development_proxy_top1_accuracy"] = sum(row["correct"] for row in trial["trips"]) / sum(row["observations"] for row in trial["trips"])
    chosen = max(trials, key=lambda row: row["development_proxy_top1_accuracy"])
    write_json(args.artifacts / "decoder_search.json", {
        "selection_partition": "student_test (development only)", "objective": "pooled proxy Top-1 accuracy",
        "warning": "These same ten trips select decoder settings and report development performance; this is not an unbiased final test.",
        "trials": trials, "chosen": {key: chosen[key] for key in ["normalize_transition_rows", "beta_m", "development_proxy_top1_accuracy"]}})
    bundle = pipeline.bundle
    original = dict(bundle["config"]["decoder"])
    bundle["config"]["decoder"].update({key: chosen[key] for key in ["normalize_transition_rows", "beta_m"]})
    bundle["decoder_selection"] = {"partition": "student_test", "original_configuration": original,
                                   "objective": "pooled proxy Top-1 accuracy"}
    joblib.dump(bundle, model_file, compress=3)
    write_json(args.artifacts / "final_model.json", {key: value for key, value in bundle.items() if key != "model"})
    print(json.dumps({"chosen": chosen, "all_scores": [(x["normalize_transition_rows"], x["beta_m"], x["development_proxy_top1_accuracy"]) for x in trials]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
