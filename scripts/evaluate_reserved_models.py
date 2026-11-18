"""Report-only: score every model family on the reserved evaluator trips.

Fits the four non-final families (and untuned boosting) on the 80 training trips with the recorded
settings, loads the frozen final model unchanged, and scores all through the same inference path
(candidate scoring -> classifier arg-max -> Viterbi -> bridged route). Nothing here selects, tunes
or alters any model, decoder setting or artifact other than the output file named below.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from pathsnap.data import trip_directories
from pathsnap.evaluation import score_prediction
from pathsnap.models import MODEL_SPECS, make_model
from pathsnap.pipeline import Pipeline
from pathsnap.prepare import training_arrays

LABELS = {"logistic": "Logistic regression", "gaussian_nb": "Gaussian Naive Bayes",
          "decision_tree": "Decision tree", "random_forest": "Random forest",
          "hist_gradient_boosting": "Hist gradient boosting"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, default=Path("data/evaluator_test"))
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument("--workdir", type=Path, required=True, help="scratch folder for temporary bundles")
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluator_model_comparison.json"))
    args = parser.parse_args()
    final_path = Path("artifacts/models/final.joblib")
    final = joblib.load(final_path)
    config = final["config"]
    x, y, _ = training_arrays(args.cache, config)
    folders = trip_directories(args.evaluation_root)
    if len(folders) != 10:
        raise RuntimeError("Expected exactly ten reserved trips")
    args.workdir.mkdir(parents=True, exist_ok=True)
    variants = [(name, name, make_model(name, config["seed"])) for name in MODEL_SPECS]
    variants[-1] = ("hist_gradient_boosting", "hist_gradient_boosting_untuned", variants[-1][2])
    results = []
    for name, key, model in variants:
        start = perf_counter()
        with threadpool_limits(limits=4):
            model.fit(x, y)
        fit_seconds = perf_counter() - start
        results.append(evaluate(name, key, model, final, config, folders, args.workdir, fit_seconds, tuned=False))
    results.append(evaluate("hist_gradient_boosting", "hist_gradient_boosting_tuned", final["model"], final, config,
                            folders, args.workdir, None, tuned=True))
    payload = {"purpose": "Report-only comparison on the reserved evaluator trips; did not select or alter any setting.",
               "label_convention": "proxy: monotone geometric alignment to reference arc sequence",
               "decoder": config["decoder"], "trips": [f.name for f in folders],
               "training_rows": int(len(y)), "models": results}
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def evaluate(name, key, model, final, config, folders, workdir, fit_seconds, tuned) -> dict:
    bundle = {**final, "model": model}
    path = workdir / f"{key}.joblib"
    joblib.dump(bundle, path)
    pipeline = Pipeline(path)
    rows = []
    for folder in folders:
        prediction = pipeline.predict(folder)
        metrics = score_prediction(prediction, folder)
        rows.append({"trip": folder.name, "timings": prediction.timings, **metrics})
        print(f"{key} {folder.name}: classifier={metrics['classifier']['proxy_top1_accuracy']:.4f} "
              f"hmm={metrics['hmm']['proxy_top1_accuracy']:.4f} F1={metrics['bridged_route']['route_length_f1']:.4f}", flush=True)
    total = sum(r["observations"] for r in rows)
    pooled = lambda part, field: sum(r[part][field] * r["observations"] for r in rows) / total
    mean = lambda part, field: float(np.mean([r[part][field] for r in rows]))
    summary = {
        "observations": total,
        "classifier_proxy_top1": pooled("classifier", "proxy_top1_accuracy"),
        "classifier_proxy_macro_f1_mean": mean("classifier", "proxy_arc_macro_f1"),
        "classifier_route_f1_mean": mean("classifier", "route_length_f1"),
        "classifier_continuity_mean": mean("classifier", "direct_arc_continuity"),
        "hmm_proxy_top1": pooled("hmm", "proxy_top1_accuracy"),
        "bridged_route_f1_mean": mean("bridged_route", "route_length_f1"),
        "bridged_route_precision_mean": mean("bridged_route", "route_length_precision"),
        "bridged_route_recall_mean": mean("bridged_route", "route_length_recall"),
        "bridged_continuity_mean": mean("bridged_route", "direct_arc_continuity"),
        "model_scoring_seconds": sum(r["timings"]["model_seconds"] for r in rows),
        "total_prediction_seconds": sum(r["timings"]["total_prediction_seconds"] for r in rows),
        "fit_seconds": fit_seconds,
    }
    return {"name": name, "key": key, "label": LABELS[name], "tuned": tuned, "family": MODEL_SPECS[name]["family"],
            "summary": summary, "trips": rows}


if __name__ == "__main__":
    main()
