"""Measured comparison and Bayesian tuning with whole-trip validation folds."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import optuna
import sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from .data import load_reference, load_trip, trip_directories
from .decoder import emission_probabilities
from .features import FEATURE_NAMES
from .geometry import RoadNetwork
from .metrics import candidate_metrics, point_metrics, route_metrics
from .models import MODEL_SPECS, make_model, search_parameters
from .prepare import training_arrays


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def grouped_folds(x: np.ndarray, y: np.ndarray, groups: np.ndarray, config: dict):
    folds = list(StratifiedGroupKFold(n_splits=config["cv_folds"], shuffle=True,
                                     random_state=config["seed"]).split(x, y, groups))
    for train, validation in folds:
        if set(groups[train]) & set(groups[validation]):
            raise RuntimeError("Trip leakage detected")
    return folds


def cross_validate(name: str, parameters: dict, x: np.ndarray, y: np.ndarray,
                   folds: list, config: dict) -> tuple[dict, np.ndarray]:
    oof = np.zeros(len(y))
    rows = []
    for fold_id, (train, validation) in enumerate(folds):
        model = make_model(name, config["seed"], parameters)
        start = perf_counter()
        with threadpool_limits(limits=4):
            model.fit(x[train], y[train])
            elapsed = perf_counter() - start
            probabilities = model.predict_proba(x[validation])[:, 1]
        oof[validation] = probabilities
        rows.append({"fold": fold_id, "training_seconds": elapsed,
                     **candidate_metrics(y[validation], probabilities)})
    return {"cv_candidate_macro_f1": float(np.mean([row["candidate_macro_f1"] for row in rows])),
            "cv_candidate_macro_f1_std": float(np.std([row["candidate_macro_f1"] for row in rows])),
            "cv_folds": rows, "oof": candidate_metrics(y, oof)}, oof


def evaluate_model(model, data_root: Path, cache_root: Path) -> dict:
    """Full 10-trip development evaluation; model inference timing excludes preprocessing."""
    rows, all_labels, all_scores = [], [], []
    for folder in trip_directories(data_root / "student_test"):
        network = RoadNetwork(load_trip(folder))
        route = load_reference(folder, len(network.trip.arcs))
        with np.load(cache_root / "student_test" / f"{folder.name}.npz") as data:
            arcs, features = data["arcs"], data["features"]
            valid = arcs >= 0
            start = perf_counter()
            with threadpool_limits(limits=4):
                scores = model.predict_proba(features[valid])[:, 1]
            elapsed = perf_counter() - start
            all_labels.append(data["labels"][valid])
            all_scores.append(scores)
            matrix = np.zeros_like(arcs, dtype=float)
            matrix[valid] = scores
            emissions = emission_probabilities(matrix, valid)
            predicted = arcs[np.arange(len(arcs)), np.argmax(emissions, axis=1)]
            reference_arcs = data["reference_arcs"]
            hits = arcs == reference_arcs[:, None]
            reference_mass = np.sum(emissions * hits, axis=1)
            row = {"trip": folder.name, "observations": len(arcs),
                   "model_inference_seconds": elapsed,
                   "proxy_candidate_miss_rate": float(np.mean(~hits.any(axis=1))),
                   "observation_proxy_log_loss": float(-np.log(np.maximum(reference_mass, 1e-12)).mean()),
                   "proxy_label_distance_p95_m": float(np.quantile(data["reference_distance"], 0.95)),
                   **point_metrics(predicted, reference_arcs),
                   **route_metrics(predicted, route, network.trip.arcs, network.lengths)}
            rows.append(row)
    summary = candidate_metrics(np.concatenate(all_labels), np.concatenate(all_scores))
    total = sum(row["observations"] for row in rows)
    for key in ["proxy_top1_accuracy", "proxy_candidate_miss_rate", "observation_proxy_log_loss"]:
        summary[key] = sum(row[key] * row["observations"] for row in rows) / total
    for key in ["proxy_arc_macro_f1", "route_length_f1", "ordered_route_lcs_f1", "direct_arc_continuity"]:
        summary[f"mean_trip_{key}"] = float(np.mean([row[key] for row in rows]))
    summary["model_inference_seconds"] = sum(row["model_inference_seconds"] for row in rows)
    summary["observations"] = total
    return {"summary": summary, "trips": rows}


def run_experiments(data_root: Path, cache_root: Path, output_root: Path, config: dict) -> None:
    x, y, groups = training_arrays(cache_root, config)
    folds = grouped_folds(x, y, groups, config)
    models_folder = output_root / "models"
    models_folder.mkdir(parents=True, exist_ok=True)
    fold_record = [{"fold": i, "train_trips": sorted(set(groups[a])), "validation_trips": sorted(set(groups[b]))}
                   for i, (a, b) in enumerate(folds)]
    write_json(output_root / "folds.json", fold_record)
    write_json(output_root / "training_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "candidate_training_rows": len(y), "positive_rows": int(y.sum()),
        "training_trips_with_usable_rows": len(set(groups)), "feature_names": FEATURE_NAMES,
        "python_version": __import__("platform").python_version(), "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__, "optuna_version": optuna.__version__,
        "selection_objective": "mean training-trip grouped-CV binary candidate Macro F1",
        "pointwise_label_warning": "Monotone geometric reference alignment creates proxy labels, not official per-point ground truth.",
    })
    rows = []
    for name, spec in MODEL_SPECS.items():
        print(f"Training {name}", flush=True)
        cv, oof = cross_validate(name, {}, x, y, folds, config)
        model = make_model(name, config["seed"])
        start = perf_counter()
        with threadpool_limits(limits=4):
            model.fit(x, y)
            elapsed = perf_counter() - start
            fit_scores = model.predict_proba(x)[:, 1]
        joblib.dump(model, models_folder / f"{name}.joblib", compress=3)
        result = {"name": name, **spec, **cv, "full_fit_seconds": elapsed,
                  "training": candidate_metrics(y, fit_scores),
                  "student_test": evaluate_model(model, data_root, cache_root)}
        rows.append(result)
        write_json(output_root / "model_comparison.json", rows)
        np.savez_compressed(output_root / f"{name}_oof.npz", probability=oof, labels=y)
        print(f"{name}: CV Macro F1={cv['cv_candidate_macro_f1']:.4f}; dev Top-1={result['student_test']['summary']['proxy_top1_accuracy']:.4f}", flush=True)
    chosen = max(rows, key=lambda row: row["cv_candidate_macro_f1"])
    name = chosen["name"]
    write_json(output_root / "selection.json", {"chosen_family": name, "rule": "maximum mean grouped CV candidate Macro F1", "student_test_used_in_selection": False})
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config["seed"], n_startup_trials=4))

    def objective(trial):
        params = search_parameters(trial, name)
        result, _ = cross_validate(name, params, x, y, folds, config)
        trial.set_user_attr("fold_macro_f1", [row["candidate_macro_f1"] for row in result["cv_folds"]])
        print(f"TPE trial {trial.number}: {result['cv_candidate_macro_f1']:.4f}", flush=True)
        return result["cv_candidate_macro_f1"]

    start = perf_counter()
    study.optimize(objective, n_trials=config["bayesian_trials"])
    tuning_time = perf_counter() - start
    trials = [{"number": t.number, "value": t.value, "state": t.state.name, "parameters": t.params,
               "distributions": {k: optuna.distributions.distribution_to_json(v) for k, v in t.distributions.items()},
               "fold_macro_f1": t.user_attrs.get("fold_macro_f1"), "duration_seconds": t.duration.total_seconds()}
              for t in study.trials]
    write_json(output_root / "bayesian_trials.json", trials)
    with (output_root / "bayesian_trials.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["number", "value", "parameters", "duration_seconds"])
        writer.writeheader()
        writer.writerows({key: json.dumps(row[key]) if key == "parameters" else row[key] for key in writer.fieldnames} for row in trials)
    tuned = make_model(name, config["seed"], study.best_params)
    with threadpool_limits(limits=4):
        fit_start = perf_counter()
        tuned.fit(x, y)
        tuned_fit_time = perf_counter() - fit_start
    tuned_cv, tuned_oof = cross_validate(name, study.best_params, x, y, folds, config)
    tuning = {"family": name, "best_parameters": study.best_params, "trials": len(trials),
              "startup_random_trials": 4, "adaptive_tpe_trials": max(0, len(trials) - 4),
              "optimisation_seconds": tuning_time, "tuned_fit_seconds": tuned_fit_time,
              "untuned_cv_macro_f1": chosen["cv_candidate_macro_f1"], **tuned_cv,
              "tuned_student_test": evaluate_model(tuned, data_root, cache_root),
              "untuned_student_test": chosen["student_test"]}
    write_json(output_root / "tuning.json", tuning)
    joblib.dump(tuned, models_folder / "tuned.joblib", compress=3)
    final_is_tuned = tuned_cv["cv_candidate_macro_f1"] >= chosen["cv_candidate_macro_f1"]
    final = tuned if final_is_tuned else joblib.load(models_folder / f"{name}.joblib")
    bundle = {"model": final, "model_name": name, "is_tuned": final_is_tuned,
              "parameters": study.best_params if final_is_tuned else MODEL_SPECS[name]["parameters"],
              "config": config, "feature_names": FEATURE_NAMES, "sklearn_version": sklearn.__version__,
              "label_convention": "monotone geometric proxy reference alignment"}
    joblib.dump(bundle, models_folder / "final.joblib", compress=3)
    write_json(output_root / "final_model.json", {key: value for key, value in bundle.items() if key != "model"})
    print(f"Final artifact: {name}, tuned={final_is_tuned}", flush=True)
