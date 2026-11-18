"""Train and select the opt-in improved variant. Never reads evaluator data.

Selection uses grouped CV on the 80 training trips (classifier) and the ten student-test
trips (decoder weights), the same partitions the baseline used.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import optuna
from threadpoolctl import threadpool_limits

from pathsnap.data import load_reference, load_trip, trip_directories
from pathsnap.decoder import DecoderConfig, decode, emission_probabilities
from pathsnap.experiments import cross_validate, grouped_folds, write_json
from pathsnap.features import Candidates
from pathsnap.geometry import RoadNetwork
from pathsnap.improved import IMPROVED_FEATURE_NAMES, context_features, decoder_with, weighted_scores
from pathsnap.metrics import candidate_metrics, point_metrics, route_metrics
from pathsnap.models import make_model, search_parameters

BASE = 13


def cached_trip(data_root: Path, cache: Path, split: str, trip: str):
    network = RoadNetwork(load_trip(data_root / split / trip))
    with np.load(cache / split / f"{trip}.npz") as d:
        data = {key: d[key] for key in d.files}
    candidates = Candidates(data["arcs"], data["fraction"], data["distance"], data["features"], data["fallback"])
    features = np.concatenate([data["features"], context_features(network, candidates)], axis=-1)
    return network, candidates, features, data


def training_rows(data_root: Path, cache: Path, config: dict, stride: int):
    xs, ys, groups = [], [], []
    for folder in trip_directories(data_root / "train"):
        _, _, features, data = cached_trip(data_root, cache, "train", folder.name)
        selected = (np.arange(len(data["arcs"])) % stride == 0)
        selected &= data["reference_distance"] <= config["training_label_max_distance_m"]
        valid = (data["arcs"] >= 0) & selected[:, None]
        xs.append(features[valid]); ys.append(data["labels"][valid].astype(np.int8))
        groups.append(np.full(int(valid.sum()), folder.name))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(groups)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    out = Path("artifacts/improved")
    out.mkdir(parents=True, exist_ok=True)
    final = joblib.load("artifacts/models/final.joblib")
    config = final["config"]
    seed = config["seed"]
    x, y, groups = training_rows(args.data_root, args.cache, config, args.stride)
    folds = grouped_folds(x, y, groups, config)
    print(f"rows={len(y)} positives={int(y.sum())} features={x.shape[1]}", flush=True)

    # 1) Same model and parameters as the served baseline, base vs improved features.
    baseline_params = final["parameters"]
    base_cv, _ = cross_validate("hist_gradient_boosting", baseline_params, x[:, :BASE], y, folds, config)
    ctx_cv, _ = cross_validate("hist_gradient_boosting", baseline_params, x, y, folds, config)
    print(f"CV macro F1 base={base_cv['cv_candidate_macro_f1']:.4f} improved={ctx_cv['cv_candidate_macro_f1']:.4f}", flush=True)

    # 2) Bayesian re-tuning on the improved features (same search space and objective).
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=4))
    study.enqueue_trial(baseline_params)

    def objective(trial):
        result, _ = cross_validate("hist_gradient_boosting", search_parameters(trial, "hist_gradient_boosting"), x, y, folds, config)
        print(f"TPE trial {trial.number}: {result['cv_candidate_macro_f1']:.4f}", flush=True)
        return result["cv_candidate_macro_f1"]

    start = perf_counter()
    study.optimize(objective, n_trials=args.trials)
    tuning_seconds = perf_counter() - start
    best = study.best_params
    tuned_cv, _ = cross_validate("hist_gradient_boosting", best, x, y, folds, config)
    model = make_model("hist_gradient_boosting", seed, best)
    with threadpool_limits(limits=4):
        start = perf_counter(); model.fit(x, y); fit_seconds = perf_counter() - start
        train_fit = candidate_metrics(y, model.predict_proba(x)[:, 1])
    base_model = make_model("hist_gradient_boosting", seed, baseline_params)
    with threadpool_limits(limits=4):
        base_model.fit(x[:, :BASE], y)
        base_train_fit = candidate_metrics(y, base_model.predict_proba(x[:, :BASE])[:, 1])

    # 3) Student-test: classifier scores once, then a small decoder grid (development data only).
    grid = [(p, b, norm) for p in (1.0, 2.0, 3.0, 4.0) for b in (10.0, 25.0, 50.0) for norm in (False,)]
    trips, labels_all, scores_all, base_scores_all = [], [], [], []
    decoder_base = DecoderConfig(**config["decoder"])
    for folder in trip_directories(args.data_root / "student_test"):
        network, candidates, features, data = cached_trip(args.data_root, args.cache, "student_test", folder.name)
        valid = candidates.valid
        scores, base_scores = np.zeros(valid.shape), np.zeros(valid.shape)
        with threadpool_limits(limits=4):
            scores[valid] = model.predict_proba(features[valid])[:, 1]
            base_scores[valid] = final["model"].predict_proba(features[valid][:, :BASE])[:, 1]
        labels_all.append(data["labels"][valid]); scores_all.append(scores[valid]); base_scores_all.append(base_scores[valid])
        reference = data["reference_arcs"]
        route = load_reference(folder, len(network.trip.arcs))
        rows = np.arange(len(scores))
        row = {"trip": folder.name, "observations": len(rows),
               "base_classifier_correct": int(np.sum(candidates.arc_ids[rows, emission_probabilities(base_scores, valid).argmax(1)] == reference)),
               "classifier_correct": int(np.sum(candidates.arc_ids[rows, emission_probabilities(scores, valid).argmax(1)] == reference)),
               "decoder": {}}
        for p, b, norm in grid:
            cfg = decoder_with(decoder_base, beta_m=b, normalize_transition_rows=norm)
            decoded = decode(network, candidates, weighted_scores(scores, p), cfg)
            row["decoder"][f"{p}|{b}|{norm}"] = {"correct": int(np.sum(decoded.arc_ids == reference)),
                                                 **route_metrics(decoded.arc_ids, route, network.trip.arcs, network.lengths)}
        trips.append(row)
        print("student_test", folder.name, flush=True)
    total = sum(r["observations"] for r in trips)
    grid_scores = {key: sum(r["decoder"][key]["correct"] for r in trips) / total for key in trips[0]["decoder"]}
    chosen = max(grid_scores, key=grid_scores.get)
    power, beta, norm = chosen.split("|")
    dev = candidate_metrics(np.concatenate(labels_all), np.concatenate(scores_all))
    base_dev = candidate_metrics(np.concatenate(labels_all), np.concatenate(base_scores_all))

    decoder = {**config["decoder"], "beta_m": float(beta), "normalize_transition_rows": norm == "True"}
    bundle = {"model": model, "model_name": "hist_gradient_boosting", "variant": "improved", "is_tuned": True,
              "parameters": best, "config": {**config, "decoder": decoder}, "emission_power": float(power),
              "feature_names": IMPROVED_FEATURE_NAMES, "training_stride": args.stride,
              "label_convention": "monotone geometric proxy reference alignment"}
    joblib.dump(bundle, out / "model.joblib", compress=3)
    results = {
        "variant": "improved (opt-in; the default served model is unchanged)",
        "selection": "classifier: grouped CV on 80 training trips; decoder: pooled proxy Top-1 on student_test. Evaluator data never read.",
        "feature_names": IMPROVED_FEATURE_NAMES, "training_rows": int(len(y)), "training_stride": args.stride,
        "baseline": {"train_candidate_macro_f1": base_train_fit["candidate_macro_f1"],
                     "cv_candidate_macro_f1": base_cv["cv_candidate_macro_f1"], "cv_candidate_macro_f1_std": base_cv["cv_candidate_macro_f1_std"],
                     "student_test_candidate_macro_f1": base_dev["candidate_macro_f1"], "student_test_log_loss": base_dev["candidate_log_loss"],
                     "student_test_classifier_top1": sum(r["base_classifier_correct"] for r in trips) / total},
        "improved_untuned_cv_candidate_macro_f1": ctx_cv["cv_candidate_macro_f1"],
        "improved": {"parameters": best, "train_candidate_macro_f1": train_fit["candidate_macro_f1"],
                     "cv_candidate_macro_f1": tuned_cv["cv_candidate_macro_f1"], "cv_candidate_macro_f1_std": tuned_cv["cv_candidate_macro_f1_std"],
                     "cv_folds": tuned_cv["cv_folds"], "fit_seconds": fit_seconds, "tuning_seconds": tuning_seconds,
                     "tuning_trials": [{"number": t.number, "value": t.value, "parameters": t.params} for t in study.trials],
                     "student_test_candidate_macro_f1": dev["candidate_macro_f1"], "student_test_log_loss": dev["candidate_log_loss"],
                     "student_test_classifier_top1": sum(r["classifier_correct"] for r in trips) / total,
                     "decoder_grid_student_test_top1": grid_scores,
                     "chosen_decoder": {"emission_power": float(power), **decoder},
                     "student_test_viterbi_top1": grid_scores[chosen]},
        "trips": trips,
    }
    write_json(out / "results.json", results)
    print(json.dumps({k: v for k, v in results.items() if k in ("baseline",)}, indent=1))
    print(json.dumps({k: v for k, v in results["improved"].items() if k not in ("cv_folds", "tuning_trials", "decoder_grid_student_test_top1")}, indent=1))


if __name__ == "__main__":
    main()
