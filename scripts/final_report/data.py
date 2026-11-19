"""Load recorded artifacts and derive every number the report quotes.

Nothing here reads the raw dataset. Each value is either copied from an artifact or computed
from artifact fields (sums, medians, pooled averages), so it can be re-derived by anyone.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"

MODEL_ORDER = ["logistic", "gaussian_nb", "decision_tree", "random_forest", "hist_gradient_boosting"]
NAMES = {"logistic": "Logistic regression", "gaussian_nb": "Gaussian Naive Bayes", "decision_tree": "Decision tree",
         "random_forest": "Random forest", "hist_gradient_boosting": "Histogram gradient boosting"}

SOURCES = ["inventory.json", "folds.json", "training_manifest.json", "model_comparison.json", "selection.json",
           "bayesian_trials.json", "tuning.json", "final_model.json", "decoder_search.json", "decoder_comparison.json",
           "evaluation_freeze.json", "evaluator_manifest.json", "evaluator_results.json",
           "evaluator_model_comparison.json", "improved/results.json", "improved/student_test_results.json",
           "improved/evaluator_test_results.json", "logs/reserved_http.log", "logs/training.log",
           "../reports/evidence/api_smoke.json", "../configs/baseline.json"]


def f(x, d=4):
    return f"{x:.{d}f}"


def pct(x, d=2):
    return f"{100 * x:.{d}f}%"


def n(x):
    return f"{int(round(x)):,}"


def load(name):
    return json.loads((ART / name).read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pooled(rows, key, weight="observations"):
    return sum(r[key] * r[weight] for r in rows) / sum(r[weight] for r in rows)


def build() -> dict:
    D = {}
    inv = load("inventory.json")
    D["inv"] = inv
    for split in ["train", "student_test"]:
        rows = [r for r in inv if r["split"] == split]
        obs = [r["gps_observations"] for r in rows]
        steps = sum(r["total_steps"] for r in rows)
        D[split] = {
            "rows": rows, "trips": len(rows), "files": 4 * len(rows), "obs_total": sum(obs), "obs_min": min(obs),
            "obs_median": float(np.median(obs)), "obs_mean": float(np.mean(obs)), "obs_max": max(obs),
            "nodes_total": sum(r["nodes"] for r in rows), "nodes_median": float(np.median([r["nodes"] for r in rows])),
            "nodes_min": min(r["nodes"] for r in rows), "nodes_max": max(r["nodes"] for r in rows),
            "arcs_total": sum(r["arcs"] for r in rows), "arcs_median": float(np.median([r["arcs"] for r in rows])),
            "arcs_min": min(r["arcs"] for r in rows), "arcs_max": max(r["arcs"] for r in rows),
            "route_min": min(r["reference_route_arcs"] for r in rows),
            "route_median": float(np.median([r["reference_route_arcs"] for r in rows])),
            "route_max": max(r["reference_route_arcs"] for r in rows),
            "one_second_share": sum(r["one_second_steps"] for r in rows) / steps,
            "gaps60": sum(r["gaps_over_60s"] for r in rows),
            "gap_trips": sum(r["gaps_over_60s"] > 0 for r in rows),
            "max_gap": max(r["max_gap_s"] for r in rows),
            "weak_total": sum(r["weak_components"] for r in rows),
            "weak_median": float(np.median([r["weak_components"] for r in rows])),
            "weak_max": max(r["weak_components"] for r in rows),
            "isolated": sum(r["isolated_nodes"] for r in rows),
            "zero_len": sum(r["zero_length_arcs"] for r in rows),
            "zero_len_trips": sum(r["zero_length_arcs"] > 0 for r in rows),
            "nonfinite": sum(r["missing_nonfinite"] for r in rows),
            "nonincreasing": sum(r["non_increasing_timestamps"] for r in rows),
            "breaks": sum(r["reference_endpoint_breaks"] for r in rows),
            "break_trips": [r["trip"] for r in rows if r["reference_endpoint_breaks"]],
            "miss_pooled": sum(r["proxy_candidate_miss_rate"] * r["gps_observations"] for r in rows) / sum(obs),
            "fallback_pooled": sum(r["candidate_fallback_rate"] * r["gps_observations"] for r in rows) / sum(obs),
            "fallback_trips": sum(r["candidate_fallback_rate"] > 0 for r in rows),
            "label_med_med": float(np.median([r["proxy_label_distance_median_m"] for r in rows])),
            "label_p95_med": float(np.median([r["proxy_label_distance_p95_m"] for r in rows])),
            "excluded": sum(r["training_quality_excluded_observations"] for r in rows),
            "extent_max_km": max(max(r["network_extent_km"]) for r in rows),
            "lon": (min(r["longitude_extent"][0] for r in rows), max(r["longitude_extent"][1] for r in rows)),
            "lat": (min(r["latitude_extent"][0] for r in rows), max(r["latitude_extent"][1] for r in rows)),
            "prep_seconds": sum(r["preparation_seconds"] for r in rows),
        }
    clean = [r for r in D["train"]["rows"] if r["trip"] not in ("00000002", "00000033", "00000054")]
    D["train"]["miss_pooled_clean"] = (sum(r["proxy_candidate_miss_rate"] * r["gps_observations"] for r in clean)
                                       / sum(r["gps_observations"] for r in clean))
    # First-order east-west scale error of the local equirectangular projection: tan(lat0) * half latitude span.
    errs = []
    for r in inv:
        lat0 = np.mean(r["latitude_extent"])
        half = np.deg2rad((r["latitude_extent"][1] - r["latitude_extent"][0]) / 2)
        errs.append((abs(np.tan(np.deg2rad(lat0))) * half, r["trip"]))
    D["proj_err"] = max(errs)
    D["by_trip"] = {r["trip"]: r for r in inv}

    D["folds"] = load("folds.json")
    D["manifest"] = load("training_manifest.json")
    D["config"] = json.loads((ROOT / "configs/baseline.json").read_text())
    models = load("model_comparison.json")
    D["models"] = {m["name"]: m for m in models}
    D["selection"] = load("selection.json")
    D["trials"] = load("bayesian_trials.json")
    D["tuning"] = load("tuning.json")
    D["final"] = load("final_model.json")
    D["search"] = load("decoder_search.json")
    dec = load("decoder_comparison.json")
    D["dev"] = dec
    D["dev_summary"] = summarise_decoder(dec, lambda r: r)
    D["freeze"] = load("evaluation_freeze.json")
    D["ev_manifest"] = load("evaluator_manifest.json")
    ev = load("evaluator_results.json")
    D["ev"] = ev
    D["ev_summary"] = summarise_decoder(ev["requests"], lambda r: r["metrics"])
    D["ev_all"] = load("evaluator_model_comparison.json")
    D["imp"] = load("improved/results.json")
    D["imp_st"] = load("improved/student_test_results.json")
    D["imp_ev"] = load("improved/evaluator_test_results.json")
    D["smoke"] = json.loads((ROOT / "reports/evidence/api_smoke.json").read_text())
    D["http_log"] = (ART / "logs/reserved_http.log").read_text().strip().splitlines()
    D["hashes"] = {s: sha((ART / s).resolve()) for s in SOURCES}
    D["dev_seconds"] = [r["timings"]["total_prediction_seconds"] for r in dec]
    D["ev_http"] = [q["http_seconds"] for q in ev["requests"]]
    return D


def summarise_decoder(rows, metrics):
    total = sum(metrics(r)["observations"] for r in rows)
    out = {"observations": total}
    for part in ["classifier", "hmm"]:
        out[f"{part}_top1"] = sum(metrics(r)[part]["proxy_top1_accuracy"] * metrics(r)["observations"] for r in rows) / total
    for part in ["classifier", "hmm", "bridged_route"]:
        for key in ["route_length_f1", "ordered_route_lcs_f1", "direct_arc_continuity", "route_length_precision",
                    "route_length_recall"]:
            out[f"{part}_{key}"] = float(np.mean([metrics(r)[part][key] for r in rows]))
    out["resets"] = sum(metrics(r)["decoder_resets"] for r in rows)
    out["miss"] = sum(metrics(r)["proxy_candidate_miss_rate"] * metrics(r)["observations"] for r in rows) / total
    up = [r for r in rows if metrics(r)["hmm"]["proxy_top1_accuracy"] > metrics(r)["classifier"]["proxy_top1_accuracy"]]
    out["top1_up"] = [r["trip"] for r in up]
    out["top1_down"] = [r["trip"] for r in rows if r not in up]
    out["f1_up"] = [r["trip"] for r in rows if metrics(r)["bridged_route"]["route_length_f1"] > metrics(r)["classifier"]["route_length_f1"]]
    return out
