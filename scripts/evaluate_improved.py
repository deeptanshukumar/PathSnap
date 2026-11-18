"""End-to-end scoring of the opt-in improved variant on one partition.

student_test is development data. evaluator_test is report-only: its results never
choose or change any feature, parameter or decoder setting.
"""

import argparse
from pathlib import Path

import numpy as np

from pathsnap.data import trip_directories
from pathsnap.evaluation import score_prediction
from pathsnap.experiments import write_json
from pathsnap.improved import ImprovedPipeline
from pathsnap.pipeline import Pipeline


def summarise(rows: list[dict]) -> dict:
    total = sum(r["observations"] for r in rows)
    pooled = lambda part, key: sum(r[part][key] * r["observations"] for r in rows) / total
    mean = lambda part, key: float(np.mean([r[part][key] for r in rows]))
    return {"observations": total,
            "classifier_proxy_top1": pooled("classifier", "proxy_top1_accuracy"),
            "hmm_proxy_top1": pooled("hmm", "proxy_top1_accuracy"),
            "classifier_route_f1_mean": mean("classifier", "route_length_f1"),
            "hmm_route_f1_mean": mean("hmm", "route_length_f1"),
            "bridged_route_f1_mean": mean("bridged_route", "route_length_f1"),
            "bridged_lcs_f1_mean": mean("bridged_route", "ordered_route_lcs_f1"),
            "classifier_continuity_mean": mean("classifier", "direct_arc_continuity"),
            "bridged_continuity_mean": mean("bridged_route", "direct_arc_continuity")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trips", type=Path, required=True, help="folder holding exactly the trip directories to score")
    parser.add_argument("--label", required=True, choices=["student_test", "evaluator_test"])
    args = parser.parse_args()
    folders = trip_directories(args.trips)
    out = {}
    for name, pipeline in [("default", Pipeline(Path("artifacts/models/final.joblib"))),
                           ("improved", ImprovedPipeline(Path("artifacts/improved/model.joblib")))]:
        rows = []
        for folder in folders:
            prediction = pipeline.predict(folder)
            rows.append({"trip": folder.name, "timings": prediction.timings, **score_prediction(prediction, folder)})
            print(name, folder.name, f"cls={rows[-1]['classifier']['proxy_top1_accuracy']:.4f}",
                  f"hmm={rows[-1]['hmm']['proxy_top1_accuracy']:.4f}", flush=True)
        out[name] = {"summary": summarise(rows), "trips": rows}
    out["note"] = ("Development partition." if args.label == "student_test" else
                   "Reserved partition, scored once after every setting was fixed; report-only, selected nothing.")
    write_json(Path(f"artifacts/improved/{args.label}_results.json"), out)
    for name in ("default", "improved"):
        print(name, {k: round(v, 4) for k, v in out[name]["summary"].items()})


if __name__ == "__main__":
    main()
