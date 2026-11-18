"""Export compact CSV tables and source/model hashes from recorded JSON results."""

import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path("artifacts")
    rows = []
    for model in json.loads((root / "model_comparison.json").read_text()):
        s = model["student_test"]["summary"]
        rows.append({"model": model["name"], "family": model["family"],
                     "configuration": json.dumps(model["parameters"], sort_keys=True),
                     "cv_candidate_macro_f1": model["cv_candidate_macro_f1"],
                     "cv_candidate_macro_f1_std": model["cv_candidate_macro_f1_std"],
                     **{key: s[key] for key in ["proxy_top1_accuracy", "candidate_macro_f1", "candidate_precision",
                        "candidate_recall", "candidate_log_loss", "mean_trip_proxy_arc_macro_f1",
                        "mean_trip_direct_arc_continuity", "mean_trip_route_length_f1", "model_inference_seconds"]},
                     "fit_seconds": model["full_fit_seconds"]})
    write_csv(root / "model_comparison.csv", rows)
    rows = []
    for row in json.loads((root / "decoder_comparison.json").read_text()):
        rows.append({"trip": row["trip"], "observations": row["observations"],
                     "classifier_proxy_accuracy": row["classifier"]["proxy_top1_accuracy"],
                     "hmm_proxy_accuracy": row["hmm"]["proxy_top1_accuracy"],
                     "bridged_overlap_f1": row["bridged_route"]["route_length_f1"],
                     "bridged_direct_continuity": row["bridged_route"]["direct_arc_continuity"],
                     "bridged_ordered_lcs_f1": row["bridged_route"]["ordered_route_lcs_f1"],
                     "decoder_resets": row["decoder_resets"],
                     "prediction_seconds": row["timings"]["total_prediction_seconds"]})
    write_csv(root / "decoder_comparison.csv", rows)
    files = [*Path("pathsnap").glob("*.py"), Path("configs/baseline.json"), root / "models/final.joblib"]
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "sha256": {str(p): sha256(p.read_bytes()).hexdigest() for p in sorted(files)},
                  "notes": "Original decoder search retained in decoder_initial.json and decoder_search.json. Final route assembly includes directed loops for backwards projection on the same arc. All pointwise predictions are unchanged by that reconstruction correction."}
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
