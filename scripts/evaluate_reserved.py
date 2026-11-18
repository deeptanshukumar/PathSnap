"""Verify the pre-evaluation freeze, then exercise reserved trips through the live API.

This is final evaluation, never training, model selection or decoder tuning.
"""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    root = Path("artifacts")
    freeze = json.loads((root / "evaluation_freeze.json").read_text())
    predictor_files = ["data", "geometry", "features", "reference", "decoder", "pipeline", "metrics", "evaluation"]
    checks = [f"pathsnap/{name}.py" for name in predictor_files]
    checks += ["configs/baseline.json", "artifacts/models/final.joblib", "artifacts/final_model.json"]
    for name in checks:
        if sha256(Path(name).read_bytes()).hexdigest() != freeze["source_sha256"][name]:
            raise RuntimeError(f"Frozen predictor/scoring definition changed: {name}")
    import httpx
    response = httpx.get(args.url.rstrip("/") + "/health")
    response.raise_for_status()
    if response.json()["mode"] != "evaluation":
        raise RuntimeError("Reserved evaluation requires a server explicitly in evaluation mode")
    output = root / "evaluator_results.json"
    subprocess.run([sys.executable, "scripts/smoke_api.py", "--url", args.url, "--output", str(output)], check=True)
    results = json.loads(output.read_text())
    results["prediction_freeze_verified"] = True
    results["evaluation_freeze_commit"] = "a61ce50"
    results["interpretation"] = "Reserved final-partition evaluation of the previously frozen system. Pointwise metrics remain geometric-alignment proxies. Results did not select or modify any model/decoder setting."
    output.write_text(json.dumps(results, indent=2) + "\n")
    rows = [{"trip": r["trip"], "observations": r["observations"], "request_id": r["request_id"],
             "classifier_proxy_accuracy": r["metrics"]["classifier"]["proxy_top1_accuracy"],
             "hmm_proxy_accuracy": r["metrics"]["hmm"]["proxy_top1_accuracy"],
             "bridged_overlap_f1": r["metrics"]["bridged_route"]["route_length_f1"],
             "bridged_direct_continuity": r["metrics"]["bridged_route"]["direct_arc_continuity"],
             "http_seconds": r["http_seconds"]} for r in results["requests"]]
    with (root / "evaluator_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
