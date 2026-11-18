# PathSnap

Team PT-088's ML hackathon system for matching noisy GPS observations to directed road networks. The Linux laptop hosts the model/API; Chiranth's Mac runs the static evaluator UI.

**Start here:** [Chiranth's Mac handoff](docs/CHIRANTH_HANDOFF.md), [technical report](reports/PathSnap_Technical_Report.pdf), [presentation](reports/PathSnap_Presentation_Final.pptx), [viva preparation](docs/VIVA.md), [current state](docs/PROJECT_STATE.md), [decisions](docs/DECISIONS.md).

**Viva preparation:** [20-page study PDF](reports/PathSnap_Viva_Study_Guide.pdf) with 154 short Q&As, including Chiranth's trace UI and all-model metrics. [Deliverables audit](docs/DELIVERABLES_AUDIT.md) lists remaining submission gaps.

## Implemented and measured

- Validated raw loading, metre-based segment projection, spatial candidate indexing and 13 route-independent features.
- Five algorithms on the same three trip-grouped CV folds; 12 Optuna TPE trials for histogram gradient boosting.
- Directed shortest-path transitions, segmented Viterbi decoding, road bridges and explicit outage/disconnection resets.
- FastAPI, a static Mac-compatible UI, Folium maps, recorded experiments, a consolidated PDF and viva deck.
- 12 passing correctness/API tests and actual HTTP requests for all ten development trips.

| Development metric | Classifier only | Viterbi / reconstructed route |
|---|---:|---:|
| Pooled proxy pointwise accuracy | 87.46% | 85.42% |
| Mean trip length-weighted route overlap F1 | 0.9339 | 0.9570 |
| Mean trip direct arc continuity | 0.7432 | 0.9961 |

The reconstructed route includes intermediate shortest-path arcs; HMM-state continuity before bridging is 0.8822. Pointwise labels are **inferred** by monotone alignment to `.route`, which contains ordered arcs, not per-GPS labels. These are development results, not official evaluator scores. See [metric definitions and per-trip results](artifacts/decoder_comparison.json).

The separately frozen reserved evaluation completed all ten trips (19,674 GPS observations): classifier proxy accuracy 88.44%, HMM proxy accuracy 77.78%, reconstructed-route overlap F1 0.9250. These results did not select or change any setting; see [final-partition records](artifacts/evaluator_results.json) and [the pre-evaluation freeze](artifacts/evaluation_freeze.json).

**Still requires the team:** physical Mac-to-Linux test, a genuine receiving-terminal screenshot, report review and Bhuvan's novelty. No novelty is claimed for the baseline.

## Setup: Linux or macOS

Use Python 3.12 or newer. Python 3.12.16, NumPy 2.3.5, SciPy 1.16.3 and scikit-learn 1.7.2 were exercised locally on Linux/CPU. macOS ML execution is designed for portability but has not been executed here. The Mac UI needs only Python 3 and a browser.

```bash
git clone https://github.com/deeptanshukumar/PathSnap.git
cd PathSnap
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,report]'
```

For the exact tested package snapshot, install `requirements.lock.txt`, then `python -m pip install -e . --no-deps`. The lock includes test/report extras; it excludes machine-specific editable paths. No global Python changes are required. Only load the team's trusted joblib model artifacts.

## Data layout

Keep raw data outside Git and pass its location explicitly. The original package has 80 `train` and ten `student_test` folders. The later `evaluator_test.zip` has ten reserved trips and is extracted locally under ignored `data/evaluator_test/`.

```text
Student folder/
  train/00000001/00000001.{track,nodes,arcs,route}
  ...
  student_test/00000000/00000000.{track,nodes,arcs,route}
  ...
```

`track`: longitude, latitude, timestamp seconds; `nodes`: longitude, latitude; `arcs`: zero-based source/target node IDs; `route`: chronological reference arc IDs. IDs are trip-local. Rows must never be joined by their raw row number.

## Run the API on this Linux laptop

The saved final model is included; retraining is not needed to demonstrate it.

```bash
source .venv/bin/activate
python -m pathsnap.cli serve \
  --data-root "/path/to/Student folder" \
  --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000` locally. `/docs` provides the FastAPI schema. The server visibly uses **development mode** and exposes exactly the ten student-test IDs. Each request performs live inference and prints a `PATHSNAP` log with its request ID.

## Run the UI on Chiranth's Mac

```bash
git clone https://github.com/deeptanshukumar/PathSnap.git
cd PathSnap
python3 -m http.server 8010 --directory ui
```

Open `http://localhost:8010`. Enter `http://<Linux-LAN-IP>:8000` in **Model server**, connect, select any trip and reconstruct. `127.0.0.1` on the Mac points to the Mac, so use the Linux laptop's actual LAN address. Both devices must be on a network that permits peer connections. If connection fails, try the API `/health` URL directly in the Mac browser and check network isolation/firewall settings. CORS is configured for this unauthenticated LAN demo; do not deploy this demo API as a public Internet service.

Folium JavaScript/CSS and OpenStreetMap tiles require internet access. Keep a generated development map available as a local artifact; it also needs those external display assets.

## Reserved final-evaluation mode

The supplied evaluator archive has been extracted to `data/evaluator_test/`. No training or tuning command reads this folder. The live server currently uses this evaluation mode.

```bash
python -m pathsnap.cli serve \
  --mode evaluation --evaluation-root data/evaluator_test \
  --host 0.0.0.0 --port 8000
```

The server refuses evaluation mode without exactly ten trip directories. The UI loads the new IDs from the API. Prediction reads only `.track`, `.nodes` and `.arcs`; reference scoring reads `.route` afterward. Reference labels and geometry never enter the evaluator response or map. If `.route` is absent, predictions still return and reference metrics are unavailable. Do not use evaluator requests to revise the model or settings.

`python scripts/evaluate_reserved.py` verifies the frozen predictor/scorer hashes and records actual HTTP results separately in `artifacts/evaluator_results.json/.csv`. It requires an evaluation-mode server. Commit `a61ce50` froze the predictor before archive access. Do not use the final results for subsequent model or decoder selection.

## Reproduce experiments

Run these from the repository root in the isolated environment. Replace the dataset path once; it is not hardcoded in source.

```bash
PATHSNAP_DATA_ROOT="/path/to/Student folder"
python -m pathsnap.cli prepare --data-root "$PATHSNAP_DATA_ROOT"
python -m pathsnap.cli train --data-root "$PATHSNAP_DATA_ROOT"
python scripts/tune_decoder.py --data-root "$PATHSNAP_DATA_ROOT"
python -m pathsnap.cli evaluate --data-root "$PATHSNAP_DATA_ROOT"
python scripts/export_results.py
python -m pathsnap.cli report --data-root "$PATHSNAP_DATA_ROOT"
python -m pytest -q
```

Training overwrites experiment outputs and the final model. `train` restores baseline decoder settings; run `tune_decoder.py` and `evaluate` afterward to reproduce the chosen decoder. Restart any running API after updating its model. The six-setting decoder search uses permitted student-test development data; its reported performance is not an untouched test estimate.

Training selects every fifth observation after a fixed 75 m proxy-label quality filter, using all 80 training trips. Every student-test observation is scored. Fits use fold-local scaling; boosting's automatic row-random early stopping is disabled. Fixed seeds do not guarantee identical floating-point timings/results across libraries or hardware. No GPU is required; Kaggle was authorised but not used because measured CPU execution was adequate.

The optional deck builder is `scripts/build_presentation.mjs`. It uses Codex's Artifact Tool 2.8.134 runtime and presentation-skill validators, supplied through `RUNTIME_NODE_MODULES`, `RUNTIME_PYTHON` and `ARTIFACT_SKILL_DIR`. These are optional artifact-authoring dependencies, not application requirements. Set `PRESENTATION_OUTPUT` to a new filename when rebuilding. Native chart values are rounded to six decimal places for Office interoperability; full result precision remains in JSON. The deck includes editable charts/tables and source notes.

## Evidence for the final demo

With the API running:

```bash
python scripts/smoke_api.py --url http://127.0.0.1:8000
```

This exercises all ten active trips and saves genuine HTTP summaries to `reports/evidence/api_smoke.json`. Development evidence is already recorded. A final evaluator smoke run is final evaluation: retain it separately and do not tune afterward.

During the physical two-laptop demo, capture the working Mac UI and the receiving Linux terminal showing the corresponding `PATHSNAP` request ID. Save the real terminal screenshot as `reports/evidence/api_terminal.png`, then rerun `report`. Existing screenshots establish local cross-origin browser operation only. GNOME denied automated native-window capture; no terminal screenshot has been fabricated.

## Repository map

| Location | Responsibility |
|---|---|
| `pathsnap/data.py`, `geometry.py` | Input validation, local metres, directed routing |
| `features.py`, `reference.py`, `prepare.py` | Candidate features, separate proxy-label alignment, inventory/cache |
| `models.py`, `experiments.py`, `metrics.py` | Five-model registry, grouped CV/TPE, explicit metrics |
| `decoder.py`, `pipeline.py`, `evaluation.py` | Viterbi, route reconstruction, prediction/scoring boundary |
| `api.py`, `ui/` | Model server and portable browser interface |
| `reporting.py`, `scripts/` | Actual-result plots, PDF, decoder search, HTTP evidence and deck source |
| `configs/`, `artifacts/` | Reproducible configuration, hashes, folds, trials, metrics, final model |
| `reports/`, `docs/` | Consolidated report, deck, actual maps/screenshots, concise project context and viva |

Add Bhuvan's features in `features.py` with an updated feature schema; add models through `models.py`; add decoder variants through `DecoderConfig`. Such changes require fresh grouped experiments and artifact regeneration. Keep evaluator data isolated and preserve the measured baseline for ablations.

AI assistance: Codex helped inspect materials, implement and test the system, execute experiments and draft explanations. All reported metrics come from actual runs. Team members must review and understand the work and meet the organisers' disclosure requirements. See the report for references and limitations.
