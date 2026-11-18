# PathSnap

Map matching for noisy GPS trajectories using machine learning and directed HMM/Viterbi decoding. PathSnap scores nearby road candidates, reconstructs connected routes, and exposes predictions through a FastAPI server and interactive browser UI.

Built for the PES University ML hackathon by team PT-088. The model server and browser UI can run on separate laptops; GPU compute is not required.

[Technical report](reports/PathSnap_Technical_Report.pdf) · [Presentation](reports/PathSnap_Presentation_Final.pptx) · [Viva study guide](reports/PathSnap_Viva_Study_Guide.pdf) · [Deliverables audit](docs/DELIVERABLES_AUDIT.md) · [Project state](docs/PROJECT_STATE.md) · [Design decisions](docs/DECISIONS.md)

## Features

- Validated raw loading, metre-based segment projection, spatial candidate indexing and 13 route-independent features.
- Five algorithms on the same three trip-grouped CV folds; 12 Optuna TPE trials for histogram gradient boosting.
- Directed shortest-path transitions, segmented Viterbi decoding, road bridges and explicit outage/disconnection resets.
- FastAPI inference, a map-first Leaflet UI, interactive Folium maps and a separate Model trace page with candidate probabilities and transition explanations.
- Recorded experiments, model comparisons, a consolidated technical report, presentation and a 154-question viva study guide.
- 13 passing correctness/API tests and recorded HTTP evaluation of all ten development and all ten reserved trips.

```text
GPS + directed road network -> candidates -> 13 features -> classifier scores
    -> Viterbi decoding -> route reconstruction -> API + map + model trace
```

## Measured results

| Development metric | Classifier only | Viterbi / reconstructed route |
|---|---:|---:|
| Pooled proxy pointwise accuracy | 87.46% | 85.42% |
| Mean trip length-weighted route overlap F1 | 0.9339 | 0.9570 |
| Mean trip direct arc continuity | 0.7432 | 0.9961 |

The reconstructed route includes intermediate shortest-path arcs; HMM-state continuity before bridging is 0.8822. Pointwise labels are **inferred** by monotone alignment to `.route`, which contains ordered arcs, not per-GPS labels. These are development results, not official evaluator scores. See [metric definitions and per-trip results](artifacts/decoder_comparison.json).

The separately frozen reserved evaluation completed all ten trips (19,674 GPS observations): classifier proxy accuracy 88.44%, HMM proxy accuracy 77.78%, reconstructed-route overlap F1 0.9250. These results did not select or change any setting; see [final-partition records](artifacts/evaluator_results.json) and [the pre-evaluation freeze](artifacts/evaluation_freeze.json).

**Verification limits:** physical two-laptop networking and the receiving-terminal screenshot remain unverified. The technical report needs current UI screenshots, and the API must be restarted after pulling endpoint changes. See the [deliverables audit](docs/DELIVERABLES_AUDIT.md). The baseline uses established methods; no scientific novelty is claimed.

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

## Start the model server

The saved final model is included; retraining is not needed to demonstrate it.

```bash
source .venv/bin/activate
python -m pathsnap.cli serve \
  --data-root "/path/to/Student folder" \
  --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000` locally. `/docs` provides the FastAPI schema. This command starts **development mode** and exposes exactly the ten student-test IDs. Each request performs live inference and prints a `PATHSNAP` log with its request ID.

## Start the browser UI

On the same machine or a second laptop, serve the checked-out UI with Python 3. No ML packages or raw dataset are needed on a UI-only laptop.

```bash
git clone https://github.com/deeptanshukumar/PathSnap.git
cd PathSnap
python3 -m http.server 8010 --directory ui
```

Open `http://localhost:8010`. Enter `http://<server-LAN-IP>:8000` in **Model server**, connect, select a trip and reconstruct its route. When both components run on one machine, use `http://127.0.0.1:8000`. On separate laptops, use the model server's LAN address; localhost always refers to the device running the browser.

Click **Model trace** to inspect the model-selection results and replay candidate probabilities, emissions and transition terms. Keep both UI tabs open for automatic synchronisation. Restart the API after code updates so the trace endpoints are available.

Both devices must be on a network that permits peer connections. To diagnose connectivity, open the API's `/health` URL from the UI laptop and check network isolation/firewall settings. CORS is configured for this unauthenticated LAN demonstration.

| Endpoint | Purpose |
|---|---|
| `GET /health`, `GET /trips` | Server readiness and active trip IDs |
| `POST /match` | Live prediction and separate server-side scoring |
| `GET /maps/{request_id}` | Interactive Folium prediction map |
| `GET /trace/{request_id}` | Pipeline stages and observation-level trace |
| `GET /explain/{request_id}/{observation}` | Candidate and transition explanations |

Folium JavaScript/CSS and OpenStreetMap tiles require internet access. Keep a generated development map available as a local artifact; it also needs those external display assets.

## Reserved final-evaluation mode

Extract the supplied evaluator archive to `data/evaluator_test/` before using this mode. The dataset is not included in Git. No training or tuning command reads this folder.

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

Training selects original observation indices divisible by five and applies a fixed 75 m proxy-label quality filter, using all 80 training trips. Every student-test observation is scored. Fits use fold-local scaling; boosting's automatic row-random early stopping is disabled. Fixed seeds do not guarantee identical floating-point timings/results across libraries or hardware. Experiments were run on CPU.

The optional deck builder is `scripts/build_presentation.mjs`. It uses Codex's Artifact Tool 2.8.134 runtime and presentation-skill validators, supplied through `RUNTIME_NODE_MODULES`, `RUNTIME_PYTHON` and `ARTIFACT_SKILL_DIR`. These are optional artifact-authoring dependencies, not application requirements. Set `PRESENTATION_OUTPUT` to a new filename when rebuilding. Native chart values are rounded to six decimal places for Office interoperability; full result precision remains in JSON. The deck includes editable charts/tables and source notes.

## Evidence for the final demo

With the API running:

```bash
python scripts/smoke_api.py --url http://127.0.0.1:8000
```

This exercises all ten active trips and saves genuine HTTP summaries to `reports/evidence/api_smoke.json`. Development evidence is already recorded. A final evaluator smoke run is final evaluation: retain it separately and do not tune afterward.

During the physical two-laptop demo, capture the working UI and the receiving API terminal showing the corresponding `PATHSNAP` request ID. Save the real terminal screenshot as `reports/evidence/api_terminal.png`, then rerun `report`. Existing screenshots establish local cross-origin browser operation only.

## Repository map

| Location | Responsibility |
|---|---|
| `pathsnap/data.py`, `geometry.py` | Input validation, local metres, directed routing |
| `features.py`, `reference.py`, `prepare.py` | Candidate features, separate proxy-label alignment, inventory/cache |
| `models.py`, `experiments.py`, `metrics.py` | Five-model registry, grouped CV/TPE, explicit metrics |
| `decoder.py`, `pipeline.py`, `evaluation.py` | Viterbi, route reconstruction, prediction/scoring boundary |
| `api.py`, `trace.py`, `ui/` | Model server, observation explanations and portable browser interface |
| `reporting.py`, `scripts/` | Actual-result plots, PDF, decoder search, HTTP evidence and deck source |
| `configs/`, `artifacts/` | Reproducible configuration, hashes, folds, trials, metrics, final model |
| `reports/`, `docs/` | Consolidated report, deck, actual maps/screenshots, concise project context and viva |

Extend candidate features in `features.py` with an updated feature schema; add models through `models.py`; add decoder variants through `DecoderConfig`. Such changes require fresh grouped experiments and artifact regeneration. Keep evaluator data isolated and preserve the measured baseline for ablations.

AI assistance: Codex helped inspect materials, implement and test the system, execute experiments and draft explanations. All reported metrics come from actual runs. Team members must review and understand the work and meet the organisers' disclosure requirements. See the report for references and limitations.
