# Project state

Team PT-088: Chiranth R (PES1UG24AM354), Darshan S Govekar (PES1UG24AM355), Daruru Bhuvan Sai (PES1UG24AM356), Deeptanshu Kumar (PES1UG24AM357). Deadline: **18 November 2026, 15:00 IST**, as specified by the team.

## Implemented and verified

| Milestone | Implementation and evidence |
|---|---|
| Requirements and data audit | Five organiser PDFs read; 80 training / 10 development trips. Inventory, missing-value/time-gap/network checks and raw hashes in `artifacts/inventory.json`. |
| Portable prediction | Strict loading, local metre geometry, spatial indexing, eight directed candidates and 13 route-independent features. Prediction does not read references. |
| Five-model comparison | Logistic, Gaussian NB, decision tree, random forest and histogram boosting; identical three whole-trip folds, seed 357. See `model_comparison.json/.csv` and `folds.json`. |
| Bayesian tuning | Twelve TPE trials, four startup/eight adaptive, training-trip CV only. Tuned boosting CV candidate Macro F1 0.9290. Trial distributions and scores retained. |
| Sequential decoding | Directed routing, partial-edge positions, log-space Viterbi, explicit resets and road bridges, including same-arc loops. Decoder search uses development trips only. |
| API and UI | Real cross-origin browser interaction; all ten development and all ten reserved trips succeeded over actual HTTP. API maps omit reference geometry. |
| Correctness checks | 13 tests passed: geometry, candidate lookup, grouping, brute-force Viterbi equivalence, reference separation, same-arc loops, API validation/CORS and evaluation-mode isolation. |
| Deliverables | 18-page PDF and 11-slide editable PPTX generated from measured results, visually inspected, and structurally validated. Folium maps, genuine browser screenshots, viva guide and source generators are included. |

Python 3.12.16 / NumPy 2.3.5 / SciPy 1.16.3 / scikit-learn 1.7.2 on Linux CPU were exercised. `.venv` is isolated. macOS ML execution and physical Mac-to-Linux networking have not been executed here. Kaggle was authorised but not needed for the measured workload.

## Measured results

| Metric | Development classifier | Development HMM/bridge | Reserved classifier | Reserved HMM/bridge |
|---|---:|---:|---:|---:|
| Pooled proxy pointwise accuracy | 87.46% | 85.42% | 88.44% | 77.78% |
| Mean length-weighted route overlap F1 | 0.9339 | 0.9570 | 0.9160 | 0.9250 |
| Mean direct arc continuity | 0.7432 | 0.9961 | 0.7604 | 0.9976 |
| Mean ordered route LCS F1 | 0.7332 | 0.8012 | 0.7630 | 0.7547 |

Bridge metrics include intermediate routed arcs. HMM-state direct continuity before bridges is 0.8822 on development and 0.8440 on reserved data. Pointwise labels are inferred by monotone reference alignment, not officially supplied per-GPS labels. Better continuity does not imply better pointwise or sequence accuracy.

## Final-evaluation boundary

The initial package omitted reserved trips. The user later supplied `evaluator_test.zip`. Commit **a61ce50** froze the predictor/scorer before archive access; `artifacts/evaluation_freeze.json` records weights, settings and source hashes. All ten reserved trips (19,674 observations) ran through the unchanged evaluation-mode API. Results/request IDs are in `evaluator_results.json/.csv`; archive SHA256 and IDs are in `evaluator_manifest.json`. A later report-only run (`scripts/evaluate_reserved_models.py`, `artifacts/evaluator_model_comparison.json`) scored all five model families on the same reserved trips and is shown on the model trace page; the tuned boosting row reproduces the frozen results exactly and no selection changed.

No evaluator results selected or modified features, labels convention, model weights/family, decoder parameters, thresholds or search space. Report/UI authoring can be updated without changing the frozen predictor/scorer.

## UI update (branch `chiranth/ui-polish`, 18 November 2026)

- `ui/index.html`: map-first evaluator screen. Leaflet draws the reconstructed route with toggles for raw GPS, classifier-only picks and resets; falls back to the server's Folium map if the API lacks `/trace`.
- `ui/details.html` ("Model trace"): separate tab that follows the main screen via BroadcastChannel. Shows each pipeline stage with server-measured numbers and a point-by-point replay of candidates, P(match), emissions and Viterbi transition terms.
- Verified on macOS (Python 3.13, same pinned NumPy/SciPy/scikit-learn) against a local development-mode API in Chrome; 13 tests pass. The Linux API must pull this branch and restart to serve `/trace`.

## Current demonstration

- API: port 8000, evaluation mode, ignored local `data/evaluator_test/`.
- Static UI: port 8010. Chiranth's Mac serves `ui/` using Python 3 and connects to this laptop's LAN IP on port 8000.
- Public repository: https://github.com/deeptanshukumar/PathSnap. Raw data and caches remain ignored. Final verification details are in `artifacts/verification.json`.

## Team steps still open

1. Verify Chiranth's physical Mac connection. Browser evidence proves local cross-origin operation only.
2. Capture the actual receiving-terminal log as `reports/evidence/api_terminal.png`, then regenerate the report. GNOME denied automated native-window capture; no terminal image was fabricated.
3. Review the report and practise `docs/VIVA.md`, retaining the AI disclosure. Bhuvan's novelty remains pending and must not use reserved results for selection.

See `docs/DECISIONS.md` for important choices and the corrected same-arc-loop metric history. Keep the source, results and concise state here; do not add overlapping context files.

## Deliverables audit and viva study guide

Latest code at `3d05683` was checked against all five organiser PDFs. `docs/DELIVERABLES_AUDIT.md` records complete, partial and missing items. Core predictor/scorer hashes still match; 13 local tests pass. The live API still needs restart for the new trace endpoints, and genuine two-laptop/terminal evidence plus current report screenshots remain open. `reports/PathSnap_Viva_Study_Guide.pdf` adds 154 short Q&As, required-question coverage, model metrics and Chiranth's full report-only reserved comparison. It is rehearsal material, separate from the single technical submission report.

## Repository documentation cleanup

README now describes the project, current UI/trace endpoints, measured results and portable server/UI setup without teammate-specific instructions. The personal handoff was removed at the user's request; its required setup and trace guidance is covered in README, and the audit link was updated. No predictor, experiment results or runtime configuration changed.
