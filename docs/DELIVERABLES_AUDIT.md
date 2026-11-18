# Deliverables audit - 18 November 2026

Audited `main` at `3d05683` against all five organiser PDFs. **Core software and experiment deliverables are present. Submission/demonstration evidence is not fully complete.** This audit does not change the frozen predictor or certify team understanding/plagiarism compliance.

## Required final items and current evidence

| Requirement | Status | Evidence / remaining action |
|---|---|---|
| Public GitHub repository and complete source | Present | Public repository; processing, training, tuning, decoder, API, UI, maps and generators tracked. |
| Saved final model and inference configuration | Present | `artifacts/models/final.joblib`, `final_model.json`, `configs/baseline.json`; model/features/decoder/scorer still match pre-evaluation freeze. |
| Portable installation and two-laptop instructions | Present | `README.md` server/UI setup, pinned core versions and dependency snapshot. |
| Consolidated technical PDF | Present, needs final refresh | `reports/PathSnap_Technical_Report.pdf`, 18 pages. Keep this as the single submission report. The separate study guide is for rehearsal. |
| Raw formats, preprocessing, missing values and timestamps | Present | `data.py`, `prepare.py`, `inventory.json`; report Sections 2-2.1. |
| Geographic extent, network connectivity, isolated nodes and anomalies | Present | Per-trip inventory plus EDA figures; 90 development/training records. |
| File availability and file sizes | Partial | Validated loading and SHA256 hashes exist, but the retained inventory does not include per-file byte sizes. Minor exploration-record gap. |
| Features and geographic distance justification | Present | Thirteen route-independent features, local-metre projection, candidate search and label-alignment explanation. |
| At least five independently fitted models | Present | Five model factories; configurations, common folds, training logs, fit/CV/student-test results. |
| Required per-model questions | Present | Report Sections 3.1-3.5 cover suitability, strengths/limits, settings, objective, student-test scores and fit-gap interpretation. |
| Consolidated comparison table | Present with formatting gap | Numeric comparison tables exist; algorithm family and key configuration appear in per-model sections, rather than in the same consolidated table as requested. Combine these columns for literal format compliance. |
| Required model comparison plots | Present | CV F1, development Top-1/F1/log loss; each model also has actual confusion and generalisation plots. No fabricated epoch histories. |
| Bayesian optimisation and tuned/untuned comparison | Present | Twelve seeded TPE trials; search distributions, identical grouped folds, best parameters, cost and both model evaluations. |
| HMM/Viterbi implementation and required six explanations | Present | Directed candidate states, classifier emissions, routed transitions, recurrence, gaps/fallbacks/disconnections and measured comparisons. |
| Pointwise, sequence and route-continuity comparisons | Present | `decoder_comparison.json/.csv`, report Sections 5-5.1. Pointwise scores explicitly labelled as inferred-reference proxies. |
| Mandatory Folium code, interactive map and PDF map screenshot | Present | `maps.py`, three committed HTML maps and actual screenshot in report Section 6. New Leaflet UI does not remove this Folium deliverable. |
| Ten reserved IDs, real inference and server-side scoring | Present in code; prior HTTP evidence retained | Ten reserved trips succeeded through the frozen predictor. Labels/reference geometry remain server-side. |
| Latest API/UI integration in the live process | Pending restart | Current code defines `/trace/{request_id}` and `/explain/{request_id}/{observation}`, but the running API's OpenAPI schema does not include them. Restart the Linux API before demonstrating Chiranth's trace page. |
| Working UI screenshots matching final implementation | Partial / stale | Existing PDF screenshots show the earlier UI. Capture Chiranth's current map-first UI and model-trace page and update the report. |
| Receiving-laptop terminal/API screenshot | Missing | `reports/evidence/api_terminal.png` is absent. Capture genuine successful request logs/response evidence on Laptop 1, with a request ID matching the Mac UI. JSON/log files do not replace this required screenshot. |
| Physical two-laptop demonstration | Unverified | Chiranth recorded a Mac-local development test. That does not establish Mac-to-Linux operation. Test the actual pair before claiming completion. |
| Latest source correctness | Verified locally | 13 tests pass; one dependency deprecation warning. Frozen prediction and scoring module hashes match. |
| Academic integrity and AI disclosure | Disclosure present; human responsibility remains | AI assistance and limitations disclosed. Team members must review/understand and independently explain the work. An automated audit cannot certify originality or viva readiness. |

## Fix before final demonstration/submission

1. Restart the Linux API with the latest code; check the trace page works after a real match.
2. Run the UI on Chiranth's physical Mac against the Linux LAN address. Capture genuine current UI/trace and receiving-terminal screenshots.
3. Refresh the consolidated technical report with that evidence. Add algorithm-family/key-configuration columns to its overall comparison table. Include the newer report-only reserved comparison if discussing it in the demonstration.
4. Retain a file-size inventory if time permits. Review the report as a team and keep the proxy-label/AI disclosures.

No parameter, model, feature or threshold should change based on reserved results. Chiranth's later all-model reserved comparison is descriptive only and must not replace the earlier grouped-CV model-selection rule.

## Official source coverage

- **Main Guidelines.pdf:** full workflow and two-laptop architecture; core implementation present, physical deployment evidence pending.
- **Dataset Exploration.pdf:** structure, geography, quality, connectivity and leakage requirements covered; per-file size inventory is the minor retained-record gap.
- **Models and Training Guidelines.pdf:** five models, metrics, grouped CV, Bayesian search, HMM and API workflow covered; latest live process needs restart.
- **Deliverables.pdf:** public repo and consolidated report present; terminal screenshot, final UI evidence and literal comparison-table format need completion.
- **Assessment and Plagiarism Policy.pdf:** actual experiment records, freeze and disclosures present; team understanding, originality and live demonstration require human verification.

Source of truth for figures: `artifacts/model_comparison.json`, `tuning.json`, `decoder_comparison.json`, `evaluator_results.json`. Later descriptive comparison: `evaluator_model_comparison.json`. Do not confuse candidate Macro F1, arc Macro F1, route F1 and proxy pointwise accuracy.
