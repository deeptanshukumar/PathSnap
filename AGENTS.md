# PathSnap engineering rules

- Read the five organiser PDFs before changing methodology. Their local location is configured outside application code.
- Deadline: 18 November 2026, 15:00 Asia/Kolkata. Prioritise a reproducible, explainable end-to-end system.
- Work independently within team PT-088. Do not copy another team's code, models, configurations or results.
- Use Python 3.12+, pathlib, small modules, explicit units and seeds. Install only in `.venv`.
- Prediction inputs are `.track`, `.nodes`, `.arcs`. `.route` belongs only to training-label construction and server-side scoring; never to features, candidate generation, decoding or the evaluator UI.
- Train and tune on `train` only, splitting entire trips. `student_test` is development data. Never inspect evaluator data during development or use its results to select anything.
- Route files are ordered arc sequences, not per-observation labels. Always identify inferred pointwise labels and metrics as proxies. Report direct route overlap separately.
- Do not invent metrics, training curves, screenshots or two-device deployment evidence. Run actual experiments and preserve configurations, trials, partitions and timings.
- Keep README.md concise. Update docs/PROJECT_STATE.md after meaningful verified milestones. Record consequential decisions in docs/DECISIONS.md; the user delegated required baseline choices, while new methodological pivots and Bhuvan's novelty require discussion.
- Extension points are candidate features, model factories and decoder configuration. Avoid introducing coordination systems or overlapping context documents.
- Before delivery, test geometry, label separation, grouped folds, Viterbi, API responses, all available demo trips and actual browser behaviour. Genuine two-laptop evidence must be captured on the participating laptops.
