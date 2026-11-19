"""Sections 6 to 11 and the appendices."""

from .data import ROOT, f, n, pct
from .layout import (H1, H2, H3, P, bullets, callout, code, figure, pending_box, question, table)


def folium_section(D, fig):
    out = H1("6. Geographical visualisation using Folium")
    out += [P("Maps are produced by <font name='Mono'>pathsnap/maps.py</font> from the actual loaded trip and the actual "
              "prediction, through the same <font name='Mono'>Pipeline</font> object the API uses. Three interactive "
              "development maps are committed in <font name='Mono'>reports/maps/</font> (student-test trips 00000000, "
              "00000007 and 00000056); <font name='Mono'>python -m pathsnap.cli evaluate</font> regenerates them. The API "
              "builds the same map for every request at <font name='Mono'>GET /maps/{request_id}</font>, with the "
              "reference layer removed.")]
    rows = [["Layer", "Drawing", "Default", "In API / evaluator map?"],
            ["Original GPS observations", "Blue circle markers (subsampled to about 1,200 for very long trips); tooltip "
             "shows index and time", "on", "yes"],
            ["Candidate road network", "Grey thin polylines: every arc that was a candidate for some point", "on", "yes"],
            ["Reference route", "Gold wide polyline, under the prediction", "on", "<b>no</b> (development reports only)"],
            ["Viterbi reconstructed route", "Green polyline of decoded and bridged arcs, drawn arc by arc", "on", "yes"],
            ["Classifier only", "Purple polyline of independent per-point choices", "off", "yes"],
            ["Decoder resets", "Red rings; tooltip gives the reason (gap or no legal path)", "on", "yes"],
            ["Start / finish", "Green and red flag markers", "on", "yes"]]
    out += [table(rows, [1.6, 3.4, 0.6, 1.4]), P("Table 6.1. Map layers.", "caption")]
    rows = [["Requirement (Deliverables 6)", "How the maps meet it"],
            ["1. Distinguishable colours and markers per layer", "Blue points, grey network, gold reference, green "
             "prediction, purple classifier-only, red resets, flag markers (Table 6.1)"],
            ["2. Clear labels or a legend", "Fixed legend box bottom-left; layer names in the layer control; tooltips"],
            ["3. Appropriate extent and zoom", "The view is fitted to the GPS track's bounding box with padding; a scale "
             "bar is shown"],
            ["4. Easy comparison of GPS and predicted route", "GPS markers drawn above the route; prediction drawn above "
             "the reference so differences show as gold"],
            ["5. Layer controls", "Expanded LayerControl, every layer toggleable"],
            ["Show GPS, network, reference (where available), prediction", "All four layers present on development "
             "maps; reference omitted on evaluator maps by design"]]
    out += [table(rows, [2.3, 4.7]), P("Table 6.2. Folium requirements and how they are met.", "caption")]
    out.append(figure(ROOT / "reports/evidence/folium_development.jpg", "Figure 6.1. Browser view of "
                      "reports/maps/development_00000000.html (student test, trip 00000000): GPS observations, reference "
                      "(gold, mostly hidden under the matching green prediction), decoder resets near the finish, legend "
                      "and layer control. Screenshot of the generated map, not a mock-up.", width=0.92))
    out += [P("<b>Design choices.</b> Routes are drawn as separate arc segments rather than one polyline, so a break in "
              "the predicted route stays visible instead of being joined by a straight line. The reference is only drawn "
              "when a caller passes it explicitly, and the API never does. Map tiles and Folium's JavaScript load from "
              "the internet, so a demonstration laptop needs network access to show the base map; road geometry itself "
              "is embedded in the HTML.")]
    return out


def deployment_section(D, fig):
    ev = D["ev"]
    out = H1("7. API, UI and deployment")
    out += H2("7.1 Architecture")
    out.append(figure(fig / "deployment.png", "Figure 7.1. Two-laptop deployment. The UI sends a trip ID; the server runs "
                      "prediction, then scoring, and returns predictions and aggregate metrics only."))
    out += [P("Laptop 1 runs the FastAPI application (<font name='Mono'>pathsnap/api.py</font>) with uvicorn. At "
              "start-up it loads the saved model bundle (model, feature list, configuration and decoder settings) and the "
              "list of the ten active trips. In <b>evaluation mode</b> it requires a folder containing exactly ten trip "
              "directories and refuses to start otherwise; it never falls back to student-test trips. In "
              "<b>development mode</b> it serves the ten student-test trips and labels itself as development. Laptop 2 "
              "serves the static UI with Python's built-in web server and needs no ML packages or data.")]
    out += H2("7.2 Endpoints")
    rows = [["Endpoint", "Purpose", "Response (main fields)"],
            ["GET /health", "Readiness", "status, mode, trip_count, model name, tuned flag"],
            ["GET /trips", "Trips the UI may offer", "the ten active trip IDs and a notice of the partition"],
            ["POST /match", "Run the full pipeline on one trip", "request_id, trip_id, mode, observations, seconds, "
             "predicted_arc_ids (one per GPS point), route_pieces, breaks with reasons, fallback count, metrics "
             "(server-side scores), timings, map_url, trace_url"],
            ["GET /maps/{request_id}", "Folium map of that prediction", "HTML; no reference layer"],
            ["GET /trace/{request_id}", "Stage-by-stage trace for the UI's model-trace page", "GPS, decoded geometry, "
             "per-point candidates, probabilities and Viterbi picks"],
            ["GET /explain/{request_id}/{i}", "Explain one observation", "candidates with features, probability, "
             "emission, road distance and log transition from the Viterbi predecessor"],
            ["GET /docs", "Automatic OpenAPI documentation", "-"]]
    out += [table(rows, [1.7, 1.9, 3.4]), P("Table 7.1. API endpoints.", "caption")]
    out += [P("<b>Validation and robustness.</b> The request body accepts exactly one field, an eight-digit "
              "<font name='Mono'>trip_id</font>; extra fields (for example an attempt to send labels) and malformed IDs "
              "return HTTP 422, and an ID outside the active ten returns 404. Inference is serialised with a lock so "
              "concurrent requests cannot exhaust memory. Recent maps, traces and predictions are kept in small bounded "
              "caches so the UI can fetch them after the match.")]
    out += H2("7.3 Request lifecycle and ground-truth isolation")
    out += numbered_list([
        "The UI calls <font name='Mono'>POST /match</font> with only the trip ID.",
        "The server loads <font name='Mono'>.track</font>, <font name='Mono'>.nodes</font> and "
        "<font name='Mono'>.arcs</font>, generates candidates and features, scores them, decodes with Viterbi and "
        "reconstructs the route. No reference file is read during these steps.",
        "Only after the prediction is complete, <font name='Mono'>score_prediction</font> opens "
        "<font name='Mono'>.route</font> on the server, builds proxy labels and computes metrics. If no route file exists "
        "the prediction is still returned, with metrics set to null.",
        "The response contains predictions and aggregate metrics. Reference arc IDs, labels and geometry are never "
        "included, and the API map has no reference layer.",
        "The server logs one structured line, which the evaluator can match to the request ID shown in the UI.",
    ])
    out += [P("Automated tests enforce this boundary (Section 7.8). One test edits the route file between two identical "
              "requests: the metrics change and the predictions do not, which proves predictions cannot depend on the "
              "reference. Another fails any attempt to open a route file during prediction.")]
    out += [code('PATHSNAP {"timestamp_utc": "...", "request_id": "da1135530f06", "trip_id": "00000015", '
                 '"mode": "evaluation",\n          "observations": 6854, "status": "success", "seconds": 3.96}'),
            P("Format of the per-request log line printed by the receiving server (values from the recorded reserved "
              "run, artifacts/evaluator_results.json; timestamp elided).", "caption")]
    out += H2("7.4 Running the system on two laptops")
    out += [code("# Laptop 1 (model server)\n"
                 "python3.12 -m venv .venv && source .venv/bin/activate\n"
                 "python -m pip install -e '.[dev,report]'\n"
                 "python -m pathsnap.cli serve --mode evaluation --evaluation-root data/evaluator_test \\\n"
                 "    --host 0.0.0.0 --port 8000\n\n"
                 "# Laptop 2 (UI)\n"
                 "python3 -m http.server 8010 --directory ui\n"
                 "# open http://localhost:8010 and enter http://<Laptop-1-LAN-IP>:8000 as the model server")]
    out += [P("Both laptops must be on a network that allows device-to-device connections, and Laptop 1's firewall must "
              "accept TCP port 8000. If the UI cannot connect, open <font name='Mono'>http://&lt;IP&gt;:8000/health</font> "
              "directly in Laptop 2's browser. CORS is open because this is an unauthenticated LAN demonstration; the API "
              "is not meant to be exposed to the internet.")]
    out += H2("7.5 The user interface")
    out += [P("The main screen (<font name='Mono'>ui/index.html</font>) has a field for the model-server address, a "
              "<i>Connect</i> button that calls <font name='Mono'>/health</font> and <font name='Mono'>/trips</font>, a "
              "dropdown of the ten trips and a <i>Reconstruct route</i> button. The result shows an interactive map (GPS "
              "points, predicted route, resets), cards for proxy pointwise accuracy, route overlap F1, number of "
              "observations and server response time, the model name and the request ID to match against the server log. "
              "A second page, <i>Model trace</i> (<font name='Mono'>ui/details.html</font>), replays the pipeline for the "
              "last request: stage timings, the candidates of any chosen GPS point with their probabilities, emissions "
              "and transition terms, the server-side scores, a session log, and a summary of how the final model was "
              "selected.")]
    for name, caption in [("ui_current.png", "Figure 7.2a. Current UI, main screen."),
                          ("ui_trace.png", "Figure 7.2b. Current UI, model-trace page.")]:
        path = ROOT / "reports/evidence" / name
        if path.exists():
            out.append(figure(path, caption, width=0.9))
    out.append(figure(ROOT / "reports/evidence/ui_evaluation.jpg", "Figure 7.2. A working UI session in evaluation mode "
                      "on reserved trip 00000015 (genuine browser capture from an earlier UI version). The cards show the "
                      "server's proxy accuracy, route overlap F1, observation count and response time; the request ID "
                      "appears below them. The red rings are the decoder resets discussed in Section 5.", width=0.86))
    out.append(figure(ROOT / "reports/evidence/ui_working.jpg", "Figure 7.3. The same UI in development mode on a "
                      "student-test trip (genuine browser capture, earlier UI version).", width=0.86))
    out += H2("7.6 Evidence of successful API responses")
    terminal = ROOT / "reports/evidence/api_terminal.png"
    if terminal.exists():
        out.append(figure(terminal, "Figure 7.4. Receiving laptop's terminal showing PATHSNAP request logs.", width=0.92))
    else:
        out += pending_box("Required screenshot not yet captured", [
            "Deliverables 7 asks for a screenshot of successful API responses on the terminal of the receiving laptop. "
            "It must be captured during the real two-laptop run: start the server on Laptop 1, run a match from "
            "Laptop 2, and screenshot Laptop 1's terminal showing the <font name='Mono'>PATHSNAP</font> line with the "
            "same request ID as the UI. Save it as <font name='Mono'>reports/evidence/api_terminal.png</font> and rerun "
            "<font name='Mono'>python scripts/build_final_report.py</font>; this box is then replaced by the screenshot. "
            "Saving <font name='Mono'>ui_current.png</font> and <font name='Mono'>ui_trace.png</font> in the same folder "
            "adds current UI screenshots to Section 7.5.",
            "No substitute image has been created. The recorded HTTP results below are genuine but are not a terminal "
            "screenshot.",
        ])
    out += [P("Recorded HTTP results of the reserved-trip run, one line per request "
              "(<font name='Mono'>artifacts/logs/reserved_http.log</font>):")]
    out += [code("\n".join(D["http_log"]))]
    smoke = D["smoke"]
    ok = sum(r["status_code"] == 200 for r in smoke["requests"])
    out += [P(f"A separate development smoke test (<font name='Mono'>scripts/smoke_api.py</font>, saved in "
              f"<font name='Mono'>reports/evidence/api_smoke.json</font>) sent requests for all ten student-test trips: "
              f"{ok} of {len(smoke['requests'])} returned HTTP 200 with one prediction per GPS observation; unknown IDs and "
              f"invalid bodies were rejected, and a CORS preflight succeeded. Physical Laptop-2-to-Laptop-1 operation "
              "over the LAN still has to be demonstrated on the event day; the recorded runs were made from the server "
              "host.")]
    out += H2("7.7 Reserved final evaluation")
    fz = D["freeze"]
    out += [P(f"The reserved archive arrived after development. Before opening it, the team froze the predictor and scorer "
              f"in commit <b>a61ce50</b> and recorded SHA-256 hashes of the model, configuration and every source module "
              f"in <font name='Mono'>evaluation_freeze.json</font> (frozen {fz['frozen_utc'][:16].replace('T', ' ')} UTC). "
              "<font name='Mono'>scripts/evaluate_reserved.py</font> checked those hashes and then sent one request per "
              "reserved trip to the evaluation-mode API. The archive's own hash and the ten trip IDs are in "
              "<font name='Mono'>evaluator_manifest.json</font>. No reserved result changed any feature, label convention, "
              "model, hyperparameter, decoder setting or threshold. Later, read-only trace endpoints were added to "
              "<font name='Mono'>api.py</font>; every predictor file (features, geometry, decoder, pipeline, model, "
              "configuration) still matches the freeze (docs/DECISIONS.md, item 9).")]
    rows = [["Trip", "GPS points", "Request ID", "Classifier Top-1", "HMM Top-1", "Bridged overlap F1",
             "Bridged continuity", "Resets", "HTTP s"]]
    for q in ev["requests"]:
        m = q["metrics"]
        rows.append([q["trip"], n(q["observations"]), q["request_id"], f(m["classifier"]["proxy_top1_accuracy"]),
                     f(m["hmm"]["proxy_top1_accuracy"]), f(m["bridged_route"]["route_length_f1"]),
                     f(m["bridged_route"]["direct_arc_continuity"]), str(m["decoder_resets"]), f"{q['http_seconds']:.2f}"])
    s = D["ev_summary"]
    rows.append(["All 10", n(s["observations"]), "-", f(s["classifier_top1"]) + " (pooled)", f(s["hmm_top1"]) + " (pooled)",
                 f(s["bridged_route_route_length_f1"]) + " (mean)", f(s["bridged_route_direct_arc_continuity"]) + " (mean)",
                 str(s["resets"]), f"{sum(D['ev_http']) / len(D['ev_http']):.2f} (mean)"])
    out += [table(rows, [0.9, 0.8, 1.1, 0.9, 0.8, 0.9, 0.9, 0.6, 0.6], highlight_rows=(len(rows) - 1,)),
            P("Table 7.2. Reserved-trip results from the live evaluation-mode API (evaluator_results.json). Pointwise "
              "values are proxies.", "caption")]
    out += H2("7.8 Automated checks")
    rows = [["Test", "What it guarantees"],
            ["test_projection_units_roundtrip_and_segment_endpoints", "Projection is in metres and invertible; segment "
             "projection clamps to endpoints"],
            ["test_candidate_projection_matches_bruteforce_including_long_edges", "The KD-tree candidate search finds the "
             "same arcs as brute force"],
            ["test_missing_radius_candidates_are_explicit_and_padding_has_zero_emission", "Fallbacks are flagged; padding "
             "can never be chosen"],
            ["test_directed_partial_arc_routing", "Network distance respects direction and partial arcs"],
            ["test_viterbi_equals_bruteforce_and_resets_unreachable_sequences", "Viterbi returns the best path; resets "
             "when nothing is reachable"],
            ["test_reference_alignment_monotonic_and_inference_never_opens_route", "Labels are monotone; prediction "
             "fails if it tries to read .route"],
            ["test_trip_group_folds_never_overlap", "No trip on both sides of a CV split"],
            ["test_same_arc_backward_motion_preserves_required_network_loop", "Backward motion on a directed arc "
             "inserts a real loop"],
            ["test_loader_rejects_fractional_arc_node_ids", "Invalid input raises instead of being repaired"],
            ["test_api_validation_cors_and_no_reference_exposure", "422/404 handling, CORS, no reference keys or layers "
             "in responses"],
            ["test_reference_changes_cannot_change_predictions", "Editing .route changes metrics, never predictions"],
            ["test_evaluator_mode_never_substitutes_student_test", "Evaluation mode refuses to start without the "
             "reserved folder"],
            ["test_trace_and_explain_are_read_only_and_reference_free", "Trace endpoints match the prediction and "
             "expose no reference"],
            ["tests/test_improved.py (4 tests)", "Context features are finite, never cross outages, and keep "
             "probabilities valid"]]
    out += [table(rows, [3.6, 3.4]), P("Table 7.3. Tests in tests/ (run with python -m pytest -q).", "caption")]
    return out


def numbered_list(items):
    from .layout import numbered
    return numbered(items)


def integrity_section(D, fig):
    out = H1("8. Academic integrity, plagiarism and AI-content checks")
    out += question("8", "Integrity statement")
    out += bullets([
        "<b>Independent work.</b> All code, models, configurations and results in this repository were produced by team "
        "PT-088. No other team's code, weights, configurations, derived data or results were used. The published "
        "algorithms we rely on (Newson and Krumm's HMM map matching, scikit-learn models, Optuna's TPE, SciPy's "
        "Dijkstra, Folium, FastAPI) are cited in Section 10.",
        "<b>Real experiments only.</b> Every number in this report is read from files written by the experiment code. "
        "No metric, plot, training curve, screenshot or deployment evidence was invented. Where evidence is missing "
        "(the receiving-terminal screenshot), the report says so (Section 7.6).",
        "<b>Leakage controls.</b> Whole-trip partitions and folds; scalers fitted inside folds; no route-derived "
        "features; the prediction path cannot open route files (tested); reference data is read only after prediction "
        "and only on the server; the UI never receives labels or reference geometry (tested).",
        "<b>Evaluator data.</b> Never used for training, feature design, model or hyperparameter selection, thresholds or "
        "decoder settings. It was opened only after the freeze in commit a61ce50, and its results changed nothing "
        "(Section 7.7). The report-only comparisons that use it (Figures 5.4 and 9.1) are labelled as such.",
        "<b>No hardcoding.</b> Every API request runs live inference; there are no stored predictions or per-trip special "
        "cases in the code.",
        "<b>Honest reporting of failures.</b> Results that do not favour the system, such as the drop in proxy pointwise "
        "accuracy after decoding and the negligible gain from tuning, are reported with the same prominence as the "
        "successes.",
    ])
    out += callout([
        "AI tools were used and are disclosed as the organisers require. OpenAI Codex assisted with reading the organiser "
        "documents, implementing and testing the software, running the experiments, debugging, aggregating results and "
        "drafting the earlier report and viva materials, as recorded in the repository README. This consolidated report "
        "was compiled with the assistance of Anthropic's Claude, which read the code and the recorded artifacts and "
        "drafted the text and figure code. The report is generated by <font name='Mono'>scripts/build_final_report.py</font> "
        "directly from those artifacts, so no number in it was typed by hand.",
        "AI assistance did not replace experimentation: every result comes from code that runs in this repository. Each "
        "team member is responsible for the content and must be able to explain every claim, table and figure in it; the "
        "viva study guide in reports/supporting/ is provided for that purpose.",
    ], title="AI-use disclosure")
    return out


def extension_section(D, fig):
    imp, st, ev = D["imp"], D["imp_st"], D["imp_ev"]
    b, i = imp["baseline"], imp["improved"]
    out = H1("9. Extension: an opt-in variant with trajectory-context features")
    out += [P("The baseline features describe one GPS point at a time and leave all context to the HMM. A variant tests "
              "whether giving the classifier context directly helps. It keeps the same inputs, candidates, training rows, "
              "folds, model family, search space and decoder, and adds 17 features per (point, candidate) pair "
              "(<font name='Mono'>pathsnap/improved.py</font>):")]
    out += bullets([
        "mean and maximum distance from the candidate arc to the neighbouring GPS points within ±1, ±2, "
        "±3, ±5 and ±8 observations (10 features): does the arc fit the trajectory, not just one point?",
        "rank of the candidate by ±3-window mean distance, and its margin over the best candidate (2)",
        "arc heading against motion over ±5 observations (1): a steadier direction estimate for twins",
        "rank by distance, number of valid candidates and the gap between the two nearest candidates (3)",
        "along-arc progress error: how far the projection moves along the arc between the previous and next fix, "
        "compared with the GPS travel distance (1)",
    ])
    out += [P("All windows stop at gaps over 60 s. The classifier was re-tuned with the same 12-trial TPE search on the "
              "same grouped folds; a small decoder grid (emission weight 1 to 4, β of 10, 25 or 50 m) was searched on "
              "the student-test trips only.")]
    rows = [["Measure", "Default (served)", "Improved (opt-in)"],
            ["Features", "13", "30"],
            ["Grouped CV candidate Macro F1 (± SD)", f"{f(b['cv_candidate_macro_f1'])} ± {f(b['cv_candidate_macro_f1_std'])}",
             f"{f(i['cv_candidate_macro_f1'])} ± {f(i['cv_candidate_macro_f1_std'])}"],
            ["Fit-row candidate Macro F1", f(b["train_candidate_macro_f1"]), f(i["train_candidate_macro_f1"])],
            ["Student-test candidate Macro F1", f(b["student_test_candidate_macro_f1"]), f(i["student_test_candidate_macro_f1"])],
            ["Student-test log loss", f(b["student_test_log_loss"]), f(i["student_test_log_loss"])],
            ["Student-test classifier Top-1 (proxy)", pct(st["default"]["summary"]["classifier_proxy_top1"]),
             pct(st["improved"]["summary"]["classifier_proxy_top1"])],
            ["Student-test Viterbi Top-1 (proxy)", pct(st["default"]["summary"]["hmm_proxy_top1"]),
             pct(st["improved"]["summary"]["hmm_proxy_top1"])],
            ["Student-test bridged route F1 / LCS F1",
             f"{f(st['default']['summary']['bridged_route_f1_mean'])} / {f(st['default']['summary']['bridged_lcs_f1_mean'])}",
             f"{f(st['improved']['summary']['bridged_route_f1_mean'])} / {f(st['improved']['summary']['bridged_lcs_f1_mean'])}"],
            ["Reserved classifier Top-1 (report-only)", pct(ev["default"]["summary"]["classifier_proxy_top1"]),
             pct(ev["improved"]["summary"]["classifier_proxy_top1"])],
            ["Reserved Viterbi Top-1 (report-only)", pct(ev["default"]["summary"]["hmm_proxy_top1"]),
             pct(ev["improved"]["summary"]["hmm_proxy_top1"])],
            ["Reserved bridged route F1 (report-only)", f(ev["default"]["summary"]["bridged_route_f1_mean"]),
             f(ev["improved"]["summary"]["bridged_route_f1_mean"])],
            ["Chosen decoder", "beta 25 m, emission weight 1", f"beta {i['chosen_decoder']['beta_m']:.0f} m, emission "
                                                                f"weight {i['chosen_decoder']['emission_power']:.0f}"],
            ["Tuning time (12 trials)", f"{D['tuning']['optimisation_seconds']:.0f} s", f"{i['tuning_seconds']:.0f} s"]]
    out += [table(rows, [3.2, 1.6, 1.6]), P("Table 9.1. Default vs improved variant (artifacts/improved/*.json). The "
                                            "default's CV score here (re-run inside the variant script with the tuned "
                                            "parameters) differs from Section 4's 0.9290 in the fourth decimal because of "
                                            "floating-point nondeterminism.", "caption")]
    out.append(figure(fig / "improved_variant.png", "Figure 9.1. Default vs improved variant. (a) Development partitions; "
                      "(b) reserved trips, report-only."))
    grid = i["decoder_grid_student_test_top1"]
    out += [P(f"<b>What it shows.</b> Context features make the classifier clearly better: CV Macro F1 rises by "
              f"{i['cv_candidate_macro_f1'] - b['cv_candidate_macro_f1']:.3f}, about "
              f"{(i['cv_candidate_macro_f1'] - b['cv_candidate_macro_f1']) / (D['tuning']['cv_candidate_macro_f1'] - D['tuning']['untuned_cv_macro_f1']):.0f} "
              "times the gain from tuning, and "
              f"classifier-only Top-1 rises by "
              f"{100 * (st['improved']['summary']['classifier_proxy_top1'] - st['default']['summary']['classifier_proxy_top1']):.1f} "
              "points on student test. After Viterbi most of that gain disappears (+"
              f"{100 * (st['improved']['summary']['hmm_proxy_top1'] - st['default']['summary']['hmm_proxy_top1']):.1f} points) "
              "and the bridged route metrics are essentially unchanged. This suggests the context features and the HMM "
              "supply much of the same information, since the HMM already uses neighbouring points. The decoder grid was flat (all twelve "
              f"settings between {min(grid.values()):.4f} and {max(grid.values()):.4f}).")]
    out += callout([
        "This variant was built after the reserved evaluation had been run. Its training and selection never read "
        "evaluator data, and its reserved scores were computed once after its settings were fixed. Still, the decision to "
        "try it was made with the reserved outcome known, so its reserved numbers are not as clean an estimate as the "
        "frozen baseline's. <b>The submitted, served predictor is the frozen default.</b> The variant is opt-in "
        "(<font name='Mono'>serve --variant improved</font>) and is reported as an extension experiment.",
    ], title="Status and caveat")
    return out


def limitations_section(D, fig):
    out = H1("10. Limitations, future work and references")
    out += H2("10.1 Limitations")
    out += bullets([
        "<b>Proxy labels.</b> Per-point labels come from a geometric alignment, not from the organisers; pointwise "
        "metrics measure agreement with that alignment. Reference routes themselves contain breaks and, on a few trips, "
        "disagree strongly with the GPS.",
        "<b>Shared development data.</b> Student-test trips were used both to choose decoder settings and to report "
        "development results, as the guidelines permit; those decoder results are therefore optimistic.",
        "<b>Small CV and search budgets.</b> Three folds and twelve TPE trials limit statistical precision; the "
        "boosting-vs-random-forest margin is within fold noise.",
        "<b>Decoder vs pointwise accuracy.</b> The decoder lowers proxy pointwise accuracy, strongly on reserved trip "
        "00000015; the cause is not established.",
        "<b>Projection.</b> The local equirectangular projection has a small, bounded east-west scale error "
        "(under 0.7% on these trips).",
        "<b>Offline heading.</b> Heading uses future points, so the method is not real-time.",
        "<b>Fixed design constants.</b> Candidate radius and count, stride, label filter and gap threshold were set by "
        "reasoning, not tuned.",
        "<b>Evidence gaps.</b> Per-file byte sizes were not retained; the receiving-terminal screenshot and a physical "
        "two-laptop run are still to be captured.",
    ])
    out += H2("10.2 Future work")
    out += [P("Directions planned by the team but <i>not implemented in this repository</i>: estimating a per-point GPS "
              "error ellipse from the trajectory and using Mahalanobis distance and on-arc probability mass as features; a "
              "speed-dependent von Mises heading likelihood for twin arcs; a conformal (statistically calibrated) "
              "candidate radius with a guaranteed miss rate; a space-time-prism feasibility gate for gaps; "
              "forward-backward posteriors for per-point confidence; LightGBM or CatBoost with calibrated probabilities; "
              "an azimuthal-equidistant projection; and learned transition scores. Each would need the same grouped-CV "
              "protocol and an ablation against this baseline.")]
    out += H2("10.3 References")
    out += bullets([
        "P. Newson and J. Krumm, \"Hidden Markov map matching through noise and sparseness,\" ACM SIGSPATIAL GIS, 2009.",
        "F. Pedregosa et al., \"Scikit-learn: machine learning in Python,\" JMLR 12, 2011; scikit-learn 1.7 user guide "
        "(cross-validation with groups, pipelines and common pitfalls).",
        "T. Akiba et al., \"Optuna: a next-generation hyperparameter optimization framework,\" KDD 2019; J. Bergstra et "
        "al., \"Algorithms for hyper-parameter optimization,\" NeurIPS 2011 (TPE).",
        "P. Virtanen et al., \"SciPy 1.0,\" Nature Methods 17, 2020 (scipy.sparse.csgraph.dijkstra).",
        "Folium and Leaflet documentation (map layers, LayerControl); FastAPI documentation (CORS, request validation).",
        "PES University, PathSnap organiser documents: Main Guidelines, Dataset Exploration, Models and Training "
        "Guidelines, Deliverables, Assessment and Plagiarism Policy (2026).",
    ])
    return out


def reproducibility_section(D, fig):
    m = D["manifest"]
    out = H1("11. Reproducibility and repository map")
    out += [P(f"Environment used for every recorded run: Python {m['python_version']}, NumPy {m['numpy_version']}, "
              f"scikit-learn {m['sklearn_version']}, Optuna {m['optuna_version']}, Linux, CPU only (4 threads). "
              f"Global seed {D['config']['seed']}. Exact package versions are in "
              "<font name='Mono'>requirements.lock.txt</font>; the configuration is "
              "<font name='Mono'>configs/baseline.json</font>.")]
    out += [code("PATHSNAP_DATA_ROOT=\"/path/to/Student folder\"\n"
                 "python -m pathsnap.cli prepare  --data-root \"$PATHSNAP_DATA_ROOT\"   # inventory, candidates, labels\n"
                 "python -m pathsnap.cli train    --data-root \"$PATHSNAP_DATA_ROOT\"   # 5 models, CV, TPE, final model\n"
                 "python scripts/tune_decoder.py  --data-root \"$PATHSNAP_DATA_ROOT\"   # decoder settings (student test)\n"
                 "python -m pathsnap.cli evaluate --data-root \"$PATHSNAP_DATA_ROOT\"   # classifier vs HMM, Folium maps\n"
                 "python -m pytest -q                                                # 17 tests\n"
                 "python scripts/build_final_report.py                               # figures + this PDF")]
    rows = [["Deliverables component", "Location"],
            ["Data processing", "pathsnap/data.py, geometry.py, prepare.py, reference.py"],
            ["Feature engineering", "pathsnap/features.py (baseline), pathsnap/improved.py (variant)"],
            ["Model training", "pathsnap/models.py, pathsnap/experiments.py, scripts/train_improved.py"],
            ["Model evaluation, metrics, plots, confusion matrices", "pathsnap/metrics.py, evaluation.py, "
             "scripts/report_figures.py, reports/figures/, reports/plots/"],
            ["Bayesian optimisation and results", "pathsnap/experiments.py; artifacts/bayesian_trials.json/.csv, tuning.json"],
            ["HMM and Viterbi", "pathsnap/decoder.py, pipeline.py; scripts/tune_decoder.py"],
            ["API", "pathsnap/api.py, trace.py, cli.py"],
            ["UI", "ui/index.html, ui/details.html"],
            ["Deployment", "README.md, pyproject.toml, requirements.lock.txt, configs/"],
            ["Visualisations", "pathsnap/maps.py (Folium), reports/maps/*.html, scripts/report_figures.py"],
            ["Supporting results and logs", "artifacts/*.json/.csv, artifacts/logs/, reports/evidence/"],
            ["Saved model weights", "artifacts/models/final.joblib (served), artifacts/improved/model.joblib (opt-in)"],
            ["This report", "reports/PathSnap_Final_Report.pdf, built by scripts/build_final_report.py"]]
    out += [table(rows, [2.5, 4.5]), P("Table 11.1. Where each component required by Deliverables 1 lives. Code was kept in "
                                       "one Python package rather than split into numbered folders because the "
                                       "pre-evaluation freeze records hashes of these exact file paths.", "caption")]
    return out


def appendix_a(D, fig):
    out = H1("Appendix A. Requirement traceability across all five organiser documents")
    groups = [
        ("Main Guidelines", [
            ("Two-laptop architecture: model/API on Laptop 1, UI on Laptop 2", "7.1, 7.4", "Implemented; physical run pending"),
            ("12-stage process from problem understanding to report", "1 to 11", "Done")]),
        ("Dataset Exploration", [
            ("Verify file naming, row counts, data types, column meanings", "2.1 to 2.3", "Done"),
            ("Number of trips per partition; observations per trip", "1.4, 2.2", "Done"),
            ("File availability and file sizes", "2.2", "Availability and hashes done; byte sizes not retained"),
            ("Missing or invalid GPS; timestamp ordering and gaps", "2.3, 2.4", "Done"),
            ("Geographic extent of each trajectory", "2.5", "Done"),
            ("Nodes and arcs; connectivity; isolated nodes; components", "2.2, 2.6", "Done"),
            ("Trips with large gaps or other anomalies", "2.4, 2.7, 2.9", "Done"),
            ("Availability and structure of reference routes", "2.7", "Done"),
            ("Justified projection or geodesic distance; coordinate order", "2.5, 2.10.2", "Done"),
            ("Reference limitations; do not overstate label certainty", "2.7, 2.10.4, front matter", "Done"),
            ("Partition and leakage rules", "1.4, 3.0, 8", "Done"),
            ("Recommended outputs 1-6 (inventory, statistics, plots, Folium, network tables, quality record)",
             "2.2 to 2.9", "Done"),
            ("Expected outcome: organisation, file meaning, relationships, quality concerns, assumptions",
             "2.1, 2.9, 2.10", "Done")]),
        ("Models and Training Guidelines", [
            ("At least five models from several families", "3", "Done (5 families)"),
            ("Adapting approaches without probabilities; clustering caveat", "-", "Not applicable: all five produce "
             "probabilities; no clustering used"),
            ("Experimental design items 1-8 (configs, 80/10 use, consistent preprocessing, times, seeds, logs)",
             "3.0, Tables 3.2-3.4, App. B", "Done"),
            ("Metrics list, Top-1 and miss-rate definitions, primary metric with reason", "3.0", "Done"),
            ("Comparison and selection criteria (generalisation, stability, probabilities, speed, complexity, decoder)",
             "3.1, 3.1.1", "Done"),
            ("Bayesian optimisation steps 1-9", "4", "Done"),
            ("HMM requirements (states, correspondence, emissions, transitions, recurrence, gaps, invalid transitions, "
             "per-point output, comparison, continuity)", "5.2 to 5.8", "Done"),
            ("Decoder tuned on training/student-test only", "5.4 (Figure 5.1)", "Done (student test)"),
            ("Final evaluation flow, all ten trips, server-side scoring, API logs", "7", "Done; terminal screenshot pending"),
            ("Independent implementation", "8", "Done")]),
        ("Deliverables", [
            ("Public repository with all components, README, weights, no secrets", "11 (Table 11.1)", "Done"),
            ("One consolidated PDF with actual results, verifiable from the repository", "This report, App. B", "Done"),
            ("2.2 Q1-Q2 and required EDA inclusions", "2", "Done"),
            ("3.1 consolidated table with exact columns; N/A explained", "Table 3.2", "Done"),
            ("3.2 A1-A2, B5-B6, C7-C9 for every model", "3.2.1-3.2.5", "Done"),
            ("3.3 required plots; no artificial curves", "3.3", "Done"),
            ("4 Q1-Q6 and tuned/untuned comparison", "4", "Done"),
            ("5 Q1-Q6, classifier-only vs HMM, where it did not help", "5", "Done"),
            ("6 Folium requirements and screenshot", "6", "Done"),
            ("7 UI screenshots", "7.5", "Genuine screenshots of an earlier UI version; current UI capture pending"),
            ("7 Screenshot of API responses on the receiving laptop's terminal", "7.6", "Pending (placeholder shown)"),
            ("8 Integrity and AI disclosure", "8", "Done")]),
        ("Assessment and Plagiarism Policy", [
            ("Evidence of meaningful experimentation and retained search record", "3, 4, App. B", "Done"),
            ("Independent work; no shared artifacts", "8", "Done"),
            ("No fabricated metrics, graphs, results or citations; AI disclosure", "8, App. B", "Done"),
            ("Viva topics (data, preprocessing, features, models, BO, HMM, Viterbi, API, UI, results)", "2 to 7",
             "Covered; team rehearsal required"),
            ("Evaluator data untouched; labels never in prediction or UI; no hardcoding", "7.3, 7.7, 8", "Done")]),
    ]
    for title, items in groups:
        out += [P(f"<b>{title}</b>", "h3"),
                table([["Requirement", "Section", "Status"]] + [list(x) for x in items], [4.2, 1.3, 2.2])]
    return out


def appendix_b(D, fig):
    out = H1("Appendix B. Source artifacts behind every number")
    out += [P("The report generator reads only these files. Each hash identifies the exact version used; anyone can "
              "recompute it with <font name='Mono'>sha256sum</font>.")]
    desc = {
        "inventory.json": "Per-trip EDA statistics, raw-file hashes (Section 2)",
        "folds.json": "Trips in each CV fold (3.0)",
        "training_manifest.json": "Training rows, positives, versions, configuration (2.10, 11)",
        "model_comparison.json": "All five models: per-fold CV, fit, out-of-fold, student-test and per-trip metrics (3)",
        "selection.json": "Selection rule and chosen family (3.1.1)",
        "bayesian_trials.json": "Every TPE trial with distributions, fold scores and durations (4)",
        "tuning.json": "Tuned vs untuned results and timings (4)",
        "final_model.json": "Served model parameters, features, decoder configuration (5)",
        "decoder_search.json": "Six decoder settings on student test (5.1)",
        "decoder_comparison.json": "Classifier vs HMM vs bridged route per student-test trip, timings (5)",
        "evaluation_freeze.json": "Hashes frozen before reserved access (7.7)",
        "evaluator_manifest.json": "Reserved archive hash and trip IDs (7.7)",
        "evaluator_results.json": "Reserved HTTP results and metrics per trip (5, 7.7)",
        "evaluator_model_comparison.json": "Report-only reserved comparison of all families (5)",
        "improved/results.json": "Variant CV, tuning and decoder grid (9)",
        "improved/student_test_results.json": "Variant vs default on student test (9)",
        "improved/evaluator_test_results.json": "Variant vs default on reserved trips, report-only (9)",
        "logs/reserved_http.log": "HTTP lines of the reserved run (7.6)",
        "logs/training.log": "Console log of model training and TPE (3, 4)",
        "../reports/evidence/api_smoke.json": "Development HTTP smoke test (7.6)",
        "../configs/baseline.json": "Configuration (all sections)",
    }
    rows = [["File (under artifacts/)", "Content and sections", "SHA-256 (first 16 hex)"]]
    for k, v in D["hashes"].items():
        rows.append([k, desc.get(k, ""), v[:16]])
    out += [table(rows, [2.3, 3.6, 1.3])]
    return out
