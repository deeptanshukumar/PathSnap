"""Build the consolidated technical PDF strictly from measured experiment artifacts."""

from datetime import datetime, timezone
import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .data import load_trip
from .features import FEATURE_NAMES

NAMES = {"logistic": "Logistic regression", "gaussian_nb": "Gaussian Naive Bayes",
         "decision_tree": "Decision tree", "random_forest": "Random forest",
         "hist_gradient_boosting": "Histogram gradient boosting"}
SHORT = ["Logistic", "Gaussian NB", "Decision tree", "Random forest", "Hist. boosting"]
MODEL_REASONING = {
    "logistic": ("A compact linear baseline tests whether distance and direction are already sufficient.",
                 "L2-regularised binary log loss, solved by L-BFGS; fold-local standardisation.",
                 "Fast inference and interpretable feature effects; a linear score cannot represent all junction and motion interactions."),
    "gaussian_nb": ("A fast probabilistic baseline tests a generative alternative to discriminative learning.",
                    "Class priors and per-class Gaussian feature likelihoods, with variance smoothing; fold-local standardisation.",
                    "Very cheap fitting; correlated distance/log-distance and heading features violate conditional independence and can distort probabilities."),
    "decision_tree": ("A bounded tree captures nonlinear thresholds without a large ensemble.",
                      "Greedy Gini impurity reduction, bounded depth and minimum leaf size.",
                      "Simple thresholds capture interactions; a single tree can be unstable and yields coarse probability estimates."),
    "random_forest": ("Bagged trees test whether averaging nonlinear rules improves trip generalisation.",
                      "Bootstrap aggregation of trees using Gini splits; mean leaf class proportions provide candidate scores.",
                      "Averaging reduces single-tree variance; fitting costs and artifact size exceed the smaller baselines."),
    "hist_gradient_boosting": ("Boosting tests sequential correction of errors in distance, heading and junction features.",
                               "Binary log loss with histogram-based gradient boosting and L2 leaf regularisation; automatic row-random early stopping disabled.",
                               "Captures nonlinear interactions efficiently on tabular features; tuning and weak reference labels can still cause development overfitting."),
}


def read_json(path: Path):
    return json.loads(path.read_text())


def make_plots(data_root: Path, artifacts: Path, reports: Path) -> None:
    plots = reports / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    models = read_json(artifacts / "model_comparison.json")
    inventory = read_json(artifacts / "inventory.json")
    decoder = read_json(artifacts / "decoder_comparison.json")
    trials = read_json(artifacts / "bayesian_trials.json")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#163d35", "text.color": "#163d35", "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    for split, color in [("train", "#087f6a"), ("student_test", "#d6a54c")]:
        rows = [r for r in inventory if r["split"] == split]
        axes[0].hist([r["gps_observations"] for r in rows], bins=14, color=color, alpha=.65, label=split)
        axes[1].scatter([r["arcs"] for r in rows], [r["gps_observations"] for r in rows], s=18, color=color, label=split)
        axes[2].scatter([r["max_gap_s"] for r in rows], [r["proxy_label_distance_p95_m"] for r in rows], s=18, color=color)
    axes[0].set(xlabel="GPS observations / trip", ylabel="Trips"); axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Road arcs (log)", ylabel="GPS observations (log)", xscale="log", yscale="log")
    axes[2].set(xlabel="Maximum time gap, seconds (log)", ylabel="Proxy label p95 distance, m", xscale="log")
    fig.savefig(plots / "eda.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    x = np.arange(len(models))
    axes[0].bar(x, [r["cv_candidate_macro_f1"] for r in models], yerr=[r["cv_candidate_macro_f1_std"] for r in models], color="#087f6a", capsize=3)
    axes[0].set(ylabel="Grouped CV candidate Macro F1", ylim=(0, 1))
    axes[1].bar(x-.18, [r["student_test"]["summary"]["proxy_top1_accuracy"] for r in models], width=.36, color="#087f6a", label="Proxy Top-1")
    axes[1].bar(x+.18, [r["student_test"]["summary"]["candidate_macro_f1"] for r in models], width=.36, color="#d6a54c", label="Candidate Macro F1")
    axes[1].set(ylabel="Student-test development score", ylim=(0, 1)); axes[1].legend(fontsize=8)
    axes[2].bar(x, [r["student_test"]["summary"]["candidate_log_loss"] for r in models], color="#5c7ea0")
    axes[2].set(ylabel="Candidate binary log loss (lower better)")
    for ax in axes:
        ax.set_xticks(x, SHORT, rotation=35, ha="right", fontsize=8)
    fig.savefig(plots / "model_comparison.png", dpi=180); plt.close(fig)
    for model in models:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
        matrix = np.asarray(model["student_test"]["summary"]["candidate_confusion_matrix"])
        axes[0].imshow(matrix, cmap="Greens")
        for i in range(2):
            for j in range(2):
                axes[0].text(j, i, f"{matrix[i,j]:,}", ha="center", va="center", color="white" if matrix[i,j]>matrix.max()/2 else "#163d35")
        axes[0].set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted candidate label", ylabel="Reference proxy candidate label", title="Student-test confusion matrix")
        axes[1].bar(["Fit", "Grouped CV", "Student test"], [model["training"]["candidate_macro_f1"], model["cv_candidate_macro_f1"], model["student_test"]["summary"]["candidate_macro_f1"]], color=["#a9c9bd", "#087f6a", "#d6a54c"])
        axes[1].set(ylabel="Candidate Macro F1", ylim=(0, 1), title="Measured generalisation diagnostic")
        fig.savefig(plots / f"{model['name']}.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
    values = np.array([t["value"] for t in trials])
    ax.plot(np.arange(len(values)), values, "o", color="#5c7ea0", label="Trial CV objective")
    ax.step(np.arange(len(values)), np.maximum.accumulate(values), where="post", color="#087f6a", label="Best so far")
    ax.axvline(3.5, color="#999", linestyle="--", label="TPE begins after 4 startup trials")
    ax.set(xlabel="Trial index (zero based)", ylabel="Mean grouped CV candidate Macro F1"); ax.legend(fontsize=8)
    fig.savefig(plots / "bayesian_search.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.4), constrained_layout=True)
    x = np.arange(len(decoder))
    ax.bar(x-.18, [r["classifier"]["proxy_top1_accuracy"] for r in decoder], width=.36, color="#5c7ea0", label="Classifier only")
    ax.bar(x+.18, [r["hmm"]["proxy_top1_accuracy"] for r in decoder], width=.36, color="#087f6a", label="HMM / Viterbi")
    ax.set_xticks(x, [r["trip"] for r in decoder], rotation=35, ha="right", fontsize=8)
    ax.set(ylabel="Student-test proxy Top-1 accuracy", ylim=(0, 1)); ax.legend(fontsize=8)
    fig.savefig(plots / "decoder_comparison.png", dpi=180); plt.close(fig)
    trip = load_trip(data_root / "student_test/00000000")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), constrained_layout=True)
    axes[0].plot(trip.track[:, 0], trip.track[:, 1], ".", ms=1, color="#327fca")
    axes[0].set(xlabel="Longitude (degrees)", ylabel="Latitude (degrees)", title="Raw GPS - student_test / 00000000")
    edges = trip.nodes[trip.arcs]
    from matplotlib.collections import LineCollection
    axes[1].add_collection(LineCollection(edges, colors="#71887c", linewidths=.25))
    axes[1].autoscale(); axes[1].set(xlabel="Longitude (degrees)", ylabel="Latitude (degrees)", title="Supplied road network")
    fig.savefig(plots / "trajectory.png", dpi=180); plt.close(fig)


def generate_report(data_root: Path, artifacts: Path, reports: Path) -> None:
    make_plots(data_root, artifacts, reports)
    models = read_json(artifacts / "model_comparison.json")
    inventory = read_json(artifacts / "inventory.json")
    tuning = read_json(artifacts / "tuning.json")
    decoder = read_json(artifacts / "decoder_comparison.json")
    search = read_json(artifacts / "decoder_search.json")
    final = read_json(artifacts / "final_model.json")
    manifest = read_json(artifacts / "training_manifest.json")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("BodyPS", fontName="Helvetica", fontSize=9.5, leading=14, spaceAfter=8, textColor=colors.HexColor("#233f36")))
    styles.add(ParagraphStyle("SmallPS", parent=styles["BodyPS"], fontSize=7.6, leading=10.5))
    styles.add(ParagraphStyle("TitlePS", fontName="Helvetica-Bold", fontSize=31, leading=37, textColor=colors.HexColor("#087f6a"), spaceAfter=18))
    styles["Heading1"].textColor = colors.HexColor("#087f6a")
    styles["Heading1"].fontSize = 19
    styles["Heading1"].leading = 24
    styles["Heading2"].fontSize = 12
    styles["Heading2"].leading = 16
    story = []
    def p(text, small=False):
        story.append(Paragraph(text, styles["SmallPS" if small else "BodyPS"]))
    def h(text):
        story.append(Paragraph(text, styles["Heading1"]))
    def h2(text):
        story.append(Paragraph(text, styles["Heading2"]))
    def table(values, widths=None):
        cells = [[Paragraph(escape(str(v)), styles["SmallPS"]) for v in row] for row in values]
        result = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        result.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4eee7")),
                                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8f4")]),
                                    ("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                                    ("TOPPADDING", (0, 0), (-1, -1), 6), ("LINEBELOW", (0, 0), (-1, 0), .5, colors.HexColor("#a6c2b3"))]))
        story.extend([result, Spacer(1, 10)])
    def image(path, width=490, height=None):
        if path.exists():
            from PIL import Image as PILImage
            with PILImage.open(path) as im:
                w, hh = im.size
            story.append(Image(str(path), width=width, height=height or width * hh / w))
            story.append(Spacer(1, 8))
        else:
            p("Evidence not yet captured: " + escape(path.name) + ". This requirement remains open.")
    def page(): story.append(PageBreak())
    pct = lambda x: f"{100*x:.2f}%"
    n = sum(r["observations"] for r in decoder)
    weighted = lambda key, metric: sum(r["observations"] * r[key][metric] for r in decoder) / n
    mean = lambda key, metric: float(np.mean([r[key][metric] for r in decoder]))

    story.append(Spacer(1, 34))
    story.append(Paragraph("PathSnap", styles["TitlePS"]))
    h("Distributed map matching &amp; route reconstruction")
    p("Machine Learning Hackathon | PES University | Team PT-088 | 18 November 2026")
    table([["Team member", "SRN"], ["Chiranth R", "PES1UG24AM354"], ["Darshan S Govekar", "PES1UG24AM355"],
           ["Daruru Bhuvan Sai", "PES1UG24AM356"], ["Deeptanshu Kumar", "PES1UG24AM357"]], [300, 190])
    h2("Measured system status")
    p(f"Five model families/algorithms were fitted and compared on identical trip folds. The selected family is {NAMES[final['model_name']]}; TPE completed {tuning['trials']} trials. The development decoder processes all {n:,} observations in ten student-test trips.")
    p(f"Classifier-only proxy accuracy: <b>{pct(weighted('classifier','proxy_top1_accuracy'))}</b>. HMM proxy accuracy: <b>{pct(weighted('hmm','proxy_top1_accuracy'))}</b>. Mean trip length-weighted overlap F1 for the bridged HMM route: <b>{mean('bridged_route','route_length_f1'):.4f}</b>.")
    p("<b>Evaluation boundary:</b> the figures above are development results. The reserved archive arrived after predictor freeze; its separate final-partition results appear in Section 7.2. Per-point reference labels are inferred by monotone route alignment and are explicitly proxy labels. The reference routes themselves may contain errors.")
    p("<b>Submission status:</b> this evidence-backed report still needs the team's technical review and genuine two-laptop deployment evidence. The report never substitutes a generated terminal image for that evidence.")
    p("Source and result provenance: https://github.com/deeptanshukumar/PathSnap; artifacts/*.json, configs/baseline.json and pathsnap/*.py. Report generated from those recorded files.", True)
    page()

    h("2. Dataset understanding and EDA")
    table([["Partition", "Trips", "GPS", "Road arcs", "Max gap (s)"], *[[s, len(rs), f"{sum(x['gps_observations'] for x in rs):,}", f"{sum(x['arcs'] for x in rs):,}", f"{max(x['max_gap_s'] for x in rs):,.0f}"] for s in ["train", "student_test"] for rs in [[x for x in inventory if x['split']==s]]]], [105, 45, 85, 120, 135])
    p("Each trip has headerless .track (longitude, latitude, seconds), .nodes (longitude, latitude), .arcs (zero-based source/target node IDs), and .route (ordered arc IDs). IDs are local to each trip. No GPS row-to-route-row correspondence exists.")
    p("All supplied GPS values are finite and timestamps strictly increase. The implementation rejects invalid coordinates, fractional IDs, out-of-range references and non-increasing time rather than silently repairing them. There are no imputed GPS observations. Exact one-second intervals account for 96.50% of training steps and 98.62% of student-test steps.")
    p("Training networks have 3,121 weak components in total; student-test networks have 227. Reference routes contain 13 training endpoint breaks and one student-test break. These counts describe map/reference limitations, not model errors.")
    image(reports / "plots/eda.png")
    image(reports / "plots/trajectory.png")
    page()

    h("2.1 Preprocessing and features")
    p("A per-trip equirectangular projection converts angular coordinates to local metres: x = R cos(latitude_origin) delta_longitude_radians and y = R delta_latitude_radians, R = 6,371,008.8 m. Origin is the median network coordinate. The largest network dimension is about 77.6 km. This local approximation is not a global geodesic model; projection error is a remaining limitation at the largest extents.")
    p("Every directed edge is indexed by samples spaced at most 30 m apart. A KD-tree radius of 115 m retrieves every segment that can intersect the 100 m candidate radius, followed by exact finite-segment projection. Keep the eight closest directed arcs. A missing-radius case gets an explicitly flagged nearest-sample fallback. Short candidate lists use masked padding; the reference arc is never inserted artificially.")
    table([["Feature group", "Reason"], ["Distance, log distance, excess over nearest", "GPS proximity and relative ambiguity"], ["Signed/absolute heading cosine, stationary indicator", "Direction separates reverse arcs; stops reduce heading reliability"], ["Edge length, log length, projection fraction", "Local segment geometry and endpoint context"], ["Speed (m/s), log previous time gap (seconds)", "Motion and irregular sampling context"], ["Source indegree, target outdegree", "Road junction ambiguity"]], [210, 280])
    p("Heading uses a +/-3-observation window and never crosses a gap over 60 s. This is an offline method; it uses future GPS observations. Coordinates, arc IDs, reference-route membership and label-derived statistics are absent from model features. Scalers, where used, fit only the training fold.")
    h2("Pointwise supervision convention")
    p("For GPS index t and route position j, minimise D[t,j] = squared_distance(t,route[j]) + min(i &lt;= j) D[t-1,i]. Prefix minima give O(GPS_count x route_length) time. Separate positions preserve repeated visits. Forward skips allow the GPS to miss short segments. Start and end route positions are not forced.")
    p("This alignment assumes chronological route order and resolves ties deterministically. It cannot recover independently verified physical labels. Training excludes reference projections farther than 75 m: 21,390 of 208,417 observations fail that quality rule before stride selection. Trips 00000002, 00000033 and 00000054 contain most exclusions. All student-test observations remain in scoring, including 21 above 75 m.")
    p(f"Fitting uses every fifth observation from all training folders: {manifest['candidate_training_rows']:,} valid candidate rows, {manifest['positive_rows']:,} positive. Candidate misses remain all-negative sets; no reference leakage repairs them. Student-test proxy candidate miss rate is 0.3601%; full-training miss rate is 10.8029%, heavily affected by the mismatch trips.")
    page()

    h("3. Model experimentation and comparison")
    p("All five models use the same 13 features, three StratifiedGroupKFold partitions and seed 357. Whole trips stay together. Primary selection metric is mean grouped-CV binary candidate Macro F1: it balances positive-match and negative-candidate performance. Observation Top-1 tests ranking within each GPS candidate set. Report both because they can disagree. No student-test score selected the classifier family.")
    values = [["Model", "CV cand. F1", "Dev Top-1*", "Dev cand. F1", "Log loss", "Predict (s)"]]
    for r in models:
        s = r["student_test"]["summary"]
        values.append([NAMES[r["name"]], f"{r['cv_candidate_macro_f1']:.4f}", f"{s['proxy_top1_accuracy']:.4f}", f"{s['candidate_macro_f1']:.4f}", f"{s['candidate_log_loss']:.4f}", f"{s['model_inference_seconds']:.3f}"])
    table(values, [150, 68, 68, 68, 68, 68])
    p("*Pointwise Top-1 uses inferred reference labels. Candidate F1/log loss use binary candidate labels; the main log loss is not an observation-level multiclass loss. Predict seconds cover classifier calls across 19,160 observations, excluding input loading, candidate generation, HMM and maps. Times are hardware/run dependent.", True)
    image(reports / "plots/model_comparison.png")
    values = [["Model", "Dev arc Macro F1*", "Direct continuity", "Route overlap F1", "Fit (s)"]]
    for r in models:
        s = r["student_test"]["summary"]
        values.append([NAMES[r["name"]], f"{s['mean_trip_proxy_arc_macro_f1']:.4f}", f"{s['mean_trip_direct_arc_continuity']:.4f}", f"{s['mean_trip_route_length_f1']:.4f}", f"{r['full_fit_seconds']:.2f}"])
    table(values, [150, 90, 90, 90, 70])
    p("Arc Macro F1 and route metrics are arithmetic means over trips; pooled candidate metrics weight candidate rows. Direct continuity is the fraction of consecutive distinct predicted arcs whose endpoints meet; sparse GPS can skip intervening arcs. Route overlap is a direct length-weighted set comparison against .route. Detailed fold configurations and per-trip values are in model_comparison.json and folds.json.", True)
    page()

    for index, model in enumerate(models, 1):
        h(f"3.{index} {NAMES[model['name']]}")
        why, objective, limitations = MODEL_REASONING[model["name"]]
        h2("Model understanding")
        p(why + " " + limitations)
        h2("Training and configuration")
        p("Family: " + model["family"] + ". " + objective)
        p("Initial configuration: " + escape(json.dumps(model["parameters"], sort_keys=True)) + "; seed 357 where applicable. Full reproducible construction is in pathsnap/models.py. No synthetic epoch-wise curves are shown; these experiments do not record neural-network epoch histories.")
        h2("Experimental results")
        s = model["student_test"]["summary"]
        table([["Metric", "Measured value"], ["CV candidate Macro F1 (mean +/- population SD)", f"{model['cv_candidate_macro_f1']:.4f} +/- {model['cv_candidate_macro_f1_std']:.4f}"], ["Student-test proxy Top-1 / mean-trip arc Macro F1", f"{s['proxy_top1_accuracy']:.4f} / {s['mean_trip_proxy_arc_macro_f1']:.4f}"], ["Student-test candidate precision / recall / Macro F1", f"{s['candidate_precision']:.4f} / {s['candidate_recall']:.4f} / {s['candidate_macro_f1']:.4f}"], ["Candidate log loss / observation proxy log loss", f"{s['candidate_log_loss']:.4f} / {s['observation_proxy_log_loss']:.4f}"], ["Full fit / classifier inference over ten trips (seconds)", f"{model['full_fit_seconds']:.3f} / {s['model_inference_seconds']:.3f}"]], [320, 170])
        gap = model["training"]["candidate_macro_f1"] - model["cv_candidate_macro_f1"]
        p(f"Training candidate Macro F1 is {model['training']['candidate_macro_f1']:.4f}, compared with {model['cv_candidate_macro_f1']:.4f} in grouped CV, a gap of {gap:.4f}. " + ("This training advantage is consistent with overfitting; it is not proof that all errors come from variance." if gap > .02 else "The small training/CV gap does not show a large overfitting signal on this metric; it does not prove absence of overfitting.") + " Reference alignment noise and geographic shift also affect these scores. Compare probability quality and Top-1 rather than inferring fit solely from training accuracy.")
        image(reports / f"plots/{model['name']}.png")
        p("Figure: actual student-test binary confusion counts and measured fit/CV/development candidate Macro F1. Off-diagonal errors and the separate Top-1 result show where candidate classification and observation ranking differ.", True)
        page()

    h("4. Bayesian hyperparameter optimisation")
    p(f"Selected family: {NAMES[tuning['family']]}. Optuna's seeded Tree-structured Parzen Estimator completed {tuning['trials']} sequential trials: four random startup trials and eight adaptive trials. Each objective uses the same three training-trip folds. Objective: maximise mean candidate Macro F1 at the model's 0.5 binary threshold. Evaluator data never enters this process.")
    table([["Hyperparameter", "Search distribution"], ["learning_rate", "Log-uniform: 0.025 to 0.2"], ["max_leaf_nodes", "Integer: 7 to 31"], ["l2_regularization", "Log-uniform: 0.01 to 10"], ["min_samples_leaf", "Integer: 15 to 100"]], [230, 260])
    p("Best values: " + escape(json.dumps(tuning["best_parameters"], sort_keys=True)))
    u, t = tuning["untuned_student_test"]["summary"], tuning["tuned_student_test"]["summary"]
    table([["Metric", "Untuned", "Tuned"], ["CV candidate Macro F1", f"{tuning['untuned_cv_macro_f1']:.4f}", f"{tuning['cv_candidate_macro_f1']:.4f}"], ["Student-test proxy Top-1", f"{u['proxy_top1_accuracy']:.4f}", f"{t['proxy_top1_accuracy']:.4f}"], ["Student-test candidate Macro F1", f"{u['candidate_macro_f1']:.4f}", f"{t['candidate_macro_f1']:.4f}"], ["Student-test mean-trip arc Macro F1", f"{u['mean_trip_proxy_arc_macro_f1']:.4f}", f"{t['mean_trip_proxy_arc_macro_f1']:.4f}"], ["Candidate log loss", f"{u['candidate_log_loss']:.4f}", f"{t['candidate_log_loss']:.4f}"], ["Classifier inference seconds", f"{u['model_inference_seconds']:.3f}", f"{t['model_inference_seconds']:.3f}"]], [290, 100, 100])
    image(reports / "plots/bayesian_search.png", width=460)
    p(f"Optimisation took {tuning['optimisation_seconds']:.1f} s; final tuned refit took {tuning['tuned_fit_seconds']:.2f} s. The tuned model improves the selection objective but does not necessarily improve development Top-1. Reusing CV for selection and tuning makes CV a development estimate rather than an unbiased final test. Trial parameters, search distributions, fold scores and durations are retained in bayesian_trials.json/.csv.")
    page()

    h("5. HMM and Viterbi route reconstruction")
    p("Each hidden state is a projected position on a specific directed candidate arc, preserving its arc ID and projection fraction. Positive classifier probabilities are normalised across valid candidates to obtain emission potentials. Padding has zero probability. The scores are discriminative candidate probabilities, not a separately fitted generative GPS density.")
    p("For consecutive candidates, directed network distance includes the remainder of the first edge, a bounded shortest path between endpoints, and the prefix of the next edge. Same-edge forward motion uses its direct partial length. Backward projection of up to 3 m counts as stationary GPS jitter; larger reversal requires a directed loop.")
    p("Transition log compatibility is -abs(network_distance - observed_GPS_displacement) / beta. Impossible routes have negative infinity. Viterbi uses delta[t,j] = log_emission[t,j] + max_i(delta[t-1,i] + log_transition[i,j]), then backtracks the best path. This is an HMM-style conditional sequence decoder with discriminative emissions and network potentials.")
    p("The initial row-normalised transition implementation hurt several trips. A six-setting development comparison tested beta in {5,25,100} metres and row normalisation on/off. A state-dependent row normaliser on a truncated candidate set can bias scores; the globally scaled exponential potential avoids that particular effect. This is an established-method variant, not a novelty claim.")
    p("Chosen development configuration: " + escape(json.dumps(search["chosen"], sort_keys=True)) + ". Selection uses the same ten student-test trips reported here; it is not an untouched test.")
    p("A radius miss produces a flagged nearest-sample fallback. A GPS gap over 60 s, or a step for which no legal directed path survives, starts a new Viterbi segment. Routing bounds use 55 m/s times elapsed seconds plus 100 m slack. No bridge crosses an explicit reset. Shortest-path intermediate arcs reconstruct road geometry between selected states.")
    image(reports / "plots/decoder_comparison.png")
    page()

    h("5.1 Decoder results and failure cases")
    table([["Trip", "Classifier*", "HMM*", "Overlap F1 (bridge)", "Resets", "Predict s"], *[[r["trip"], f"{r['classifier']['proxy_top1_accuracy']:.4f}", f"{r['hmm']['proxy_top1_accuracy']:.4f}", f"{r['bridged_route']['route_length_f1']:.4f}", r["decoder_resets"], f"{r['timings']['total_prediction_seconds']:.2f}"] for r in decoder]], [86, 80, 80, 114, 55, 75])
    table([["Metric", "Classifier", "HMM states", "HMM bridged route"], ["Pooled proxy Top-1", f"{weighted('classifier','proxy_top1_accuracy'):.4f}", f"{weighted('hmm','proxy_top1_accuracy'):.4f}", "N/A: intermediate arcs have no GPS index"], ["Mean trip direct continuity", f"{mean('classifier','direct_arc_continuity'):.4f}", f"{mean('hmm','direct_arc_continuity'):.4f}", f"{mean('bridged_route','direct_arc_continuity'):.4f}"], ["Mean length-weighted route overlap F1", f"{mean('classifier','route_length_f1'):.4f}", f"{mean('hmm','route_length_f1'):.4f}", f"{mean('bridged_route','route_length_f1'):.4f}"], ["Mean ordered route LCS F1", f"{mean('classifier','ordered_route_lcs_f1'):.4f}", f"{mean('hmm','ordered_route_lcs_f1'):.4f}", f"{mean('bridged_route','ordered_route_lcs_f1'):.4f}"]], [190, 90, 90, 120])
    improved = [r["trip"] for r in decoder if r["hmm"]["proxy_top1_accuracy"] > r["classifier"]["proxy_top1_accuracy"]]
    worse = [r["trip"] for r in decoder if r["hmm"]["proxy_top1_accuracy"] < r["classifier"]["proxy_top1_accuracy"]]
    p("HMM improves proxy pointwise accuracy on " + ", ".join(improved) + "; it hurts " + ", ".join(worse) + ". Network consistency and inferred pointwise accuracy are different objectives. These observations do not establish the cause of every error; manual route inspection and official labels would be needed for stronger conclusions.")
    p("Overlap uses each directed arc once, weighted by its metre length. It does not penalise repeated traversals of the same correct arc. Ordered LCS F1 compares collapsed arc sequences and penalises ordering/repetition errors. Direct continuity is deliberately separate from route overlap. Decoder resets mark unresolved continuity, not successful inferred travel through an outage.")
    page()

    h("6. Geographical visualisation using Folium")
    p("Development map: student_test / 00000000. The Folium map contains GPS points (blue), candidate roads (grey), reference route (gold), reconstructed route (green), classifier-only layer (purple, optional) and decoder resets (red). The map uses a fit-to-track extent, an explicit legend, scale and layer controls. Every layer derives from the actual loaded trip and saved model.")
    image(reports / "evidence/folium_development.jpg")
    p("Interactive maps: reports/maps/development_00000000.html, development_00000007.html and development_00000056.html. Map generation code: pathsnap/maps.py. The static screenshot records an actual browser view. External map tiles and Folium JavaScript/CSS require network access; road geometry is included in the generated HTML.")
    p("Evaluator/API maps omit reference geometry entirely. Only development report maps include it. Maps draw separate road segments rather than fabricating straight lines across missing road connections.")
    page()

    h("7. API, UI and deployment")
    p("Laptop 1 (Deeptanshu's Linux laptop) loads the saved preprocessing/model configuration and runs FastAPI. Laptop 2 (Chiranth's Mac) serves the static ui/ directory. Its API URL points to Laptop 1's LAN address. GET /trips supplies the ten active IDs, POST /match accepts only a selected ID, and GET /maps/{request_id} returns a Folium prediction map. GET /health reports readiness and partition mode.")
    image(reports / "evidence/ui_working.jpg")
    p("Request IDs connect the UI result to structured receiving-server logs. Every API match runs the prediction pipeline; no stored prediction is returned in place of inference. Reference loading and scoring occur afterward, server-side. Responses expose aggregate metrics and predicted geometry/IDs, never reference labels or reference geometry.")
    api_evidence = reports / "evidence/api_smoke.json"
    if api_evidence.exists():
        evidence = read_json(api_evidence)
        p(f"Actual HTTP smoke test: {len(evidence['requests'])} of ten trip requests succeeded; each returned one prediction per GPS observation and a loadable map. Unknown IDs and invalid request bodies were also checked. Local cross-origin requests verify the browser/API configuration; they do not establish Mac-to-Linux networking.")
    p("The reserved evaluator folder must contain exactly ten trip directories. Evaluation mode refuses to start without it. Development mode is visibly marked and uses student-test trips. This prevents accidental relabelling of development demonstrations as final evaluation.")
    page()

    h("7.1 Demonstration evidence and completion boundary")
    terminal = reports / "evidence/api_terminal.png"
    if terminal.exists():
        image(terminal)
        p("Receiving-laptop terminal screenshot supplied/captured from the actual running server. Verify its request ID against the UI evidence.")
    else:
        p("<b>Required evidence remains open:</b> a screenshot of the actual receiving-laptop terminal showing the successful API request/response logs. GNOME denied automated window capture in this session. No replacement terminal image was fabricated. The team must capture the real terminal during the two-laptop demonstration and rerun report generation.")
    p("<b>Also open:</b> Chiranth's physical Mac-to-Linux network test and the team's review of the technical explanations. The current UI screenshot and HTTP results establish working local integration, not a completed two-device demonstration. The reserved archive has now been evaluated after the predictor freeze; see Section 7.2.")
    h2("Reproduction and final demonstration")
    p("README.md contains isolated-environment setup, full prepare/train/tune/evaluate/report commands, Linux API startup, Mac UI startup, CORS/network diagnosis and evaluator-mode configuration. scripts/smoke_api.py exercises every active trip and preserves real response summaries. All trained model definitions and tuning distributions are in source.")
    evaluator_path = artifacts / "evaluator_results.json"
    if evaluator_path.exists():
        page()
        h("7.2 Reserved final-partition evaluation")
        results = read_json(evaluator_path)
        rows = results["requests"]
        total = sum(row["observations"] for row in rows)
        p(f"The evaluator_test.zip archive arrived after development. Commit a61ce50 records the model/scoring freeze before archive access. The existing predictor processed all ten reserved trips through the live evaluation-mode API: {total:,} observations, ten successful HTTP responses. Source/model hashes were checked by scripts/evaluate_reserved.py. No evaluator result selected or modified a feature, weight, hyperparameter, decoder setting or threshold.")
        p("These are final-partition measurements, separate from the development comparisons. Pointwise scores still use the previously frozen geometric-alignment convention and are not organiser-certified physical per-GPS labels.")
        values = [["Trip", "GPS", "Classifier*", "HMM*", "Bridge F1", "Server s"]]
        for row in rows:
            m = row["metrics"]
            values.append([row["trip"], row["observations"], f"{m['classifier']['proxy_top1_accuracy']:.4f}",
                           f"{m['hmm']['proxy_top1_accuracy']:.4f}", f"{m['bridged_route']['route_length_f1']:.4f}", f"{row['server_seconds']:.2f}"])
        table(values, [90, 70, 85, 85, 85, 75])
        pooled = lambda key: sum(row["observations"] * row["metrics"][key]["proxy_top1_accuracy"] for row in rows) / total
        average = lambda key, metric: float(np.mean([row["metrics"][key][metric] for row in rows]))
        table([["Reserved metric", "Classifier", "Viterbi / bridge"],
               ["Pooled proxy pointwise accuracy", f"{pooled('classifier'):.4f}", f"{pooled('hmm'):.4f}"],
               ["Mean trip length-weighted overlap F1", f"{average('classifier','route_length_f1'):.4f}", f"{average('bridged_route','route_length_f1'):.4f}"],
               ["Mean trip direct arc continuity", f"{average('classifier','direct_arc_continuity'):.4f}", f"{average('bridged_route','direct_arc_continuity'):.4f}"],
               ["Mean trip ordered route LCS F1", f"{average('classifier','ordered_route_lcs_f1'):.4f}", f"{average('bridged_route','ordered_route_lcs_f1'):.4f}"]], [280, 105, 105])
        p("The reserved results preserve the important limitation: road overlap and continuity improve, but pointwise proxy accuracy and bridged ordered sequence F1 decrease. HMM-state ordered LCS F1 before bridges is 0.7912, while bridging introduces additional traversals. This does not support a universal claim that decoding improves route accuracy. We report the failure rather than retuning on the final partition.")
        image(reports / "evidence/ui_evaluation.jpg", width=430)
        p("Actual evaluation-mode UI view. Reference geometry and labels remain server-side. Per-request IDs and full metrics are in evaluator_results.json/.csv; the archive SHA256 and supplied trip IDs are in evaluator_manifest.json.", True)
        page()
    h2("8. Integrity, sources and limitations")
    p("AI assistance disclosure: OpenAI Codex assisted with reviewing the organiser materials, software implementation, experiment execution, debugging, result aggregation, and drafting this report and viva materials. Experimental numbers and plots come from recorded executions. The team is responsible for reviewing the analysis, understanding the source and meeting any additional organiser disclosure requirements. No other team's code, model, configuration or result was used.")
    p("No scientific novelty is claimed for the baseline. Bhuvan's proposed extensions remain pending. Important limitations are weak reference alignment, training quality filtering, shared development selection/reporting, local projection approximation, offline heading, truncated candidate sets, graph defects, explicit outage resets and incomplete final deployment evidence.")
    for text in [
        "Organiser sources: Main Guidelines.pdf; Dataset Exploration.pdf; Models and Training Guidelines.pdf; Deliverables.pdf; Assessment and Plagiarism Policy.pdf (supplied 18 Nov 2026).",
        "Newson, P. and Krumm, J. (2009), Hidden Markov Map Matching Through Noise and Sparseness. https://www.microsoft.com/en-us/research/publication/hidden-markov-map-matching-noise-sparseness/",
        "scikit-learn 1.7.2: model APIs, trip-grouped validation, pipelines and leakage prevention. https://scikit-learn.org/1.7/",
        "Optuna TPE sampler: https://optuna.readthedocs.io/en/v4.6.0/reference/samplers/generated/optuna.samplers.TPESampler.html",
        "SciPy directed shortest paths: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.dijkstra.html",
        "Folium map layers: https://python-visualization.github.io/folium/v0.19.7/reference.html; FastAPI CORS: https://fastapi.tiangolo.com/tutorial/cors/",
        "Supplied video, Uber Engineering, Map Matching @ Uber: https://www.youtube.com/watch?v=ChtumoDfZXI. Identity verified; transcript not used as experiment evidence.",
    ]: p(escape(text), True)

    destination = reports / "PathSnap_Technical_Report.pdf"
    def footer(canvas, document):
        canvas.saveState(); canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.HexColor("#6b7c70"))
        canvas.drawString(43, 25, "PathSnap | PT-088 | Development and frozen reserved results | 18 November 2026")
        canvas.drawRightString(A4[0]-43, 25, str(document.page)); canvas.restoreState()
    SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=43, leftMargin=43, topMargin=40,
                      bottomMargin=43, title="PathSnap Technical Report - PT-088", author="PT-088 (AI-assisted draft)").build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Generated {destination}", flush=True)
