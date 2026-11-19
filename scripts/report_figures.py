"""Figures for the final consolidated report, drawn only from committed artifacts.

No raw dataset is needed: every value comes from artifacts/*.json. Conceptual diagrams
(pipeline, deployment) contain no measured values.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
OUT = ROOT / "reports" / "figures"

INK, TEAL, GOLD, BLUE, RED, GREY = "#163d35", "#087f6a", "#d6a54c", "#5c7ea0", "#c4553f", "#9aa5a0"
MODEL_ORDER = ["logistic", "gaussian_nb", "decision_tree", "random_forest", "hist_gradient_boosting"]
SHORT = {"logistic": "Logistic", "gaussian_nb": "Gaussian NB", "decision_tree": "Decision tree",
         "random_forest": "Random forest", "hist_gradient_boosting": "Hist. boosting"}
MODEL_COLOURS = {"logistic": BLUE, "gaussian_nb": "#8c6bb1", "decision_tree": GOLD,
                 "random_forest": "#6b9e5f", "hist_gradient_boosting": TEAL}


def load(name: str):
    return json.loads((ART / name).read_text())


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#6c7a74",
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.titleweight": "bold", "legend.frameon": False, "legend.fontsize": 8,
        "figure.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 200,
        "axes.grid": True, "grid.color": "#e4e8e5", "grid.linewidth": 0.6, "axes.axisbelow": True,
    })


def save(fig, name: str) -> None:
    fig.savefig(OUT / name)
    plt.close(fig)


def label_bars(ax, bars, fmt="{:.3f}", size=7, inside=False):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(fmt.format(h), (bar.get_x() + bar.get_width() / 2, h), ha="center",
                    va="top" if inside else "bottom", fontsize=size, color="white" if inside else INK,
                    xytext=(0, -2 if inside else 1.5), textcoords="offset points")


# ---------------------------------------------------------------- conceptual diagrams
def box(ax, x, y, w, h, title, body, colour):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=colour, ec="none", alpha=0.95))
    ax.text(x + w / 2, y + h - 0.04, title, ha="center", va="top", fontsize=7.8, weight="bold", color="white")
    ax.text(x + w / 2, y + h - 0.13, body, ha="center", va="top", fontsize=6.6, color="white", linespacing=1.35)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=1.2, color=INK))


def pipeline_diagram():
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    steps = [
        ("1  Load & validate", ".track .nodes .arcs\nshape, finite, IDs,\nincreasing time", BLUE),
        ("2  Project", "per-trip local metres\nx = R cos(lat0) dlon\ny = R dlat", BLUE),
        ("3  Candidates", "KD-tree on 30 m road\nsamples; arcs within\n100 m; keep 8 closest", TEAL),
        ("4  Features", "13 route-independent\nfeatures per\n(point, candidate)", TEAL),
        ("5  Classifier", "tuned histogram\ngradient boosting\nP(candidate correct)", TEAL),
        ("6  Viterbi", "emissions = scores\ntransitions = road vs\nGPS distance", GOLD),
        ("7  Reconstruct", "shortest-path bridges;\nresets at gaps or\nno legal path", GOLD),
    ]
    w, h, gap = 0.128, 0.50, 0.0153
    for i, (t, b, c) in enumerate(steps):
        x = 0.005 + i * (w + gap)
        box(ax, x, 0.40, w, h, t, b, c)
        if i:
            arrow(ax, (x - gap + 0.001, 0.65), (x - 0.001, 0.65))
    ax.add_patch(FancyBboxPatch((0.28, 0.02), 0.44, 0.25, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc="#f2ebdc", ec=GOLD, lw=1))
    ax.text(0.5, 0.235, "Training / scoring only (never inside prediction)", ha="center", va="top",
            fontsize=8.2, weight="bold", color=INK)
    ax.text(0.5, 0.165, ".route  ->  monotone alignment  ->  proxy label per GPS point\n"
            "used to label training candidates and to score predictions server-side",
            ha="center", va="top", fontsize=7.4, color=INK, linespacing=1.4)
    ax.text(0.005, 0.95, "Prediction path (API and evaluation share this code)", fontsize=8.2, weight="bold", color=INK)
    save(fig, "pipeline.png")


def deployment_diagram():
    fig, ax = plt.subplots(figsize=(10, 3.3))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.06), 0.36, 0.86, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc="#eef3f1", ec=TEAL, lw=1.2))
    ax.text(0.19, 0.88, "Laptop 2  -  evaluator UI", ha="center", weight="bold", fontsize=9.5, color=INK)
    ax.text(0.19, 0.78, "python3 -m http.server 8010 --directory ui", ha="center", fontsize=7.4, family="DejaVu Sans Mono")
    for k, line in enumerate(["1. GET /trips  ->  ten trip IDs in a dropdown",
                              "2. Evaluator picks one trip",
                              "3. POST /match {trip_id}",
                              "8. Show route map, metrics, request ID",
                              "   Model trace page: /trace, /explain"]):
        ax.text(0.04, 0.64 - k * 0.11, line, fontsize=8, color=INK)
    ax.add_patch(FancyBboxPatch((0.55, 0.06), 0.44, 0.86, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc="#f3f1ea", ec=GOLD, lw=1.2))
    ax.text(0.77, 0.88, "Laptop 1  -  model server (FastAPI)", ha="center", weight="bold", fontsize=9.5, color=INK)
    ax.text(0.77, 0.78, "python -m pathsnap.cli serve --mode evaluation ...", ha="center",
            fontsize=7.2, family="DejaVu Sans Mono")
    for k, line in enumerate(["4. Load .track .nodes .arcs for that trip",
                              "5. Candidates -> features -> classifier -> Viterbi",
                              "6. THEN open .route and score server-side",
                              "7. Log 'PATHSNAP {request_id, trip, seconds}'",
                              "   Reference labels/geometry never leave this laptop"]):
        ax.text(0.575, 0.64 - k * 0.11, line, fontsize=8, color=INK)
    arrow(ax, (0.375, 0.58), (0.545, 0.58))
    arrow(ax, (0.545, 0.36), (0.375, 0.36))
    ax.text(0.46, 0.62, "trip ID only", ha="center", fontsize=7.6, color=INK)
    ax.text(0.46, 0.27, "predictions +\naggregate metrics", ha="center", fontsize=7.6, color=INK)
    ax.text(0.46, 0.08, "same LAN, port 8000", ha="center", fontsize=7.2, color=GREY)
    save(fig, "deployment.png")


# ---------------------------------------------------------------- dataset / EDA
def eda_figures(inv):
    parts = {"train": TEAL, "student_test": GOLD}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9), constrained_layout=True)
    for split, c in parts.items():
        rows = [r for r in inv if r["split"] == split]
        bins = np.logspace(np.log10(200), np.log10(25000), 18)
        axes[0].hist([r["gps_observations"] for r in rows], bins=bins, color=c, alpha=.75, label=split)
        bins = np.logspace(np.log10(500), np.log10(400000), 18)
        axes[1].hist([r["arcs"] for r in rows], bins=bins, color=c, alpha=.75, label=split)
        bins = np.logspace(np.log10(40), np.log10(3000), 16)
        axes[2].hist([r["reference_route_arcs"] for r in rows], bins=bins, color=c, alpha=.75, label=split)
    for ax, t, xl in zip(axes, ["(a) GPS observations per trip", "(b) Directed road arcs per trip",
                                "(c) Reference route length per trip"],
                         ["observations (log scale)", "arcs (log scale)", "route entries (log scale)"]):
        ax.set(xscale="log", title=t, xlabel=xl, ylabel="trips")
    axes[0].legend()
    save(fig, "eda_sizes.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 2.9), constrained_layout=True)
    for split, c in parts.items():
        rows = sorted([r for r in inv if r["split"] == split], key=lambda r: r["max_gap_s"])
        share = [r["one_second_steps"] / r["total_steps"] for r in rows]
        axes[0].hist(share, bins=np.linspace(0.5, 1, 26), color=c, alpha=.75, label=split)
    rows = sorted(inv, key=lambda r: r["max_gap_s"])
    axes[1].scatter(np.arange(len(rows)), [r["max_gap_s"] for r in rows], s=10,
                    c=[parts[r["split"]] for r in rows])
    axes[1].axhline(60, color=RED, lw=1, ls="--")
    axes[1].text(2, 75, "decoder reset threshold: 60 s", color=RED, fontsize=7.5)
    axes[1].set(yscale="log", title="(b) Largest time gap in each trip (sorted)", xlabel="trip rank",
                ylabel="max gap, s (log scale)")
    axes[0].set(title="(a) Share of steps that are exactly 1 s", xlabel="share of steps", ylabel="trips")
    axes[0].legend(loc="upper left")
    save(fig, "eda_sampling.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2), constrained_layout=True, gridspec_kw={"width_ratios": [1.6, 1]})
    for split, c in parts.items():
        rows = [r for r in inv if r["split"] == split]
        lon = [np.mean(r["longitude_extent"]) for r in rows]
        lat = [np.mean(r["latitude_extent"]) for r in rows]
        axes[0].scatter(lon, lat, s=16, color=c, label=split, edgecolor="white", lw=.4)
        axes[1].hist([max(r["network_extent_km"]) for r in rows], bins=np.linspace(0, 80, 17), color=c, alpha=.75)
    axes[0].axhline(0, color=GREY, lw=.6); axes[0].axvline(0, color=GREY, lw=.6)
    axes[0].set(xlim=(-180, 180), ylim=(-60, 80), title="(a) Trip network centres (longitude, latitude)",
                xlabel="longitude (deg)", ylabel="latitude (deg)")
    axes[0].legend(loc="lower left")
    axes[1].set(title="(b) Largest network dimension per trip", xlabel="km", ylabel="trips")
    save(fig, "eda_geography.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 2.9), constrained_layout=True)
    for split, c in parts.items():
        rows = [r for r in inv if r["split"] == split]
        axes[0].scatter([r["arcs"] for r in rows], [r["weak_components"] for r in rows], s=13, color=c, label=split)
        axes[1].scatter([r["arcs"] for r in rows], [r["zero_length_arcs"] for r in rows], s=13, color=c)
    axes[0].set(xscale="log", yscale="log", title="(a) Weakly connected components vs network size",
                xlabel="directed arcs (log)", ylabel="weak components (log)")
    axes[0].legend()
    axes[1].set(xscale="log", title="(b) Zero-length arcs per trip", xlabel="directed arcs (log)",
                ylabel="zero-length arcs")
    save(fig, "eda_network.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.0), constrained_layout=True)
    rows = sorted(inv, key=lambda r: r["proxy_label_distance_p95_m"])
    x = np.arange(len(rows))
    axes[0].bar(x, [r["proxy_label_distance_p95_m"] for r in rows], color=[parts[r["split"]] for r in rows], width=.9)
    axes[0].plot(x, [r["proxy_label_distance_median_m"] for r in rows], "o", ms=2.4, color=INK, label="median")
    axes[0].axhline(75, color=RED, ls="--", lw=1)
    axes[0].text(2, 82, "75 m training-label filter", color=RED, fontsize=7.5)
    for r, xi in zip(rows, x):
        if r["proxy_label_distance_p95_m"] > 60:
            axes[0].annotate(r["trip"][-2:], (xi, r["proxy_label_distance_p95_m"]), fontsize=6.5,
                             ha="center", xytext=(0, 2), textcoords="offset points")
    axes[0].set(title="(a) GPS-to-reference distance per trip (bar = p95)", xlabel="trip rank",
                ylabel="metres"); axes[0].legend(loc="upper left")
    tr = sorted([r for r in inv if r["split"] == "train"], key=lambda r: -r["training_quality_excluded_observations"])[:10]
    axes[1].barh([r["trip"] for r in tr][::-1], [r["training_quality_excluded_observations"] for r in tr][::-1], color=TEAL)
    for i, r in enumerate(tr[::-1]):
        axes[1].text(r["training_quality_excluded_observations"], i,
                     f" {r['training_quality_excluded_observations']:,} of {r['gps_observations']:,}", va="center", fontsize=7)
    axes[1].set(title="(b) Training trips with most points excluded (>75 m)", xlabel="excluded observations",
                xlim=(0, 23000))
    save(fig, "eda_reference_quality.png")


# ---------------------------------------------------------------- models
def model_figures(models, tuning):
    by = {m["name"]: m for m in models}
    names = MODEL_ORDER
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), constrained_layout=True)
    cv = [by[n]["cv_candidate_macro_f1"] for n in names]
    sd = [by[n]["cv_candidate_macro_f1_std"] for n in names]
    bars = axes[0].bar(x, cv, yerr=sd, capsize=3, color=[MODEL_COLOURS[n] for n in names])
    for i, n in enumerate(names):
        axes[0].plot([i] * 3, [f["candidate_macro_f1"] for f in by[n]["cv_folds"]], "o", ms=3, color=INK, alpha=.7)
    for i, (v, s_) in enumerate(zip(cv, sd)):
        axes[0].text(i, v + s_ + 0.002, f"{v:.4f}", ha="center", fontsize=6.6)
    axes[0].set(ylim=(0.8, 0.97), title="(a) Grouped 3-fold CV candidate Macro F1", ylabel="Macro F1 (dots = folds)")
    st = [by[n]["student_test"]["summary"] for n in names]
    b1 = axes[1].bar(x - .19, [s["proxy_top1_accuracy"] for s in st], .38, color=TEAL, label="Top-1 (proxy)")
    b2 = axes[1].bar(x + .19, [s["candidate_macro_f1"] for s in st], .38, color=GOLD, label="candidate Macro F1")
    label_bars(axes[1], b1, "{:.3f}", 6); label_bars(axes[1], b2, "{:.3f}", 6)
    axes[1].set(ylim=(0.75, 0.97), title="(b) Student-test Top-1 and Macro F1"); axes[1].legend(loc="upper left")
    b = axes[2].bar(x, [s["candidate_log_loss"] for s in st], color=[MODEL_COLOURS[n] for n in names])
    label_bars(axes[2], b, size=6.8)
    axes[2].set(title="(c) Student-test candidate log loss (lower is better)", ylim=(0, 0.62))
    for ax in axes:
        ax.set_xticks(x, [SHORT[n] for n in names], rotation=28, ha="right")
    save(fig, "model_comparison.png")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), constrained_layout=True)
    b1 = axes[0].bar(x - .19, [s["candidate_precision"] for s in st], .38, color=BLUE, label="precision")
    b2 = axes[0].bar(x + .19, [s["candidate_recall"] for s in st], .38, color=RED, label="recall")
    label_bars(axes[0], b1, "{:.2f}", 6); label_bars(axes[0], b2, "{:.2f}", 6)
    axes[0].set(ylim=(0.5, 1.07), title="(a) Student-test precision / recall (positive class)")
    axes[0].legend(loc="upper center", ncol=2)
    tr = [by[n]["training"]["candidate_macro_f1"] for n in names]
    stf = [s["candidate_macro_f1"] for s in st]
    w = .26
    axes[1].bar(x - w, tr, w, color="#a9c9bd", label="fit (training rows)")
    axes[1].bar(x, cv, w, color=TEAL, label="grouped CV")
    axes[1].bar(x + w, stf, w, color=GOLD, label="student test")
    axes[1].set(ylim=(0.8, 0.96), title="(b) Fit vs CV vs student-test Macro F1"); axes[1].legend(loc="upper left", fontsize=7)
    axes[2].bar(x - .19, [by[n]["full_fit_seconds"] for n in names], .38, color=INK, label="full fit (s)")
    axes[2].bar(x + .19, [s["model_inference_seconds"] for s in st], .38, color=GOLD, label="inference, 19,160 obs (s)")
    axes[2].set(yscale="log", title="(c) Computational cost (log scale)"); axes[2].legend(loc="upper left", fontsize=7)
    for ax in axes:
        ax.set_xticks(x, [SHORT[n] for n in names], rotation=28, ha="right")
    save(fig, "model_quality_cost.png")

    fig, axes = plt.subplots(1, 5, figsize=(10.5, 2.4), constrained_layout=True)
    for ax, n in zip(axes, names):
        m = np.asarray(by[n]["student_test"]["summary"]["candidate_confusion_matrix"])
        rates = m / m.sum(axis=1, keepdims=True)
        ax.imshow(rates, cmap="Greens", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{m[i, j]:,}\n{rates[i, j]:.1%}", ha="center", va="center", fontsize=7,
                        color="white" if rates[i, j] > .5 else INK)
        ax.set(xticks=[0, 1], yticks=[0, 1], title=SHORT[n], xlabel="predicted")
        ax.grid(False)
        if n == names[0]:
            ax.set_ylabel("proxy label")
    save(fig, "confusion_matrices.png")

    trips = [r["trip"] for r in by[names[0]]["student_test"]["trips"]]
    fig, ax = plt.subplots(figsize=(10.5, 2.9), constrained_layout=True)
    for k, n in enumerate(names):
        vals = [r["proxy_top1_accuracy"] for r in by[n]["student_test"]["trips"]]
        ax.plot(np.arange(len(trips)) + (k - 2) * 0.07, vals, "o-", ms=3.5, lw=1, color=MODEL_COLOURS[n], label=SHORT[n])
    ax.set_xticks(np.arange(len(trips)), trips, rotation=25, ha="right")
    ax.set(ylabel="classifier-only Top-1 (proxy)", title="Per-trip student-test Top-1 for every model", ylim=(0.5, 1.0))
    ax.legend(ncol=5, loc="lower left")
    save(fig, "model_per_trip.png")


def bayes_figures(trials, tuning):
    vals = np.array([t["value"] for t in trials])
    fig, ax = plt.subplots(figsize=(10.5, 2.7), constrained_layout=True)
    ax.plot(np.arange(len(vals)), vals, "o", color=BLUE, label="trial objective (mean CV Macro F1)")
    ax.step(np.arange(len(vals)), np.maximum.accumulate(vals), where="post", color=TEAL, label="best so far")
    ax.axhline(tuning["untuned_cv_macro_f1"], color=GOLD, ls=":", label=f"untuned default {tuning['untuned_cv_macro_f1']:.4f}")
    ax.axvline(3.5, color=GREY, ls="--"); ax.text(3.6, vals.min(), " adaptive TPE trials ->", fontsize=7.5, color=GREY)
    ax.text(-0.3, vals.min(), "random startup", fontsize=7.5, color=GREY)
    ax.set(xlabel="trial number", ylabel="mean CV Macro F1", title="Optuna TPE optimisation history (12 trials)")
    ax.set_xticks(range(len(vals))); ax.legend(loc="lower right")
    save(fig, "bayes_history.png")
    keys = ["learning_rate", "max_leaf_nodes", "l2_regularization", "min_samples_leaf"]
    logs = {"learning_rate": True, "l2_regularization": True}
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.5), constrained_layout=True)
    best = int(np.argmax(vals))
    for ax, k in zip(axes, keys):
        xs = [t["parameters"][k] for t in trials]
        ax.scatter(xs, vals, s=18, c=np.arange(len(vals)), cmap="viridis")
        ax.scatter([xs[best]], [vals[best]], s=70, facecolors="none", edgecolors=RED, lw=1.4)
        ax.set(xlabel=k, xscale="log" if logs.get(k) else "linear")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylabel("mean CV Macro F1")
    fig.suptitle("Objective vs each hyperparameter (colour = trial order; red ring = best trial)", fontsize=9, weight="bold")
    save(fig, "bayes_parameters.png")


# ---------------------------------------------------------------- decoder
def decoder_figures(decoder, search, reserved, all_models):
    fig, ax = plt.subplots(figsize=(10.5, 2.6), constrained_layout=True)
    labels = [f"beta={t['beta_m']:g} m\n{'row-normalised' if t['normalize_transition_rows'] else 'global scale'}"
              for t in search["trials"]]
    vals = [t["development_proxy_top1_accuracy"] for t in search["trials"]]
    bars = ax.bar(range(len(vals)), vals, color=[GREY if t["normalize_transition_rows"] else TEAL for t in search["trials"]])
    label_bars(ax, bars, "{:.4f}", 7.5)
    ax.set_xticks(range(len(vals)), labels, fontsize=7.5)
    ax.set(ylim=(0.82, 0.86), ylabel="pooled proxy Top-1", title="Decoder settings compared on the student-test trips (chosen: global scale, beta = 25 m)")
    save(fig, "decoder_search.png")

    def per_trip(rows, title, name, getter):
        trips = [getter(r)[0] for r in rows]
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), constrained_layout=True)
        x = np.arange(len(trips))
        for ax, metric, t in zip(axes, ["proxy_top1_accuracy", "route_length_f1", "direct_arc_continuity"],
                                 ["(a) pointwise Top-1 (proxy)", "(b) route overlap F1", "(c) direct arc continuity"]):
            c = [getter(r)[1]["classifier"][metric] for r in rows]
            h = [getter(r)[1]["hmm"][metric] for r in rows]
            ax.bar(x - .27, c, .27, color=BLUE, label="classifier only")
            ax.bar(x, h, .27, color=TEAL, label="HMM states")
            if metric != "proxy_top1_accuracy":
                b = [getter(r)[1]["bridged_route"][metric] for r in rows]
                ax.bar(x + .27, b, .27, color=GOLD, label="HMM + bridges")
            ax.set_xticks(x, [t_[-2:] for t_ in trips])
            ax.set(title=t, xlabel="trip (last two digits)", ylim=(0.3 if metric != "route_length_f1" else 0.55, 1.02))
        handles, labs = axes[2].get_legend_handles_labels()
        fig.legend(handles, labs, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
        fig.suptitle(title, fontsize=9.5, weight="bold")
        save(fig, name)

    per_trip(decoder, "Student-test (development): classifier-only vs Viterbi, per trip", "decoder_dev.png",
             lambda r: (r["trip"], r))
    per_trip(reserved["requests"], "Reserved evaluator trips (scored once after freeze): classifier-only vs Viterbi",
             "decoder_reserved.png", lambda r: (r["trip"], r["metrics"]))

    fig, ax = plt.subplots(figsize=(10.5, 2.7), constrained_layout=True)
    ms = all_models["models"]
    x = np.arange(len(ms))
    b1 = ax.bar(x - .2, [m["summary"]["classifier_proxy_top1"] for m in ms], .4, color=BLUE, label="classifier only")
    b2 = ax.bar(x + .2, [m["summary"]["hmm_proxy_top1"] for m in ms], .4, color=TEAL, label="with Viterbi")
    label_bars(ax, b1, "{:.3f}", 6.5); label_bars(ax, b2, "{:.3f}", 6.5)
    ax.set_xticks(x, [m["label"] + (" (tuned)" if m["tuned"] else (" (untuned)" if "boost" in m["label"].lower() else "")) for m in ms],
                  rotation=12, fontsize=7.5)
    ax.set(ylim=(0.6, 0.95), ylabel="pooled proxy Top-1",
           title="Report-only: every model family on the reserved trips, before and after Viterbi")
    ax.legend(loc="upper left")
    save(fig, "reserved_all_models.png")

    fig, ax = plt.subplots(figsize=(10.5, 2.6), constrained_layout=True)
    trips = [r["trip"] for r in decoder]
    x = np.arange(len(trips))
    bottom = np.zeros(len(trips))
    for key, c, lab in [("candidate_preparation_seconds", BLUE, "candidates + features"),
                        ("model_seconds", GOLD, "classifier"), ("decoding_and_bridging_seconds", TEAL, "Viterbi + bridges")]:
        v = np.array([r["timings"][key] for r in decoder])
        ax.bar(x, v, bottom=bottom, color=c, label=lab)
        bottom += v
    for xi, r in zip(x, decoder):
        ax.text(xi, bottom[xi] + 0.03, f"{r['observations']:,} pts", ha="center", fontsize=6.8)
    ax.set_xticks(x, trips, rotation=20, ha="right")
    ax.set(ylabel="seconds", title="Prediction time per student-test trip, by stage (excludes reference scoring and map rendering)")
    ax.legend(loc="upper left")
    save(fig, "timings.png")


def improved_figure(imp, st, ev):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 2.8), constrained_layout=True)
    base, impd = imp["baseline"], imp["improved"]
    labels = ["CV Macro F1", "student-test\nMacro F1", "student-test\nclassifier Top-1", "student-test\nViterbi Top-1"]
    d = [base["cv_candidate_macro_f1"], base["student_test_candidate_macro_f1"], base["student_test_classifier_top1"],
         st["default"]["summary"]["hmm_proxy_top1"]]
    i = [impd["cv_candidate_macro_f1"], impd["student_test_candidate_macro_f1"], impd["student_test_classifier_top1"],
         st["improved"]["summary"]["hmm_proxy_top1"]]
    x = np.arange(len(labels))
    b1 = axes[0].bar(x - .2, d, .4, color=GREY, label="default (13 features, served)")
    b2 = axes[0].bar(x + .2, i, .4, color=TEAL, label="improved (30 features, opt-in)")
    label_bars(axes[0], b1, "{:.4f}", 6.5); label_bars(axes[0], b2, "{:.4f}", 6.5)
    axes[0].set_xticks(x, labels, fontsize=7.5); axes[0].set(ylim=(0.8, 0.97), title="(a) Development partitions")
    axes[0].legend(loc="upper right", fontsize=7)
    labels = ["classifier Top-1", "Viterbi Top-1", "bridged route F1", "bridged LCS F1"]
    keys = ["classifier_proxy_top1", "hmm_proxy_top1", "bridged_route_f1_mean", "bridged_lcs_f1_mean"]
    x = np.arange(len(labels))
    b1 = axes[1].bar(x - .2, [ev["default"]["summary"][k] for k in keys], .4, color=GREY)
    b2 = axes[1].bar(x + .2, [ev["improved"]["summary"][k] for k in keys], .4, color=TEAL)
    label_bars(axes[1], b1, "{:.3f}", 6.5); label_bars(axes[1], b2, "{:.3f}", 6.5)
    axes[1].set_xticks(x, labels, fontsize=7.5); axes[1].set(ylim=(0.6, 1.0), title="(b) Reserved trips (report-only, scored after development)")
    save(fig, "improved_variant.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    pipeline_diagram()
    deployment_diagram()
    eda_figures(load("inventory.json"))
    tuning = load("tuning.json")
    model_figures(load("model_comparison.json"), tuning)
    bayes_figures(load("bayesian_trials.json"), tuning)
    decoder_figures(load("decoder_comparison.json"), load("decoder_search.json"), load("evaluator_results.json"),
                    load("evaluator_model_comparison.json"))
    improved_figure(load("improved/results.json"), load("improved/student_test_results.json"),
                    load("improved/evaluator_test_results.json"))
    print(f"Figures written to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
