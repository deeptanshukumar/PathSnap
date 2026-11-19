"""Section 3: model experimentation and comparison (Deliverables 3, 3.1, 3.2, 3.3)."""

from reportlab.platypus import KeepTogether, TableStyle

from .data import MODEL_ORDER, NAMES, ROOT, f, n, pct
from .layout import H1, H2, H3, P, bullets, callout, eq, figure, question, table

FAMILY = {"logistic": "Linear", "gaussian_nb": "Probabilistic",
          "decision_tree": "Tree-based", "random_forest": "Bagging ensemble",
          "hist_gradient_boosting": "Boosting"}
CONFIG_TEXT = {
    "logistic": "L2, C = 1.0, L-BFGS, max_iter 800; StandardScaler in fold",
    "gaussian_nb": "var_smoothing 1e-7; StandardScaler in fold",
    "decision_tree": "Gini, max_depth 10, min_samples_leaf 20",
    "random_forest": "80 trees, max_depth 14, min_samples_leaf 8, max_features sqrt",
    "hist_gradient_boosting": "120 iterations, lr 0.1, 15 leaves, L2 1.0, min_samples_leaf 30, no early stopping",
}


def s(m):
    return m["student_test"]["summary"]


def section(D, fig):
    M = D["models"]
    tun = D["tuning"]
    tst = tun["tuned_student_test"]["summary"]
    out = H1("3. Model experimentation and comparison")
    out += H2("3.0 Learning formulation and experimental protocol")
    out += [P("<b>Formulation.</b> Each training example is one (GPS observation, candidate arc) pair described by the "
              "13 features of Table 2.5. Its label is 1 if the candidate equals the observation's proxy reference arc and 0 "
              "otherwise. A model outputs P(candidate is correct | features). Two uses follow from that one probability: "
              "for each observation the highest-scoring candidate is the classifier-only prediction, and the probabilities "
              "become the HMM's emission evidence. A multiclass model over arc IDs is impossible here: the classes would "
              "differ in every trip.")]
    m = D["manifest"]
    folds = D["folds"]
    out += [P(f"<b>Data.</b> {n(m['candidate_training_rows'])} training rows from all 80 training trips, "
              f"{pct(m['positive_rows'] / m['candidate_training_rows'])} positive. Every student-test observation is scored "
              f"({n(D['student_test']['obs_total'])} observations, {n(sum(map(sum, s(M['logistic'])['candidate_confusion_matrix'])))} "
              "valid candidate rows).")]
    out += [P(f"<b>Cross-validation.</b> scikit-learn StratifiedGroupKFold with {D['config']['cv_folds']} folds, "
              f"shuffled with seed {D['config']['seed']}, grouped by trip. Validation folds hold "
              f"{', '.join(str(len(x['validation_trips'])) for x in folds)} whole trips. The code raises an error if any trip "
              "appears on both sides of a split, and a unit test checks it. All five models use exactly the same three "
              "folds, recorded in <font name='Mono'>artifacts/folds.json</font>, so their scores are paired and directly "
              "comparable. Grouping by trip is essential: neighbouring 1-second points from one trip are almost identical, "
              "so a random row split would let a model see near-copies of its validation rows and report inflated scores. "
              "Three folds rather than five was a time-budget choice; each fold still validates on about 27 unseen trips.")]
    out += [P("<b>Fair treatment.</b> Same features, same rows, same folds, same seed for every model. Scaling, where "
              "needed, is fitted inside each fold. Histogram gradient boosting's built-in early stopping is disabled "
              "because it would carve its own validation set out of random rows, which reintroduces exactly the "
              "within-trip leakage that grouped folds prevent.")]
    out += H3("Metrics and their definitions")
    rows = [["Metric", "Level", "Definition", "What it tells us"],
            ["Accuracy, precision, recall, F1", "candidate row", "Usual binary definitions at threshold 0.5; positive = "
             "correct arc", "How well yes/no decisions match the proxy labels"],
            ["Macro F1", "candidate row", "Mean of F1 for the positive class and F1 for the negative class",
             "Balanced quality despite 14.5% positives"],
            ["Confusion matrix", "candidate row", "Counts of TN, FP, FN, TP at 0.5", "Which kind of error dominates"],
            ["Log loss", "candidate row", "−mean[y log p + (1 − y) log(1 − p)]",
             "Probability quality; matters because probabilities become emissions"],
            ["Observation proxy log loss", "observation", "−mean log(normalised score of the proxy arc), "
             "floor 1e-12", "How much probability lands on the right candidate"],
            ["Top-1 (proxy)", "observation", "Share of observations whose highest-scoring candidate equals the "
             "proxy label; a candidate miss counts as wrong", "Pointwise matching quality, threshold-free"],
            ["Candidate miss rate", "observation", "Share of observations whose proxy arc is not among the candidates",
             "Ceiling on Top-1, independent of the model"],
            ["Arc Macro F1 (proxy)", "observation", "Macro F1 over arc IDs within a trip, averaged over trips",
             "Whether rare arcs are matched too"],
            ["Direct arc continuity", "sequence", "Share of consecutive distinct predicted arcs where one ends at the "
             "node the next begins", "Route plausibility (1 = every step connects)"],
            ["Route overlap F1", "sequence", "Precision and recall of the predicted vs reference arc sets, weighted by "
             "arc length", "How much of the right road was found, in metres"],
            ["Ordered route LCS F1", "sequence", "2 · LCS / (|predicted| + |reference|) on collapsed arc sequences",
             "Order and repetition, which overlap F1 ignores"],
            ["Training / inference time", "-", "Wall-clock seconds on the same CPU laptop, 4 threads",
             "Computational efficiency"]]
    out += [table(rows, [1.7, 1.0, 3.2, 2.5]), P("Table 3.1. Metrics (pathsnap/metrics.py, experiments.py). "
                                                 "Candidate miss rate is a property of candidate generation, so it is the same "
                                                 f"for every model: {pct(D['student_test']['miss_pooled'])} on student test.",
                                                 "caption")]
    out += callout([
        "<b>Primary selection metric: mean grouped-CV candidate Macro F1 on the 80 training trips.</b> Reasons: (1) it is "
        "measured on training trips only, which keeps the student-test partition clean as an independent comparison; "
        "(2) Macro F1 weights the rare positive class and the common negative class equally, so a model cannot score "
        "well by predicting \"no\" everywhere (that would give about 85% accuracy); (3) the 0.5 threshold is meaningful "
        "because the scores are used as probabilities downstream. Log loss is the tie-breaker because the HMM consumes "
        "the probabilities, and Top-1 shows ranking quality within each observation. In our results the top three models "
        "are in the same order under CV Macro F1 and under student-test Top-1, so the choice of primary metric does not "
        "change the winner (Table 3.2).",
    ], title="Which metric selects the model, and why")

    out += H2("3.1 Overall model comparison")
    st_obs = D["student_test"]["obs_total"]
    rows = [["Model", "Algorithm family", "Key configuration", "CV Macro F1 (± SD)", "Top-1 (proxy)",
             "Route continuity", "Student- test Macro F1", "Log loss", "Inference time"]]
    for k in MODEL_ORDER:
        x = M[k]
        rows.append([NAMES[k], FAMILY[k], CONFIG_TEXT[k],
                     f"{f(x['cv_candidate_macro_f1'])} ± {f(x['cv_candidate_macro_f1_std'])}",
                     f(s(x)["proxy_top1_accuracy"]), f(s(x)["mean_trip_direct_arc_continuity"]),
                     f(s(x)["candidate_macro_f1"]), f(s(x)["candidate_log_loss"]),
                     f"{s(x)['model_inference_seconds']:.3f} s ({1000 * s(x)['model_inference_seconds'] / st_obs * 1000:.1f} ms / 1k points)"])
    bp = tun["best_parameters"]
    rows.append(["Histogram gradient boosting, Bayesian-tuned (final classifier)", "Boosting",
                 f"lr {bp['learning_rate']:.4f}, {bp['max_leaf_nodes']} leaves, L2 {bp['l2_regularization']:.3f}, "
                 f"min_samples_leaf {bp['min_samples_leaf']}, 120 iterations",
                 f"{f(tun['cv_candidate_macro_f1'])} ± {f(tun['cv_candidate_macro_f1_std'])}",
                 f(tst["proxy_top1_accuracy"]), f(tst["mean_trip_direct_arc_continuity"]), f(tst["candidate_macro_f1"]),
                 f(tst["candidate_log_loss"]),
                 f"{tst['model_inference_seconds']:.3f} s ({1000 * tst['model_inference_seconds'] / st_obs * 1000:.1f} ms / 1k points)"])
    dv = D["dev_summary"]
    rows.append(["Final system: tuned boosting + HMM / Viterbi", "Boosting + sequence decoder",
                 "beta 25 m, global transition scale, 60 s gap reset, 55 m/s routing bound",
                 "N/A: not CV-scored; classifier 0.9290",
                 f(dv["hmm_top1"]),
                 f"{f(dv['bridged_route_direct_arc_continuity'])} bridged ({f(dv['hmm_direct_arc_continuity'])} decoded states)",
                 "N/A: no candidate scores (arcs only)",
                 "N/A: one path, no scores",
                 f"{sum(D['dev_seconds']) / len(D['dev_seconds']):.2f} s per trip, full pipeline (mean)"])
    t = table(rows, [1.4, 1.0, 1.75, 1.0, 0.75, 1.0, 0.95, 0.85, 1.15], bold_first_col=True, highlight_rows=(6,),
              font_size=7.2)
    out += [t, P("Table 3.2. Consolidated comparison in the exact column format of Deliverables 3.1. Partition: CV on the "
                 "80 training trips; all other columns on the 10 student-test trips (19,160 observations). Route "
                 "continuity is the mean over trips of direct arc continuity for classifier-only predictions. Log loss is "
                 "candidate-level binary log loss. Inference time covers classifier scoring of all candidates of all 10 "
                 "trips. Source: model_comparison.json, tuning.json, decoder_comparison.json.", "caption")]

    rows = [["Model", "Fit F1", "OOF log loss", "Accuracy", "Precision", "Recall", "Obs. log loss", "Arc Macro F1",
             "Overlap F1", "LCS F1", "Fit (s)"]]
    for k in MODEL_ORDER:
        x = M[k]; y = s(x)
        rows.append([NAMES[k], f(x["training"]["candidate_macro_f1"]), f(x["oof"]["candidate_log_loss"]),
                     f(y["candidate_accuracy"]), f(y["candidate_precision"]), f(y["candidate_recall"]),
                     f(y["observation_proxy_log_loss"]), f(y["mean_trip_proxy_arc_macro_f1"]),
                     f(y["mean_trip_route_length_f1"]), f(y["mean_trip_ordered_route_lcs_f1"]),
                     f"{x['full_fit_seconds']:.2f}"])
    out += [table(rows, [1.9, .7, .8, .78, .78, .7, .8, .8, .75, .7, .6], bold_first_col=True, font_size=7.0),
            P("Table 3.3. Further metrics required by the Models and Training Guidelines. Fit F1: candidate Macro F1 on the "
              "training rows the final model was fitted on. OOF log loss: out-of-fold log loss over all three CV folds. "
              "All other columns: student test. Fit seconds: final fit on all 255,024 training rows.", "caption")]
    rows = [["Model", "Fold 0", "Fold 1", "Fold 2", "Mean", "SD", "Mean fold fit (s)"]]
    for k in MODEL_ORDER:
        fs = M[k]["cv_folds"]
        rows.append([NAMES[k]] + [f(x["candidate_macro_f1"]) for x in fs] +
                    [f(M[k]["cv_candidate_macro_f1"]), f(M[k]["cv_candidate_macro_f1_std"]),
                     f"{sum(x['training_seconds'] for x in fs) / 3:.2f}"])
    out += [KeepTogether([table(rows, [2.2, .8, .8, .8, .8, .8, 1.1], bold_first_col=True),
            P("Table 3.4. Candidate Macro F1 per CV fold. Fold 2 is the hardest fold for four of the five models "
              "(Naive Bayes finds fold 0 hardest): which trips a fold holds matters as much as the model.", "caption")])]

    out += H3("3.1.1 Model selection and its justification")
    hgb, rf = M["hist_gradient_boosting"], M["random_forest"]
    gap_ = hgb["cv_candidate_macro_f1"] - rf["cv_candidate_macro_f1"]
    rows = [["Criterion (Models Guidelines 6)", "Evidence", "Verdict"],
            ["Generalisation", f"Highest mean CV Macro F1 ({f(hgb['cv_candidate_macro_f1'])}); highest student-test "
             f"Macro F1 ({f(s(hgb)['candidate_macro_f1'])}) and Top-1 ({f(s(hgb)['proxy_top1_accuracy'])})", "Boosting"],
            ["Stability across folds", f"SD {f(hgb['cv_candidate_macro_f1_std'])}, similar to random forest "
             f"({f(rf['cv_candidate_macro_f1_std'])}); logistic and NB are more stable but much weaker", "Comparable"],
            ["Probability quality", f"Lowest student-test log loss ({f(s(hgb)['candidate_log_loss'])}) and OOF log loss "
             f"({f(hgb['oof']['candidate_log_loss'])})", "Boosting"],
            ["Inference speed", f"{s(hgb)['model_inference_seconds']:.2f} s for all 10 trips; slowest of the five, yet only "
             "about 10% of total prediction time (Figure 5.5)", "Acceptable"],
            ["Model complexity", f"Fit {hgb['full_fit_seconds']:.1f} s vs {rf['full_fit_seconds']:.1f} s for random forest; "
             "the saved final boosting model is 108 KB", "Boosting"],
            ["Compatibility with the decoder", "Smooth, well-calibrated probabilities avoid hard 0/1 emissions; a single "
             "deep tree gives coarse leaf frequencies (Section 3.2.3)", "Boosting"]]
    out += [table(rows, [1.6, 4.3, 0.9]), P("Table 3.5. Selection evidence against the criteria in Models and Training "
                                           "Guidelines Section 6.", "caption")]
    out += [P(f"<b>Decision: histogram gradient boosting.</b> The rule fixed before the experiments was \"maximum mean "
              f"grouped-CV candidate Macro F1\" (<font name='Mono'>selection.json</font>; student-test results were not used "
              f"to select). Boosting wins under that rule. We do not overstate the margin over random forest: it is "
              f"{gap_:.4f}, about a fifth of one fold standard deviation, and per fold boosting is ahead in folds 0 and 2 "
              "while fold 1 is a tie to three decimals. With three folds a significance test would not be meaningful. The "
              "choice is still well supported, because every secondary criterion points the same way: lower log loss, "
              "better precision/recall balance (0.871/0.879 vs 0.893/0.849), and about 12 times faster fitting. Boosting was then tuned (Section 4).")]

    out += H2("3.2 Per-model analysis")
    out += [P("Each subsection answers the compulsory questions of Deliverables 3.2 using our measured results. Question "
              "numbers follow that document (A1-A2, B5-B6, C7-C9). Each figure shows the model's actual student-test "
              "confusion matrix and its Macro F1 on fit rows, grouped CV and student test.")]
    out += per_model(D, "logistic", "3.2.1", WHY["logistic"])
    out += per_model(D, "gaussian_nb", "3.2.2", WHY["gaussian_nb"])
    out += per_model(D, "decision_tree", "3.2.3", WHY["decision_tree"])
    out += per_model(D, "random_forest", "3.2.4", WHY["random_forest"])
    out += per_model(D, "hist_gradient_boosting", "3.2.5", WHY["hist_gradient_boosting"])

    out += H2("3.3 Model visualisations")
    out.append(figure(fig / "model_comparison.png", "Figure 3.1. Required comparisons. (a) Grouped-CV Macro F1 with the "
                      "standard deviation and each fold's score (dots); (b) student-test Top-1 (proxy) and candidate Macro "
                      "F1; (c) student-test log loss. Partitions: (a) train CV; (b, c) student test."))
    out.append(figure(fig / "model_quality_cost.png", "Figure 3.2. (a) Precision and recall of the positive class; (b) "
                      "Macro F1 on fit rows, grouped CV and student test, the over/underfitting diagnostic used in 3.2; "
                      "(c) fit and inference time (log scale)."))
    out.append(figure(fig / "confusion_matrices.png", "Figure 3.3. Student-test confusion matrices at threshold 0.5; each "
                      "cell shows the count and the share of its true-label row. Rows: proxy label; columns: prediction."))
    out.append(figure(fig / "model_per_trip.png", "Figure 3.4. Classifier-only Top-1 per student-test trip for every model. "
                      "Trip-to-trip variation (for example trip 00000077 for all models) is larger than model-to-model "
                      "variation, so model differences should be read across all trips, not from one example."))
    out += callout([
        "<b>Training and validation loss curves are deliberately absent.</b> Deliverables 3.3 requires them only for "
        "models that produce epoch-wise histories and forbids artificial ones. Logistic regression (L-BFGS), Naive Bayes "
        "(closed form), the decision tree and the random forest have no epochs. Histogram gradient boosting is iterative, "
        "but with early stopping disabled scikit-learn records no per-iteration validation loss, and no curve was "
        "logged during our runs. Producing one now would require re-running the experiment, so instead we show measured "
        "fit-vs-CV-vs-test comparisons (Figure 3.2b), which answer the same over/underfitting question.",
    ], title="About training curves")
    return out


WHY = {
    "logistic": dict(
        a1="We wanted a baseline that tests the simplest hypothesis: that a weighted sum of distance, heading agreement "
           "and the other features is enough to recognise the right arc. Logistic regression gives exactly that, with "
           "probabilities that come directly from its loss and can be used as HMM emissions without conversion.",
        a2s=["Second-fastest to fit (after Naive Bayes) and to apply (after the tree); only 14 parameters.",
             "Most stable across folds (lowest SD of the five).",
             "Interpretable: each weight shows how a feature pushes the score."],
        a2l=["One linear boundary cannot express interactions our data needs: heading should matter only when the "
             "vehicle moves, and distance matters relative to the other candidates. Result: the lowest student-test "
             "Top-1 and the lowest route continuity of the five.",
             "Precision 0.796: about 4,000 wrong candidates are accepted on student test."],
        b5=[["C (inverse L2 strength)", "1.0"], ["Penalty", "L2"], ["Solver", "L-BFGS"], ["max_iter", "800"],
            ["Scaling", "StandardScaler fitted inside each fold"], ["random_state", "357"]],
        b6="Binary cross-entropy (log loss) plus an L2 penalty on the weights, minimised by L-BFGS. With w the "
           "weights and C = 1: minimise ½‖w‖² + C · Σ log-loss(y<sub>i</sub>, "
           "σ(w·x<sub>i</sub> + b)).",
        c9="<b>Underfitting (high bias).</b> Fit, CV and student-test Macro F1 are almost identical, and training log "
           "loss is as high as test log loss. The model is not memorising; it lacks the capacity to represent the "
           "pattern. The tree models reach 0.92-0.93 on the same data, which shows the gap is the model's, not the "
           "data's."),
    "gaussian_nb": dict(
        a1="A generative probabilistic baseline from a different family. It asks how typical the feature values are "
           "for correct versus incorrect candidates, fitting one Gaussian per feature per class and combining them with "
           "Bayes' rule. It is extremely cheap and needs no iterative optimisation.",
        a2s=["Fastest to fit (a single pass computing class priors, means and variances).",
             "High recall: it finds 88% of correct candidates.",
             "Its Top-1 is higher than logistic regression's, because ranking candidates within one observation needs "
             "only the order of scores, not their calibration."],
        a2l=["It assumes features are independent given the class. Ours are not: distance and log-distance, length and "
             "log-length, heading and |heading| carry overlapping information. Naive Bayes multiplies the same evidence "
             "in several times, so its probabilities become overconfident.",
             "Observed effect: the worst precision (0.590, 11,726 false positives on student test) and by far the worst "
             "log loss (0.563). Those extreme probabilities would make poor HMM emissions."],
        b5=[["var_smoothing", "1e-7 (added fraction of the largest variance, for numerical stability)"],
            ["Priors", "Estimated from training class frequencies"],
            ["Scaling", "StandardScaler fitted inside each fold"]],
        b6="No loss is minimised. Fitting is maximum-likelihood estimation of the class priors and of the mean and "
           "variance of each feature within each class; prediction applies Bayes' rule, P(y | x) proportional to "
           "P(y) · Π<sub>j</sub> N(x<sub>j</sub>; μ<sub>yj</sub>, σ<sub>yj</sub>²).",
        c9="<b>Underfitting, plus poor calibration.</b> Fit and CV Macro F1 are the same and both are low; training log "
           "loss is already 0.456. The limitation is the independence assumption, not a lack of data, so more data "
           "would not fix it."),
    "decision_tree": dict(
        a1="A single tree learns threshold rules such as \"distance below 8 m and heading agreement above 0.9\". That "
           "is the first model in our set that can express interactions between features, which logistic regression "
           "cannot.",
        a2s=["Large jump over the linear models: CV Macro F1 0.921 vs 0.891.",
             "Fastest inference of all five (0.05 s for all ten trips).",
             "No scaling needed; rules are readable."],
        a2l=["Least stable across folds (highest SD, 0.012).",
             "Probabilities are leaf frequencies, so they are coarse and overconfident: out-of-fold log loss is 0.175, "
             "more than twice its training log loss (0.078). Poor input for an HMM, which multiplies probabilities "
             "over thousands of steps."],
        b5=[["Criterion", "Gini impurity"], ["max_depth", "10"], ["min_samples_leaf", "20"], ["random_state", "357"]],
        b6="Greedy recursive partitioning that, at each node, chooses the split maximising the decrease in Gini impurity "
           "1 − Σ<sub>k</sub> p<sub>k</sub>². Depth and leaf size limit growth.",
        c9="<b>Mild overfitting.</b> Macro F1 falls from 0.937 on fit rows to 0.921 in CV, and the overfitting shows "
           "most in the probabilities: log loss more than doubles out of fold. Student-test Macro F1 equals CV, so the "
           "decisions still generalise; the probabilities do not."),
    "random_forest": dict(
        a1="Averaging many decorrelated trees should keep the single tree's ability to model interactions while "
           "reducing its variance, which matters because labels are noisy proxies and trips differ.",
        a2s=["Second-best CV Macro F1 and best precision (0.893): it accepts the fewest wrong candidates.",
             "Averaging repairs the single tree's probabilities: out-of-fold log loss 0.096 vs 0.175."],
        a2l=["Lower recall (0.849) than boosting: it misses more correct candidates.",
             "Slowest to fit by far (24.3 s; about 14 s per CV fold)."],
        b5=[["n_estimators", "80"], ["max_depth", "14"], ["min_samples_leaf", "8"], ["max_features", "sqrt (default)"],
            ["Bootstrap", "yes (default)"], ["random_state / n_jobs", "357 / 4"]],
        b6="Each tree minimises Gini impurity on a bootstrap sample with a random feature subset at every split; the "
           "forest's probability is the mean of the trees' leaf class proportions.",
        c9="<b>Mild overfitting, the largest fit-to-CV gap of the five</b> (0.947 vs 0.926 Macro F1; log loss 0.068 "
           "on fit rows vs 0.092 on student test). This is expected for deep bagged trees, which fit training rows "
           "closely. Student test matches CV (0.925 vs 0.926), so generalisation holds."),
    "hist_gradient_boosting": dict(
        a1="Boosting builds shallow trees one after another, each correcting the errors left by the previous ones. "
           "Our errors are concentrated in specific situations (junctions, twins, stops), which is what sequential "
           "error correction targets. Histogram binning makes it fast on 255,000 rows on a CPU.",
        a2s=["Best of the five on CV Macro F1, student-test Macro F1, Top-1, log loss, arc Macro F1, LCS F1 and "
             "route continuity. The decision tree is marginally ahead on route overlap F1 (0.9322 vs 0.9319) and the "
             "random forest on precision.",
             "Balanced precision and recall (0.871 / 0.879).",
             "Fits in under 2 s and gives a 108 KB model."],
        a2l=["Slowest classifier inference (0.88 s for ten trips), though still small next to decoding.",
             "Like all models here, it learns from proxy labels, so it inherits their errors.",
             "Its margin over random forest is small (Section 3.1.1)."],
        b5=[["max_iter", "120"], ["learning_rate", "0.1"], ["max_leaf_nodes", "15"], ["l2_regularization", "1.0"],
            ["min_samples_leaf", "30"], ["early_stopping", "False (avoids row-random validation leakage)"],
            ["random_state", "357"]],
        b6="Binary log loss, minimised by gradient boosting: each iteration fits a small tree to the gradient of the "
           "log loss with respect to the current scores, on features binned into histograms, with L2 regularisation of "
           "leaf values.",
        c9="<b>Slight overfitting, smaller than the random forest's.</b> Macro F1 0.940 on fit rows vs 0.928 in CV and "
           "0.927 on student test; log loss 0.078 vs 0.089. Student test matches CV, so the model generalises to "
           "unseen trips. The tuned version used in the final system is analysed in Section 4."),
}


def per_model(D, key, number, why):
    x = D["models"][key]
    y = x["student_test"]["summary"]
    out = H2(f"{number} {NAMES[key]}")
    out += question("A1", "Why did you consider this algorithm suitable for the PathSnap problem?")
    out += [P(why["a1"])]
    out += question("A2", "What are the major strengths and limitations of this algorithm for this task?")
    out += [P("<b>Strengths observed in our experiments:</b>", "small")] + bullets(why["a2s"])
    out += [P("<b>Limitations observed:</b>", "small")] + bullets(why["a2l"])
    out += question("B5", "What were the important hyperparameters and their initial values?")
    out += [table([["Hyperparameter", "Initial value used"]] + why["b5"], [2, 4]),
            P("These are the values used in the comparison (pathsnap/models.py, MODEL_SPECS). They were chosen as "
              "reasonable defaults for about 255,000 rows and 13 features, before any results were seen.", "caption")]
    out += question("B6", "What loss function or optimisation objective was used?")
    out += [P(why["b6"])]
    out += question("C7 / C8", "Performance on the student-testing dataset: observation-level Top-1, Macro F1 and log loss")
    rows = [["Measure (student test, 19,160 observations)", "Value"],
            ["Observation-level Top-1 accuracy (proxy)", f(y["proxy_top1_accuracy"])],
            ["Candidate Macro F1", f(y["candidate_macro_f1"])],
            ["Candidate log loss", f(y["candidate_log_loss"])],
            ["Observation-level proxy log loss", f(y["observation_proxy_log_loss"])],
            ["Accuracy / precision / recall", f"{f(y['candidate_accuracy'])} / {f(y['candidate_precision'])} / "
                                              f"{f(y['candidate_recall'])}"],
            ["Arc Macro F1 (mean of trips)", f(y["mean_trip_proxy_arc_macro_f1"])],
            ["Route continuity / route overlap F1 / LCS F1", f"{f(y['mean_trip_direct_arc_continuity'])} / "
                                                             f"{f(y['mean_trip_route_length_f1'])} / "
                                                             f"{f(y['mean_trip_ordered_route_lcs_f1'])}"],
            ["Classifier inference, all 10 trips", f"{y['model_inference_seconds']:.3f} s"]]
    out += [table(rows, [4, 2]), gap_par()]
    out += question("C9", "Did you observe overfitting or underfitting? What evidence supports your conclusion?")
    rows = [["Evidence", "Fit rows", "Grouped CV", "Student test"],
            ["Candidate Macro F1", f(x["training"]["candidate_macro_f1"]), f(x["cv_candidate_macro_f1"]),
             f(y["candidate_macro_f1"])],
            ["Candidate log loss", f(x["training"]["candidate_log_loss"]), f(x["oof"]["candidate_log_loss"]) + " (OOF)",
             f(y["candidate_log_loss"])],
            ["Fold standard deviation of Macro F1", "-", f(x["cv_candidate_macro_f1_std"]), "-"]]
    out += [table(rows, [2.4, 1.2, 1.4, 1.2]), gap_par(), P(why["c9"])]
    out.append(figure(ROOT / f"reports/plots/{key}.png", f"Figure {number}. {NAMES[key]}: student-test confusion matrix "
                      "(left) and Macro F1 on fit rows, grouped CV and student test (right).", width=0.78))
    return out


def gap_par():
    from reportlab.platypus import Spacer
    return Spacer(1, 4)
