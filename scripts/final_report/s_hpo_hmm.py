"""Section 4 (Bayesian hyperparameter optimisation) and Section 5 (HMM and Viterbi)."""

import json

from .data import NAMES, f, n, pct
from .layout import H1, H2, H3, P, bullets, callout, eq, figure, numbered, question, table


def tuning(D, fig):
    tun, trials = D["tuning"], D["trials"]
    unt = tun["untuned_student_test"]["summary"]
    tst = tun["tuned_student_test"]["summary"]
    best = max(trials, key=lambda t: t["value"])
    out = H1("4. Bayesian hyperparameter optimisation")
    out += [P(f"<b>Model tuned:</b> histogram gradient boosting, selected in Section 3.1.1 by the pre-set CV rule. "
              f"<b>Tool:</b> Optuna {D['manifest']['optuna_version']}, Tree-structured Parzen Estimator (TPE) sampler, "
              f"seed {D['config']['seed']}. <b>Data:</b> the 80 training trips only, with the same three whole-trip folds "
              "used for the model comparison. Neither student-test nor reserved data entered the search.")]
    out += callout([
        "Grid and random search ignore what earlier trials found. TPE keeps a record of every trial and splits it into "
        "\"good\" trials (the best-scoring fraction) and the rest. It fits one density l(x) to the hyperparameters of good "
        "trials and another, g(x), to the rest, then proposes the next setting where l(x)/g(x) is large: settings that "
        "look like past successes and unlike past failures. The first trials are random, to give the densities something "
        "to fit. This is Bayesian optimisation because each new proposal is conditioned on the evidence gathered so far.",
    ], title="How TPE works, briefly")
    out += H2("4.1 Search space")
    out += question("Q1", "Which hyperparameters were included in the search?")
    out += [P("Four parameters that together set the capacity/regularisation balance of boosted trees. Section 3.2.5 found "
              "slight overfitting, so the search is mainly about how much complexity the model can afford:")]
    out += bullets([
        "<b>learning_rate</b>: how far each new tree moves the predictions. Smaller steps generalise better but need more "
        "trees.",
        "<b>max_leaf_nodes</b>: complexity of each tree, i.e. how many feature interactions one tree can express.",
        "<b>l2_regularization</b>: shrinks leaf values, damping trees that fit a few rows strongly.",
        "<b>min_samples_leaf</b>: the smallest number of rows a leaf may describe, which stops trees memorising small "
        "groups of near-identical points from a single trip.",
    ])
    out += [P("<b>Deliberately not searched.</b> max_iter stayed at 120: early stopping is disabled to avoid leakage, so "
              "the number of trees cannot be set automatically, and fixing it keeps trial cost predictable. A consequence "
              "is that learning_rate also controls the total amount of fitting. Searching both together would have been "
              "better and is noted as a limitation. Class weights were not searched because reweighting classes distorts "
              "probabilities, which the HMM uses directly.")]
    out += question("Q2", "What ranges or distributions were considered?")
    rows = [["Hyperparameter", "Distribution", "Range", "Untuned value", "Reason for the range"],
            ["learning_rate", "log-uniform (float)", "0.025 to 0.2", "0.1", "Step sizes span an order of magnitude; a log "
             "scale gives equal attention to 0.03 vs 0.06 and 0.1 vs 0.2"],
            ["max_leaf_nodes", "uniform (integer)", "7 to 31", "15", "From very shallow trees to about twice the default"],
            ["l2_regularization", "log-uniform (float)", "0.01 to 10", "1.0", "Three orders of magnitude around the default"],
            ["min_samples_leaf", "uniform (integer)", "15 to 100", "30", "From fine-grained to strongly smoothed leaves"]]
    out += [table(rows, [1.4, 1.2, 0.9, 0.9, 3.4]), P("Table 4.1. Search space (pathsnap/models.py, search_parameters; "
                                                      "distributions are stored with every trial in bayesian_trials.json).",
                                                      "caption")]
    out += H2("4.2 Objective and trials")
    out += question("Q3", "What objective function was optimised?")
    out += [P("Maximise the <b>mean candidate Macro F1 over the three grouped CV folds</b> of the 80 training trips, at the "
              "0.5 threshold: the same quantity used to select the model, so tuning optimises what selection measured. Each "
              "fold's score is stored with the trial.")]
    out += question("Q4", "How many optimisation trials were performed?")
    rows = [["Trial", "Phase", "learning_rate", "max_leaf_nodes", "l2_regularization", "min_samples_leaf",
             "Mean CV Macro F1", "Seconds"]]
    for t in trials:
        p = t["parameters"]
        rows.append([str(t["number"]), "random startup" if t["number"] < tun["startup_random_trials"] else "TPE",
                     f"{p['learning_rate']:.4f}", str(p["max_leaf_nodes"]), f"{p['l2_regularization']:.4f}",
                     str(p["min_samples_leaf"]), f(t["value"]), f"{t['duration_seconds']:.1f}"])
    out += [table(rows, [.5, 1.1, 1, 1, 1.1, 1.1, 1.1, .7], highlight_rows=(best["number"] + 1,)),
            P(f"Table 4.2. All {tun['trials']} trials: {tun['startup_random_trials']} random startup trials, then "
              f"{tun['adaptive_tpe_trials']} adaptive TPE trials. The highlighted row is the best.", "caption")]
    out.append(figure(fig / "bayes_history.png", "Figure 4.1. Optimisation history. The best value appears at trial 5, "
                      "the second adaptive trial; later TPE trials stay close to it."))
    out.append(figure(fig / "bayes_parameters.png", "Figure 4.2. Objective against each hyperparameter. The best region "
                      "has a moderate learning rate (about 0.03 to 0.06), mid-sized trees (16 to 24 leaves) and "
                      "min_samples_leaf of about 50 to 80; the largest learning rate tested (0.20) scored lowest."))
    out += H2("4.3 Best configuration")
    out += question("Q5", "What were the best hyperparameter values obtained?")
    bp = tun["best_parameters"]
    out += [table([["Hyperparameter", "Untuned", "Best (trial %d)" % best["number"]],
                   ["learning_rate", "0.1", f"{bp['learning_rate']:.4f}"],
                   ["max_leaf_nodes", "15", str(bp["max_leaf_nodes"])],
                   ["l2_regularization", "1.0", f"{bp['l2_regularization']:.4f}"],
                   ["min_samples_leaf", "30", str(bp["min_samples_leaf"])],
                   ["max_iter (fixed)", "120", "120"]], [2, 1.5, 1.5]),
            P("Table 4.3. Best configuration, refitted on all 80 training trips and saved as the final classifier "
              "(artifacts/models/final.joblib).", "caption")]
    out += [P("The tuned model learns more slowly (learning rate 0.041 instead of 0.1) with larger leaves "
              "(minimum 60 rows instead of 30) and slightly more leaves per tree. In plain terms, it trades step size for "
              "smoother, more conservative trees, which is consistent with the mild overfitting seen before tuning.")]
    out += H2("4.4 Tuned versus untuned")
    out += question("Q6", "Did the tuned model outperform the untuned model?")
    rows = [["Metric (same partitions for both)", "Untuned", "Tuned", "Change"],
            ["CV candidate Macro F1 (train, 3 folds)", f"{f(tun['untuned_cv_macro_f1'])}",
             f"{f(tun['cv_candidate_macro_f1'])}", f"{tun['cv_candidate_macro_f1'] - tun['untuned_cv_macro_f1']:+.4f}"],
            ["CV fold standard deviation", f(D["models"]["hist_gradient_boosting"]["cv_candidate_macro_f1_std"]),
             f(tun["cv_candidate_macro_f1_std"]),
             f"{tun['cv_candidate_macro_f1_std'] - D['models']['hist_gradient_boosting']['cv_candidate_macro_f1_std']:+.4f}"],
            ["Student-test Top-1 accuracy (proxy)", f(unt["proxy_top1_accuracy"]), f(tst["proxy_top1_accuracy"]),
             f"{tst['proxy_top1_accuracy'] - unt['proxy_top1_accuracy']:+.4f}"],
            ["Student-test candidate Macro F1", f(unt["candidate_macro_f1"]), f(tst["candidate_macro_f1"]),
             f"{tst['candidate_macro_f1'] - unt['candidate_macro_f1']:+.4f}"],
            ["Student-test log loss", f(unt["candidate_log_loss"]), f(tst["candidate_log_loss"]),
             f"{tst['candidate_log_loss'] - unt['candidate_log_loss']:+.4f}"],
            ["Student-test arc Macro F1 (mean of trips)", f(unt["mean_trip_proxy_arc_macro_f1"]),
             f(tst["mean_trip_proxy_arc_macro_f1"]),
             f"{tst['mean_trip_proxy_arc_macro_f1'] - unt['mean_trip_proxy_arc_macro_f1']:+.4f}"],
            ["Classifier inference, 10 student-test trips", f"{unt['model_inference_seconds']:.3f} s",
             f"{tst['model_inference_seconds']:.3f} s", f"{tst['model_inference_seconds'] - unt['model_inference_seconds']:+.3f} s"],
            ["Final fit time", f"{D['models']['hist_gradient_boosting']['full_fit_seconds']:.2f} s",
             f"{tun['tuned_fit_seconds']:.2f} s", "-"]]
    out += [table(rows, [3, 1.1, 1.1, 1]), P("Table 4.4. Untuned vs tuned histogram gradient boosting. Source: "
                                             "tuning.json.", "caption")]
    rng = (min(t["value"] for t in trials), max(t["value"] for t in trials))
    out += [P(f"<b>Answer: only on the objective, and only marginally.</b> The tuned model improves mean CV Macro F1 by "
              f"{tun['cv_candidate_macro_f1'] - tun['untuned_cv_macro_f1']:+.4f} and reduces fold variability slightly. On "
              "the student-test trips it is equal within noise: Top-1 and Macro F1 drop by under 0.002, arc Macro F1 rises "
              "by 0.003, log loss is 0.001 worse, and inference is slightly faster. The improvement is far smaller than one "
              f"fold standard deviation (about 0.009), and all twelve trials scored between {f(rng[0])} and {f(rng[1])}. The "
              "objective is flat over this search space: the untuned defaults were already near a plateau, and the "
              "remaining errors come from information the features do not contain (for example ambiguous junctions and "
              "label noise), not from hyperparameters.")]
    out += [P("We kept the tuned model because the rule fixed in advance was to deploy it if its CV score was at least the "
              "untuned score (experiments.py). Choosing between the two using student-test results would have been "
              "selection on development data, and in any case the difference is negligible. Bigger gains require new "
              "information rather than more tuning; Section 9 tests exactly that by adding context features.")]
    out += H2("4.5 Computational cost")
    durations = [t["duration_seconds"] for t in trials]
    out += [P(f"The search took <b>{tun['optimisation_seconds']:.1f} s</b> on a CPU laptop (4 threads): "
              f"{sum(durations) / len(durations):.1f} s per trial on average (range {min(durations):.1f} to "
              f"{max(durations):.1f} s), each trial fitting three folds. The final refit took "
              f"{tun['tuned_fit_seconds']:.2f} s. No GPU was used. Twelve trials is a small budget; we chose it because each "
              "trial is cheap but the flat history (Figure 4.1) suggests a larger budget would not change the conclusion. "
              "Because the same folds were used to select the family and to tune it, the tuned CV score is slightly "
              "optimistic; the student-test comparison is the independent check.")]
    return out


def hmm(D, fig):
    dev, ev = D["dev_summary"], D["ev_summary"]
    dc = D["final"]["config"]["decoder"]
    out = H1("5. HMM and Viterbi route reconstruction")
    out += H2("5.1 Why sequential decoding is needed")
    out += [P("The classifier judges each GPS point on its own. Choosing the best candidate independently for each point "
              "produces routes that jump: consecutive points land on arcs that do not touch, or flip between a road and "
              "its twin. Measured on student test, only "
              f"{pct(dev['classifier_direct_arc_continuity'], 1)} of consecutive classifier-only arc changes connect "
              "end-to-start (mean over trips). A vehicle cannot do that. A hidden Markov model adds what the classifier "
              "lacks: consecutive positions must be linked by drivable road, and the road distance between them should "
              "agree with how far the GPS moved.")]
    out += H2("5.2 Hidden states")
    out += question("Q1", "What are the hidden states in your implementation?")
    out += [P(f"At observation t the hidden state is <b>which of that observation's candidates (at most "
              f"{D['config']['candidate_count']}) the vehicle is on</b>. Each state carries the candidate's arc ID and the "
              "fraction along the arc where the GPS point projects. Because a state is a slot in the observation's own "
              "candidate list, the candidate-to-arc correspondence is preserved exactly: decoding returns a candidate index "
              "per observation, which maps back to an arc ID. Padding slots (fewer than eight candidates) are invalid "
              "states with zero probability. The state space changes from observation to observation, which a standard "
              "fixed-state HMM does not allow, so the recurrence is written over these per-observation lattices.")]
    out += H2("5.3 Emission probabilities")
    out += question("Q2", "How are emission probabilities calculated?")
    out.append(eq("e<sub>t</sub>(k) = p<sub>t,k</sub> / Σ<sub>j valid</sub> p<sub>t,j</sub>, &nbsp;&nbsp; "
                  "p<sub>t,k</sub> = max(classifier probability, 10<super>−12</super>)"))
    out += [P("The selected classifier's probability that candidate k is correct is floored at 10<super>−12</super> "
              "(so its logarithm is finite) and normalised over that observation's valid candidates. Emission and "
              "classifier therefore use the same model output, as Models Guidelines Section 8 asks. Strictly, these are "
              "discriminative scores used as emission potentials, not a fitted density P(GPS | arc).")]
    out += callout([
        "Normalising over an observation's candidates subtracts the same constant from every state's log-emission at that "
        "time step. Every complete path passes through exactly one state per time step, so every path's total score "
        "changes by the same amount and the Viterbi winner cannot change. The same holds for the hybrid HMM/neural-network "
        "practice of dividing posteriors by the class prior. Normalisation is there to make scores interpretable and "
        "comparable, not to change the decoded route. What does matter is the <i>relative scale</i> of emission and "
        "transition terms, which beta controls.",
    ], title="A useful property for the viva")
    out += H2("5.4 Transition probabilities")
    out += question("Q3", "How are transition probabilities calculated?")
    out += [P("For candidate i at t−1 (arc a, projection fraction f<sub>a</sub>) and candidate j at t (arc b, "
              "fraction f<sub>b</sub>), the <b>directed network distance</b> is the remaining part of arc a, plus the "
              "shortest directed path from a's end node to b's start node, plus the used part of arc b:")]
    out.append(eq("d<sub>net</sub>(i, j) = (1 − f<sub>a</sub>) L<sub>a</sub> + SP(head(a), tail(b)) + "
                  "f<sub>b</sub> L<sub>b</sub>"))
    out += [P("If both candidates are on the same arc and the second projection is ahead of the first, the distance is "
              "simply (f<sub>b</sub> − f<sub>a</sub>) L<sub>a</sub>. A backward step of up to 3 m on the same arc is "
              "treated as stationary GPS jitter (distance 0); a larger backward step requires a real loop through the "
              "network. The log transition score compares that road distance with the straight-line GPS displacement "
              "between the two observations:")]
    out.append(eq(f"log T(i → j) = − | d<sub>net</sub>(i, j) − ‖g<sub>t</sub> − "
                  f"g<sub>t−1</sub>‖ | / β,&nbsp;&nbsp;&nbsp; β = {dc['beta_m']:.0f} m"))
    out += [P(f"This is the exponential transition model of Newson and Krumm (2009). Between consecutive 1 Hz fixes the "
              "road distance and the straight-line distance should be nearly equal; a large difference means a detour "
              "the GPS does not support. Transitions are <b>impossible</b> (log T = −∞) when no directed path "
              f"exists within the bound {dc['max_speed_mps']:.0f} m/s × Δt + {dc['routing_slack_m']:.0f} m "
              "(about 200 km/h plus slack), so U-turns against one-way streets and teleports are excluded. Shortest "
              "paths use SciPy's Dijkstra from each source node with that distance limit, cached per (node, distance "
              "bucket), because shortest-path search dominates decoding time.")]
    out += [P("<b>Global scale, not row normalisation.</b> The first implementation normalised each state's outgoing "
              "transitions to sum to 1 over the next observation's candidates. That looks more \"probabilistic\" but adds a "
              "bias that has nothing to do with the GPS: a previous state whose successors are all poor still hands out "
              "full probability, while one with several good successors must divide it. The Newson-Krumm form has the "
              "same normalising constant 1/β for every transition, so it cannot introduce that bias. We compared both "
              "forms with β of 5, 25 and 100 m on the student-test trips (Figure 5.1). The globally scaled form with "
              "β = 25 m scored best and was adopted.")]
    out.append(figure(fig / "decoder_search.png", "Figure 5.1. Decoder settings compared by pooled proxy Top-1 on the "
                      "student-test trips (decoder_search.json). Grey: row-normalised; green: global scale. Partition: "
                      "student test, which the guidelines permit for decoder tuning; it is therefore not an untouched "
                      "estimate of decoder performance."))

    out += H2("5.5 The Viterbi recurrence")
    out.append(eq("δ<sub>0</sub>(k) = log e<sub>0</sub>(k), &nbsp;&nbsp; δ<sub>t</sub>(j) = log e<sub>t</sub>(j) + "
                  "max<sub>i</sub> [ δ<sub>t−1</sub>(i) + log T(i → j) ], &nbsp;&nbsp; "
                  "ψ<sub>t</sub>(j) = argmax<sub>i</sub> [ … ]"))
    out += [P("δ<sub>t</sub>(j) is the log-score of the best path that ends in candidate j at time t; "
              "ψ<sub>t</sub>(j) remembers which previous candidate achieved it. After the last observation of a "
              "segment we take the best final state and follow the ψ pointers backwards to recover one candidate per "
              "observation. Working in logarithms turns products of thousands of small probabilities into sums and "
              "avoids numerical underflow; δ is also shifted by its maximum at every step. Each step costs "
              "O(K<sup>2</sup>) with K ≤ 8, plus the shortest-path queries. A unit test checks the implementation "
              "against brute-force enumeration of every path on small examples, including resets.")]
    out += callout([
        "<b>Task.</b> Decide between two candidates at time t. <b>Assumptions (invented values).</b> Two consecutive "
        "observations 1 s apart; the GPS moved 20 m. At t−1: candidates A1 and A2 with normalised emissions 0.7 and "
        "0.3, so δ<sub>t−1</sub> = ln 0.7 = −0.357 and ln 0.3 = −1.204. At t: candidates B1 and B2 "
        "with emissions 0.45 and 0.55, so the classifier alone would pick B2. Road distances: A1→B1 = 22 m, "
        "A1→B2 = 80 m, A2→B2 = 21 m, and no legal path A2→B1. β = 25 m; routing bound "
        "55 × 1 + 100 = 155 m, so all three paths are allowed.",
        "<b>Transitions.</b> log T(A1→B1) = −|22 − 20|/25 = −0.08; log T(A1→B2) = "
        "−|80 − 20|/25 = −2.40; log T(A2→B2) = −|21 − 20|/25 = −0.04; "
        "log T(A2→B1) = −∞.",
        "<b>Recurrence.</b> δ<sub>t</sub>(B1) = ln 0.45 + max(−0.357 − 0.08, −∞) = −0.799 "
        "− 0.437 = <b>−1.236</b>, with ψ = A1. δ<sub>t</sub>(B2) = ln 0.55 + max(−0.357 "
        "− 2.40, −1.204 − 0.04) = −0.598 − 1.244 = <b>−1.842</b>, with ψ = A2.",
        "<b>Result.</b> B1 wins although the classifier preferred B2: B1 is reached from the likely previous state by a "
        "road distance that matches the GPS movement, while B2 needs either an 80 m detour or the unlikely state A2.",
    ], title="Worked example of one Viterbi step")

    out += H2("5.6 Missing candidates and large GPS gaps")
    out += question("Q4", "How are missing candidates or large GPS gaps handled?")
    out += bullets([
        f"<b>Reference arc missing from the candidate set</b> ({pct(dev['miss'])} of student-test observations, "
        f"{pct(ev['miss'])} on reserved trips): the decoder cannot choose it and the observation counts as wrong. We do "
        "not repair it, since that would need the reference.",
        "<b>No arc within 100 m</b>: the observation receives the nearest arcs from a k-nearest search, flagged as a "
        "fallback (0 student-test observations). The API reports the number of fallback observations.",
        "<b>Fewer than eight candidates</b>: unused slots are padding with zero emission, so they can never be chosen.",
        f"<b>GPS gap longer than {dc['gap_reset_s']:.0f} s</b>: there is no transition across the gap. Decoding ends the "
        "current segment, backtracks it, and starts a new independent segment from the next observation's emissions. "
        "Within shorter gaps the routing bound grows with Δt, so a 30 s gap allows a longer drive than a 1 s step. "
        "Nothing is interpolated, and the reconstructed route does not bridge across a reset.",
        "Each reset is recorded with its reason (for example \"GPS gap 210 s exceeds 60 s\") and returned by the API and "
        "drawn on the map.",
    ])
    out += H2("5.7 Disconnected road segments and route reconstruction")
    out += question("Q5", "What happens when consecutive GPS observations suggest disconnected road segments?")
    out += [P("If, for some step, no pair of candidates is connected by a directed path within the routing bound, every "
              "δ<sub>t</sub> becomes −∞. The decoder then finishes and backtracks the segment so far, marks a "
              "reset with the reason \"No directed path survives within the speed and routing bound\", and restarts from "
              f"observation t. Student-test decoding produced {dev['resets']} resets across 10 trips; the reserved trips "
              f"produced {ev['resets']}, of which {D['ev']['requests'][0]['metrics']['decoder_resets']} were in trip "
              "00000015.")]
    out += [P("<b>Route reconstruction.</b> Decoding gives one arc per observation. To turn that into a drivable route, "
              "consecutive distinct arcs are joined with the arcs of the actual shortest directed path between them (a "
              "<i>bridge</i>). If the same arc appears twice but the vehicle has moved backwards along it by more than 3 m, "
              "a loop through the network is inserted, because a vehicle on a directed arc cannot reverse. If no bridge "
              "exists within the bound, a new route piece starts. The output therefore has two parts: "
              "<font name='Mono'>predicted_arc_ids</font>, one arc for every GPS observation, and "
              "<font name='Mono'>route_pieces</font>, connected arc sequences separated only at genuine breaks. A straight "
              "line is never drawn across a break.")]

    out += H2("5.8 Did decoding improve the route?")
    out += question("Q6", "Did HMM decoding improve the final route? Support your answer with experimental results.")
    rows = [["Student test (development)", "Classifier only", "HMM states", "HMM + bridges"],
            ["Pointwise Top-1 (proxy, pooled)", pct(dev["classifier_top1"]), pct(dev["hmm_top1"]),
             "N/A: bridge arcs have no GPS observation"],
            ["Route overlap F1 (mean of trips)", f(dev["classifier_route_length_f1"]), f(dev["hmm_route_length_f1"]),
             f(dev["bridged_route_route_length_f1"])],
            ["Route precision / recall", f"{f(dev['classifier_route_length_precision'], 3)} / "
             f"{f(dev['classifier_route_length_recall'], 3)}",
             f"{f(dev['hmm_route_length_precision'], 3)} / {f(dev['hmm_route_length_recall'], 3)}",
             f"{f(dev['bridged_route_route_length_precision'], 3)} / {f(dev['bridged_route_route_length_recall'], 3)}"],
            ["Ordered route LCS F1", f(dev["classifier_ordered_route_lcs_f1"]), f(dev["hmm_ordered_route_lcs_f1"]),
             f(dev["bridged_route_ordered_route_lcs_f1"])],
            ["Direct arc continuity", f(dev["classifier_direct_arc_continuity"]), f(dev["hmm_direct_arc_continuity"]),
             f(dev["bridged_route_direct_arc_continuity"])]]
    out += [table(rows, [2.4, 1.3, 1.3, 1.6]), P(f"Table 5.1. Student test, {n(dev['observations'])} observations, 10 trips. "
                                                 "Source: decoder_comparison.json.", "caption")]
    out.append(figure(fig / "decoder_dev.png", "Figure 5.2. Student test per trip: classifier only (blue), decoded HMM "
                      "states (green), and the bridged route (gold)."))
    out += [P(f"<b>Where decoding helped (development).</b> The route becomes connected: direct continuity rises from "
              f"{f(dev['classifier_direct_arc_continuity'], 3)} to {f(dev['hmm_direct_arc_continuity'], 3)} from decoding "
              f"alone, and to {f(dev['bridged_route_direct_arc_continuity'], 3)} with bridges. Ordered agreement with the "
              f"reference sequence rises (LCS F1 {f(dev['classifier_ordered_route_lcs_f1'], 3)} to "
              f"{f(dev['hmm_ordered_route_lcs_f1'], 3)}), and length-weighted overlap rises on "
              f"{len(dev['f1_up'])} of 10 trips. Decoding raises route precision "
              f"({f(dev['classifier_route_length_precision'], 3)} to {f(dev['hmm_route_length_precision'], 3)}) because "
              "it stops the route wandering onto short side arcs; bridging then raises recall (to "
              f"{f(dev['bridged_route_route_length_recall'], 3)}) by filling in the arcs between observations. Pointwise accuracy improved on trips {', '.join(t[-2:] for t in dev['top1_up'])}.")]
    out += [P(f"<b>Where it did not help.</b> Pointwise proxy accuracy fell overall ({pct(dev['classifier_top1'])} to "
              f"{pct(dev['hmm_top1'])}) and on trips {', '.join(t[-2:] for t in dev['top1_down'])}; trip 00000007 lost 8 "
              "points (0.928 to 0.849) even though its route overlap and LCS improved. Bridging also lowers LCS slightly "
              f"(to {f(dev['bridged_route_ordered_route_lcs_f1'], 3)}), because shortest-path fill-ins sometimes differ "
              "from the reference's chosen path between the same arcs. Trip 00000076 is the one trip where route overlap "
              "fell after bridging.")]
    rows = [["Reserved evaluator trips (final)", "Classifier only", "HMM states", "HMM + bridges"],
            ["Pointwise Top-1 (proxy, pooled)", pct(ev["classifier_top1"]), pct(ev["hmm_top1"]),
             "N/A: bridge arcs have no GPS observation"],
            ["Route overlap F1 (mean of trips)", f(ev["classifier_route_length_f1"]), f(ev["hmm_route_length_f1"]),
             f(ev["bridged_route_route_length_f1"])],
            ["Ordered route LCS F1", f(ev["classifier_ordered_route_lcs_f1"]), f(ev["hmm_ordered_route_lcs_f1"]),
             f(ev["bridged_route_ordered_route_lcs_f1"])],
            ["Direct arc continuity", f(ev["classifier_direct_arc_continuity"]), f(ev["hmm_direct_arc_continuity"]),
             f(ev["bridged_route_direct_arc_continuity"])]]
    out += [table(rows, [2.4, 1.3, 1.3, 1.6]), P(f"Table 5.2. Reserved trips, {n(ev['observations'])} observations, scored "
                                                 "once through the frozen API. Source: evaluator_results.json.", "caption")]
    out.append(figure(fig / "decoder_reserved.png", "Figure 5.3. Reserved trips per trip. Trip 00000015 dominates the "
                      "pointwise drop."))
    t15 = D["ev"]["requests"][0]["metrics"]
    out += [P(f"On the reserved trips continuity and overlap again improve, but pointwise accuracy falls by "
              f"{100 * (ev['classifier_top1'] - ev['hmm_top1']):.1f} points. Trip 00000015 (6,854 points, a third of the "
              f"partition) falls from {f(t15['classifier']['proxy_top1_accuracy'], 3)} to "
              f"{f(t15['hmm']['proxy_top1_accuracy'], 3)} with {t15['decoder_resets']} resets. Its reference looks "
              f"normal (median GPS-to-reference distance {t15['proxy_label_distance_median_m']:.1f} m), and the "
              f"classifier-only route overlap is already poor ({f(t15['classifier']['route_length_f1'], 3)}), so this "
              "trip is hard for the whole system, not only for the decoder. Pointwise accuracy improved on trips "
              f"{', '.join(t[-2:] for t in ev['top1_up'])} and fell on {', '.join(t[-2:] for t in ev['top1_down'])}.")]
    out.append(figure(fig / "reserved_all_models.png", "Figure 5.4. Report-only check after the final evaluation: every "
                      "model family run through the same frozen decoder on the reserved trips "
                      "(evaluator_model_comparison.json). This comparison selected nothing."))
    out += [P("This report-only comparison shows the pointwise drop for every family, by about 0.09 to 0.11, so the drop "
              "comes from the decoder's behaviour on these trips rather than from the chosen classifier. We have not "
              "established the cause. Plausible explanations, <i>none of them tested</i>: (i) proxy labels come from a "
              "geometric alignment that, like the classifier, favours the geometrically nearest arc, while the HMM prefers "
              "network-consistent arcs, so near junctions the two can disagree about which of two touching arcs a point "
              "belongs to without the route changing much; (ii) frequent resets leave short segments in which the decoder "
              "has little context; (iii) β and the routing bound were chosen from six settings on ten development "
              "trips and may not transfer.")]
    out += callout([
        "<b>Yes for the route, no for proxy pointwise accuracy.</b> Decoding turns a fragmented set of independent "
        "guesses into a connected, drivable route that overlaps the reference better on both partitions. It does not "
        "improve agreement with the inferred per-point labels, and on the reserved trips it clearly lowers it. Because the "
        "deliverable is a reconstructed route and the decoder settings were fixed before the reserved data was opened, we "
        "keep the decoder and report the trade-off rather than retuning on final results.",
    ], title="Answer to Q6")
    out += H2("5.9 Runtime")
    out.append(figure(fig / "timings.png", "Figure 5.5. Prediction time per student-test trip by stage "
                      "(decoder_comparison.json). Decoding with shortest-path queries is the largest stage, as expected."))
    out += [P(f"Mean prediction time is {sum(D['dev_seconds']) / len(D['dev_seconds']):.2f} s per student-test trip "
              f"(maximum {max(D['dev_seconds']):.2f} s for 3,958 points). Over HTTP, including reference scoring and map "
              f"rendering, the reserved trips took {sum(D['ev_http']) / len(D['ev_http']):.2f} s on average and at most "
              f"{max(D['ev_http']):.2f} s (trip 00000015, 6,854 points).")]
    return out
