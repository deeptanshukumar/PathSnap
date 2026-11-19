"""Cover page, reading guide, contents and Section 1 (overview)."""

from datetime import date

from reportlab.platypus import NextPageTemplate, PageBreak, Spacer, Table, TableStyle

from .data import f, n, pct
from .layout import (GOLD, INK, S, TEAL, TEXT_W, H1, H2, P, bullets, callout, figure, gap, table, toc)


def cover(D, fig):
    team = [["Team member", "SRN"], ["Chiranth R", "PES1UG24AM354"], ["Darshan S Govekar", "PES1UG24AM355"],
            ["Daruru Bhuvan Sai", "PES1UG24AM356"], ["Deeptanshu Kumar", "PES1UG24AM357"]]
    out = [Spacer(1, 2.2 * 72), P("PathSnap", "title"), gap(6),
           P("Distributed map matching and route reconstruction: a learned candidate scorer with "
             "HMM / Viterbi decoding", "subtitle"), gap(16),
           P("<b>Final consolidated technical report</b>", "h2"),
           P("UE24CS352A Machine Learning &nbsp;|&nbsp; PathSnap hackathon &nbsp;|&nbsp; PES University, RR Campus"),
           P(f"Team PT-088 &nbsp;|&nbsp; Report generated {date.today():%d %B %Y}"), gap(14),
           table(team, [3, 2], zebra=True), gap(16),
           P("Repository: <font name='Mono'>https://github.com/deeptanshukumar/PathSnap</font>"),
           P("Regenerate this PDF: <font name='Mono'>python scripts/build_final_report.py</font> "
             "(needs only the committed artifacts, not the raw dataset)."),
           gap(16)]
    out += callout([
        "This is the single consolidated report required by <i>Deliverables.pdf</i>. Sections 2 to 8 follow that "
        "document's numbering and answer each of its questions explicitly (look for the green question bars). "
        "Appendix A maps every requirement in all five organiser PDFs to the section that answers it.",
        "<b>Every number, table and plot is produced from recorded experiment files in <font name='Mono'>"
        "artifacts/</font></b>. Appendix B lists those files and their SHA-256 hashes. Nothing was typed in by hand, "
        "estimated or simulated.",
        "<b>Proxy labels.</b> The supplied <font name='Mono'>.route</font> files list road arcs in order but do not say "
        "which GPS point was on which arc. Per-point labels are therefore inferred by alignment (Section 2.10.4). Every "
        "pointwise accuracy in this report is measured against those inferred labels and is called a "
        "<i>proxy</i>. Route-level metrics compare directly with <font name='Mono'>.route</font>.",
    ], title="How to read this report")
    out += [NextPageTemplate("page"), PageBreak()]
    return out


def contents():
    return [P("Contents", "h1"), toc(), PageBreak()]


def overview(D, fig):
    dev, ev = D["dev_summary"], D["ev_summary"]
    hgb = D["models"]["hist_gradient_boosting"]
    out = H1("1. Overview")
    out += H2("1.1 The problem")
    out += [P("A GPS receiver in a moving vehicle does not report where the vehicle is; it reports a noisy estimate. Tall "
              "buildings, multipath reflections, signal loss and irregular sampling push the reported point off the road, "
              "onto a parallel street, or across a block. PathSnap must take a trip's GPS trace and that trip's road "
              "network and reconstruct the sequence of <b>directed road segments (arcs)</b> the vehicle actually drove. "
              "The output must be a plausible, connected route, plus one predicted arc for every GPS observation.")]
    out += H2("1.2 Our approach in one paragraph")
    out += [P("Arc IDs are local to each trip's map, so a model cannot learn \"arc 57\" as a class. We therefore turn the "
              "task into a question that means the same thing on every map: <i>for this GPS point and this nearby road "
              "arc, is the arc the one the vehicle was on?</i> For each GPS point we collect up to "
              f"{D['config']['candidate_count']} nearby directed arcs within {D['config']['candidate_radius_m']:.0f} m, "
              "describe each (point, arc) pair with 13 geometric and motion features, and score it with a binary "
              "classifier. Five model families were compared on whole-trip cross-validation; histogram gradient "
              "boosting was selected and Bayesian-tuned. A hidden Markov model then treats the candidates as hidden states, "
              "uses the classifier scores as emission evidence and the road network's directed shortest-path distances as "
              "transition evidence, and Viterbi decoding picks one connected sequence. Gaps between chosen arcs are filled "
              "with real shortest-path arcs, and the sequence is reset (never bridged with an invented straight line) at "
              "long GPS outages or when no legal road path exists. A FastAPI server on Laptop 1 runs this pipeline on "
              "request; a browser UI on Laptop 2 sends only a trip ID and shows the result.")]
    out.append(figure(fig / "pipeline.png", "Figure 1.1. The prediction pipeline. Blue: input handling; green: learned "
                      "candidate scoring; gold: sequential decoding. The reference route is used only to build "
                      "training labels and to score finished predictions on the server."))
    out += H2("1.3 Headline results")
    rows = [["Metric", "Student test (development, 10 trips)", "", "Reserved evaluator trips (final, 10 trips)", ""],
            ["", "Classifier only", "Viterbi route", "Classifier only", "Viterbi route"],
            ["Pointwise Top-1 accuracy (proxy, pooled)", pct(dev["classifier_top1"]), pct(dev["hmm_top1"]),
             pct(ev["classifier_top1"]), pct(ev["hmm_top1"])],
            ["Route overlap F1 (length-weighted, mean of trips)", f(dev["classifier_route_length_f1"]),
             f(dev["bridged_route_route_length_f1"]), f(ev["classifier_route_length_f1"]),
             f(ev["bridged_route_route_length_f1"])],
            ["Ordered route LCS F1 (mean of trips)", f(dev["classifier_ordered_route_lcs_f1"]),
             f(dev["bridged_route_ordered_route_lcs_f1"]), f(ev["classifier_ordered_route_lcs_f1"]),
             f(ev["bridged_route_ordered_route_lcs_f1"])],
            ["Direct arc continuity (mean of trips)", f(dev["classifier_direct_arc_continuity"]),
             f(dev["bridged_route_direct_arc_continuity"]), f(ev["classifier_direct_arc_continuity"]),
             f(ev["bridged_route_direct_arc_continuity"])],
            ["GPS observations scored", n(dev["observations"]), "", n(ev["observations"]), ""]]
    t = table(rows, [3.3, 1.5, 1.5, 1.5, 1.5], header_rows=2)
    t.setStyle(TableStyle([("SPAN", (1, 0), (2, 0)), ("SPAN", (3, 0), (4, 0)), ("SPAN", (0, 0), (0, 1))]))
    out += [t, P("Table 1.1. \"Viterbi route\" means HMM-decoded states for pointwise accuracy and the bridged route "
                 "(decoded arcs plus shortest-path fill-in) for route metrics. Sources: decoder_comparison.json, "
                 "evaluator_results.json.", "caption")]
    out += [P("Three findings frame the rest of the report:")]
    out += bullets([
        f"<b>The learned scorer is strong and stable.</b> Selected model: histogram gradient boosting, CV candidate "
        f"Macro F1 {f(hgb['cv_candidate_macro_f1'])} ± {f(hgb['cv_candidate_macro_f1_std'])} before tuning and "
        f"{f(D['tuning']['cv_candidate_macro_f1'])} after. It led every other family on CV Macro F1, log loss and "
        "student-test Top-1 (Section 3).",
        "<b>Viterbi decoding makes the route connected and closer to the reference route</b>: continuity rises from about "
        "0.74 to above 0.99, and route overlap F1 improves on both partitions (Section 5).",
        "<b>Decoding lowers proxy pointwise accuracy</b>, mildly on development data and strongly on the reserved trips "
        f"({pct(ev['classifier_top1'])} to {pct(ev['hmm_top1'])}), with one trip (00000015) accounting for most of the "
        "drop. We report this trade-off rather than retune on reserved results (Sections 5.8 and 7.7).",
    ])
    out += H2("1.4 Data partitions and how each one was used")
    rows = [["Partition", "Trips", "GPS points", "Used for", "Never used for"],
            ["train", "80", n(D["train"]["obs_total"]), "EDA, label construction, model fitting, 3-fold whole-trip CV, "
             "model selection, Bayesian tuning", "-"],
            ["student_test", "10", n(D["student_test"]["obs_total"]), "EDA, development comparison of all models, "
             "decoder setting search, API rehearsal", "classifier selection, hyperparameter search"],
            ["evaluator_test (reserved)", "10", n(D["ev_summary"]["observations"]), "one final scoring run through the "
             "frozen API after commit a61ce50", "any training, tuning, selection, threshold or setting"]]
    out += [table(rows, [1.7, 0.6, 0.9, 3.4, 2.4]), P("Table 1.2. Partition roles. No trip is split across partitions.",
                                                    "caption")]
    out += H2("1.5 Where each Deliverables question is answered")
    rows = [["Deliverables item", "Section"],
            ["2.2 Q1 How did you preprocess the data?", "2.10"],
            ["2.2 Q2 Patterns observed; features extracted and why", "2.11"],
            ["2.2 Missing values, GPS trajectory plots, road-network plots, Folium map, partitions", "2.3 to 2.8"],
            ["3.1 Consolidated comparison table (exact columns)", "3.1, Table 3.2"],
            ["3.2 A1-A2, B5-B6, C7-C9 for every model", "3.2.1 to 3.2.5"],
            ["3.3 Required model plots; no artificial training curves", "3.3"],
            ["4 Bayesian optimisation Q1-Q6, tuned vs untuned", "4"],
            ["5 HMM / Viterbi Q1-Q6, classifier-only vs HMM, where it did not help", "5"],
            ["6 Folium maps (layers, legend, extent, layer control, screenshot, code)", "6"],
            ["7 API, UI, deployment, screenshots", "7"],
            ["8 Academic integrity and AI disclosure", "8"],
            ["Models and Training Guidelines, Dataset Exploration, Assessment Policy", "Appendix A"]]
    out += [table(rows, [5, 1]), P("Table 1.3. Compact traceability. The full matrix is Appendix A.", "caption")]
    return out
