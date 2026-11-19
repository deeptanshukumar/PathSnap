"""Section 2: dataset understanding and exploratory data analysis (Deliverables 2.2)."""

from .data import ROOT, f, n, pct
from .layout import H1, H2, H3, P, bullets, callout, code, eq, figure, numbered, question, table


def section(D, fig):
    tr, st = D["train"], D["student_test"]
    cfg = D["config"]
    out = H1("2. Dataset understanding and exploratory data analysis")
    out += [P("<b>Partitions used.</b> Every analysis in this section uses the 80 <i>train</i> trips and the 10 "
              "<i>student_test</i> trips, and each table or plot states which. The 10 reserved evaluator trips were not "
              "explored: they were only counted, hashed and scored once after the predictor was frozen (Section 7.7). "
              "All statistics come from <font name='Mono'>artifacts/inventory.json</font>, written by "
              "<font name='Mono'>pathsnap/prepare.py</font>.")]

    out += H2("2.1 What the raw files contain and how they relate")
    rows = [["File", "Columns (headerless, whitespace-separated)", "One row is", "ID meaning"],
            [".track", "longitude, latitude, timestamp in seconds", "one GPS observation", "row order = time order"],
            [".nodes", "longitude, latitude", "one road-network node (junction or shape point)",
             "row number = node ID (0-based)"],
            [".arcs", "from_node_id, to_node_id (integers)", "one directed road segment", "row number = arc ID (0-based)"],
            [".route", "arc_id", "one traversed arc, in driving order", "values index rows of .arcs"]]
    out += [table(rows, [0.8, 2.6, 2.3, 2.1]), P("Table 2.1. The four files present for every trip. Verified by the loader "
                                                  "(pathsnap/data.py) for all 90 development trips.", "caption")]
    out += [P("Three facts about these files drive the whole design:")]
    out += numbered([
        "<b>IDs are local to a trip.</b> Node 10 and arc 10 in one trip have nothing to do with node 10 and arc 10 in "
        "another. A model therefore cannot learn arc IDs as classes; it must learn from geometry and motion.",
        "<b>Arcs are directed.</b> A two-way street is two arcs with identical geometry and opposite direction. Position "
        "alone cannot tell them apart; only the direction of travel can.",
        "<b>A GPS row does not correspond to a route row.</b> A trip with 2,000 GPS points may have 300 route arcs. The "
        "route says which arcs were driven and in which order, not which GPS point belongs to which arc. Per-point "
        "labels must be inferred (Section 2.10.4).",
    ])
    out += callout([
        "<b>Example (invented values, for illustration only).</b> Task: show why GPS rows cannot be matched to route "
        "rows. Assume a straight street at latitude 12.9° with three nodes: node 0 at (77.5000, 12.9000), node 1 "
        "at (77.5010, 12.9000), node 2 at (77.5020, 12.9000), given as (longitude, latitude). Four arcs: arc 0 = 0→1, "
        "arc 1 = 1→0, arc 2 = 1→2, arc 3 = 2→1, so arcs 0/1 and 2/3 are twins. The route is [0, 2]: the "
        "vehicle drove east from node 0 to node 2. The track has three rows: (77.50005, 12.90003, 1 s), "
        "(77.50050, 12.89998, 2 s), (77.50105, 12.90004, 3 s).",
        "There are 3 GPS rows but 2 route rows, so there is no row-to-row pairing. GPS row 2 sits about 2 m from "
        "<i>both</i> arc 0 and its twin arc 1; only the eastward motion between rows 1 and 3 says arc 0 is right. Each "
        "arc is about 108 m long, because 0.001° of longitude at 12.9° latitude is about 108 m, while "
        "0.001° of latitude is about 111 m everywhere. That difference is why we never measure distance in degrees.",
    ], title="Worked example: rows, IDs and twin arcs")

    out += H2("2.2 Dataset inventory")
    rows = [["Statistic", "train", "student_test"],
            ["Trips / files present", f"{tr['trips']} / {tr['files']}", f"{st['trips']} / {st['files']}"],
            ["GPS observations, total", n(tr["obs_total"]), n(st["obs_total"])],
            ["GPS observations per trip: min / median / mean / max",
             f"{n(tr['obs_min'])} / {n(tr['obs_median'])} / {n(tr['obs_mean'])} / {n(tr['obs_max'])}",
             f"{n(st['obs_min'])} / {n(st['obs_median'])} / {n(st['obs_mean'])} / {n(st['obs_max'])}"],
            ["Road nodes, total (per-trip min / median / max)",
             f"{n(tr['nodes_total'])} ({n(tr['nodes_min'])} / {n(tr['nodes_median'])} / {n(tr['nodes_max'])})",
             f"{n(st['nodes_total'])} ({n(st['nodes_min'])} / {n(st['nodes_median'])} / {n(st['nodes_max'])})"],
            ["Directed arcs, total (per-trip min / median / max)",
             f"{n(tr['arcs_total'])} ({n(tr['arcs_min'])} / {n(tr['arcs_median'])} / {n(tr['arcs_max'])})",
             f"{n(st['arcs_total'])} ({n(st['arcs_min'])} / {n(st['arcs_median'])} / {n(st['arcs_max'])})"],
            ["Reference route entries per trip: min / median / max",
             f"{n(tr['route_min'])} / {n(tr['route_median'])} / {n(tr['route_max'])}",
             f"{n(st['route_min'])} / {n(st['route_median'])} / {n(st['route_max'])}"],
            ["Data types", "float lon/lat/time; integer-valued IDs", "same"]]
    out += [table(rows, [3.2, 2.4, 2.4]), P("Table 2.2. Inventory per partition. Every file's SHA-256 is recorded in "
                                            "inventory.json. File sizes in bytes were not retained in the inventory; row "
                                            "counts per file are given instead.", "caption")]
    out.append(figure(fig / "eda_sizes.png", "Figure 2.1. Size distributions per trip (train in green, student_test in "
                      "gold; log scales). Trip length spans two orders of magnitude and network size spans almost three."))
    out += [P(f"<b>What we observed.</b> Trip lengths are heavily skewed: most trips have under 4,000 points, but training "
              f"trip 00000054 has {n(D['by_trip']['00000054']['gps_observations'])}. Network size is not tied to trip "
              f"length: trip 00000013 has {n(D['by_trip']['00000013']['gps_observations'])} points but a "
              f"{n(D['by_trip']['00000013']['arcs'])}-arc map. Two consequences follow. Pooled metrics are dominated by "
              "long trips, so we also report per-trip means and per-trip tables. Candidate search and shortest paths must "
              "scale to very large graphs, so we use a spatial index and a bounded, cached Dijkstra (Sections 2.10.3 and 5.4).")]

    out += H2("2.3 Data types, validity checks and missing-value analysis")
    rows = [["Check (implemented in pathsnap/data.py and prepare.py)", "train", "student_test", "Decision"],
            ["Non-finite (missing / NaN / inf) values in any file", n(tr["nonfinite"]), n(st["nonfinite"]),
             "None to impute; loader raises an error if any appear"],
            ["Non-increasing timestamps in .track", n(tr["nonincreasing"]), n(st["nonincreasing"]),
             "Time order is trustworthy; loader enforces it"],
            ["Implausible coordinates (|lon| > 180 or |lat| ≥ 90)", "0 (all trips loaded)", "0 (all trips loaded)", "Rejected if present"],
            ["Fractional or out-of-range node IDs in .arcs", "0 (all trips loaded)", "0 (all trips loaded)", "Rejected if present (unit-tested)"],
            ["Route IDs outside the arc table", "0 (all trips loaded)", "0 (all trips loaded)", "Rejected if present"],
            ["Isolated nodes (degree 0)", n(tr["isolated"]), n(st["isolated"]), "No action needed"],
            ["Zero-length arcs", f"{n(tr['zero_len'])} in {tr['zero_len_trips']} trips",
             f"{n(st['zero_len'])} in {st['zero_len_trips']} trips", "Kept; safe division gives heading 0"],
            ["Missing time (steps longer than 1 s)", f"{pct(1 - tr['one_second_share'])} of steps",
             f"{pct(1 - st['one_second_share'])} of steps", "Handled as gaps, never interpolated (2.4)"]]
    out += [table(rows, [3.5, 1.2, 1.2, 2.8]), P("Table 2.3. Missing-value and validity analysis. Counts are over all "
                                                 "rows of all trips in each partition.", "caption")]
    out += [P("<b>Interpretation.</b> There are no missing cells in the usual sense. The data's real \"missingness\" is "
              "temporal: sometimes the receiver produced no fix for seconds, minutes or hours. We therefore treat gaps as the "
              "missing-value problem and handle them explicitly in feature construction and decoding. We deliberately do "
              "not fill gaps with interpolated GPS points: an invented point would look like evidence that the vehicle was "
              "at a place it may never have been.")]

    out += H2("2.4 Timestamps, sampling rate and gaps")
    out.append(figure(fig / "eda_sampling.png", "Figure 2.2. (a) Per-trip share of 1-second steps. (b) Largest gap in each "
                      "trip, sorted; the red line is the 60 s threshold at which the decoder starts a new segment. "
                      "Partitions: train (green) and student_test (gold)."))
    out += [P(f"Sampling is nominally 1 Hz: {pct(tr['one_second_share'])} of training steps and "
              f"{pct(st['one_second_share'])} of student-test steps are exactly 1 s. Training contains {tr['gaps60']} gaps "
              f"longer than 60 s, spread over {tr['gap_trips']} trips. The longest is {n(tr['max_gap'])} s (about 42 hours) "
              f"in trip 00000090, which has only {n(D['by_trip']['00000090']['gps_observations'])} points. Trip 00000078 has "
              f"a {n(D['by_trip']['00000078']['max_gap_s'])} s gap, and trips 00000042 and 00000081 have 17 and 15 gaps over "
              "60 s respectively. The student-test partition has a single gap over 60 s: 210 s in trip 00000065.")]
    out += [P("<b>Decisions.</b> (i) Heading and speed are estimated from a ±3-observation window that is shrunk so it "
              "never spans a gap over 60 s. (ii) The decoder starts a new independent segment after any gap over 60 s "
              "(Section 5.6). (iii) The log of the previous time step is a model feature, so the classifier can learn that "
              "evidence after a gap is weaker.")]

    out += H2("2.5 Geographic extent and the choice of coordinate system")
    out.append(figure(fig / "eda_geography.png", "Figure 2.3. (a) Network centre of every trip; the data spans several "
                      "continents. (b) Largest network dimension per trip in km, measured after projection. "
                      "Partitions: train and student_test."))
    lon = (min(tr["lon"][0], st["lon"][0]), max(tr["lon"][1], st["lon"][1]))
    lat = (min(tr["lat"][0], st["lat"][0]), max(tr["lat"][1], st["lat"][1]))
    err, err_trip = D["proj_err"]
    out += [P(f"Trips lie between longitudes {lon[0]:.1f}° and {lon[1]:.1f}° and latitudes {lat[0]:.1f}° "
              f"and {lat[1]:.1f}°. Because several trips have first-column values beyond ±90, that column can "
              "only be longitude; the column order (longitude, latitude) is therefore confirmed by the data, not just "
              "assumed. Folium and Leaflet expect (latitude, longitude), so display code swaps the order explicitly.")]
    out += [P("Longitude and latitude are angles. One degree of latitude is about 111 km everywhere, but one degree of "
              "longitude shrinks with the cosine of latitude, so Euclidean distance in degrees is wrong and its error "
              "depends on where the trip is. No single fixed projection suits trips on several continents. We therefore "
              "project each trip into its own local metric plane, centred on the median coordinate of its road nodes "
              "(φ<sub>0</sub>, λ<sub>0</sub>):")]
    out.append(eq("x = R · cos(φ<sub>0</sub>) · (λ − λ<sub>0</sub>),&nbsp;&nbsp;&nbsp; "
                  "y = R · (φ − φ<sub>0</sub>),&nbsp;&nbsp;&nbsp; R = 6,371,008.8 m,&nbsp; angles in radians"))
    out += [P(f"<b>Why this is accurate enough.</b> This local equirectangular projection is exact along the central "
              "parallel and meridian. Its main error is the east-west scale away from φ<sub>0</sub>; to first order "
              "the relative scale error is tan(φ<sub>0</sub>) · Δφ. Computed from each trip's recorded "
              f"latitude extent, the worst case over all 90 trips is {pct(err)} (trip {err_trip}). At that rate a 100 m "
              f"distance at the far edge of the largest network is off by under {100 * err:.1f} m, against a typical "
              f"GPS-to-reference distance of about {tr['label_med_med']:.1f} m (Section 2.7). The largest network spans {tr['extent_max_km']:.1f} km. "
              "This is an analytic bound computed from the inventory, not a measured geodesic comparison; a per-trip "
              "azimuthal-equidistant projection would remove even this error and is listed as future work.")]

    out += H2("2.6 Road-network size and connectivity")
    out.append(figure(fig / "eda_network.png", "Figure 2.4. (a) Weakly connected components grow with network size. "
                      "(b) Zero-length arcs per trip. Partitions: train and student_test."))
    out += [P(f"A network is <i>weakly connected</i> if every node can reach every other when road direction is ignored. "
              f"Only one of the 90 networks is a single component. Training networks contain {n(tr['weak_total'])} weak "
              f"components in total (median {tr['weak_median']:.0f} per trip, maximum {tr['weak_max']} in trip 00000043); "
              f"student-test networks contain {n(st['weak_total'])} (median {st['weak_median']:.0f}, maximum "
              f"{st['weak_max']}). A likely reason is that each map is a clipped extract around its trip, so roads cut at "
              "the boundary become small islands. No node is isolated. Being in the same weak component does not guarantee a "
              "legal <i>directed</i> path, because one-way arcs can still block travel.")]
    out += [P("<b>Decisions.</b> Shortest paths are computed on the directed graph with SciPy's Dijkstra. When two "
              "parallel arcs join the same pair of nodes, the graph keeps the shorter one rather than summing their lengths "
              "(a pitfall of building a sparse matrix from duplicate entries). Whenever no legal directed path exists "
              "between consecutive candidates, the decoder resets instead of inventing a connection (Section 5.7). We did "
              "not measure the share of arcs that have a reverse twin; heading features handle twins whether or not they "
              "are present.")]

    out += H2("2.7 Reference routes: structure and limitations")
    out.append(figure(fig / "eda_reference_quality.png", "Figure 2.5. (a) Distance from each GPS point to its aligned "
                      "reference arc, per trip: bar = 95th percentile, dot = median, labels = last two digits of trip IDs "
                      "above 60 m. (b) Training trips with most observations excluded by the 75 m label-quality filter."))
    out += [P(f"Reference routes have {n(tr['route_min'])} to {n(tr['route_max'])} entries in training. Repeated arcs occur "
              "(a vehicle may drive the same street twice), so the label method must keep separate route positions. "
              f"Consecutive route arcs should share a node; that fails {tr['breaks']} times in training "
              f"(trips {', '.join(t[-2:] for t in tr['break_trips'])}) and {st['breaks']} time in student test "
              "(trip 00000076). For typical trips the GPS-to-reference distance has a median of "
              f"{tr['label_med_med']:.1f} m and a 95th percentile of {tr['label_p95_med']:.1f} m (medians over training "
              "trips), which is ordinary GPS error. A few trips disagree badly with their reference: in trip 00000054 the "
              f"median distance is {D['by_trip']['00000054']['proxy_label_distance_median_m']:.0f} m and in trip "
              f"00000033 it is {D['by_trip']['00000033']['proxy_label_distance_median_m']:.0f} m. Trip 00000002 has a 95th "
              f"percentile of {D['by_trip']['00000002']['proxy_label_distance_p95_m']:.0f} m.")]
    out += callout(["Following <i>Dataset Exploration.pdf</i> Section 6, we treat <font name='Mono'>.route</font> as "
                    "reference information, not independently verified physical truth. Disagreements can come from GPS "
                    "error, from map errors, or from the reference itself. We do not try to decide which; we limit the "
                    "damage. Training excludes points farther than 75 m from their aligned reference arc, while scoring "
                    "always uses every point. All pointwise accuracies are reported as proxies, and route-level metrics "
                    "compare with the reference sequence directly."], title="Consequence for evaluation")

    out += H2("2.8 GPS trajectory and road-network visualisations")
    out.append(figure(ROOT / "reports/plots/trajectory.png", "Figure 2.6. Partition student_test, trip 00000000 "
                      "(2,503 points). Left: raw GPS in degrees. Right: the complete supplied road network for the same "
                      "trip. The vehicle uses a thin slice of a large network, which is why candidates are searched "
                      "locally around each point.", width=0.95))
    out.append(figure(ROOT / "reports/evidence/folium_development.jpg", "Figure 2.7. Folium map of student_test trip "
                      "00000000 generated by pathsnap/maps.py (interactive file: reports/maps/development_00000000.html). "
                      "Blue: GPS observations; gold: reference route (development maps only); green: Viterbi route; red "
                      "rings: decoder resets; grey: candidate road network; layer control top right. Section 6 describes "
                      "the map design.", width=0.9))

    out += H2("2.9 Data-quality record: issues, evidence, decisions and impact")
    rows = [["Issue", "Evidence", "Decision", "Impact"],
            ["Angular coordinates; worldwide trips", "lon −100 to 137°, lat −35 to 62°",
             "Per-trip local metre projection", f"Distance error ≤ {pct(err)} (analytic bound)"],
            ["Long GPS outages", f"{tr['gaps60']} gaps > 60 s in {tr['gap_trips']} train trips; max {n(tr['max_gap'])} s",
             "Reset decoder at > 60 s; windows never cross gaps", "No invented travel across outages"],
            ["No per-point labels", ".route is an ordered arc list", "Monotone alignment gives proxy labels",
             "Pointwise metrics labelled as proxies"],
            ["Reference / GPS disagreement", "Trips 54, 33, 2: median or p95 far above 75 m",
             f"Exclude > 75 m points from fitting only ({n(tr['excluded'])} of {n(tr['obs_total'])})",
             "Cleaner training labels; scoring unaffected"],
            ["Fragmented, directed networks", f"{n(tr['weak_total'])} weak components (train)",
             "Directed Dijkstra; reset when no legal path", "Explicit breaks instead of straight lines"],
            ["Reference endpoint breaks", f"{tr['breaks']} (train), {st['breaks']} (student_test)",
             "Reported; not repaired", "Reference continuity is not exactly 1"],
            ["Twin arcs (identical geometry)", "Two-way streets are two arcs", "Signed heading features",
             "Direction becomes learnable"],
            ["Very large networks", f"up to {n(tr['arcs_max'])} arcs", "KD-tree candidate index; bounded, cached Dijkstra",
             f"Mean {sum(D['dev_seconds']) / len(D['dev_seconds']):.2f} s prediction per student-test trip"],
            ["Duplicate parallel arcs", "Same (from, to) pair repeated", "Graph keeps the minimum length",
             "Correct shortest paths"],
            ["Zero-length arcs", f"{tr['zero_len']} (train), {st['zero_len']} (student_test)",
             "Safe division", "Finite features (asserted)"]]
    out += [table(rows, [1.7, 2.1, 2.2, 1.9]), P("Table 2.4. Data-quality record.", "caption")]

    out += H2("2.10 Preprocessing")
    out += question("Deliverables 2.2, Q1", "How did you preprocess the data?")
    out += [P("<b>Answer in brief.</b> We validate every file; project each trip into local metres; index every road "
              f"segment for fast search; generate up to {cfg['candidate_count']} directed candidate arcs within "
              f"{cfg['candidate_radius_m']:.0f} m of each GPS point; compute 13 route-independent features per (point, "
              "candidate) pair; and, for training and scoring only, derive a proxy label per GPS point by monotone "
              f"alignment to the reference route. For fitting, we keep every {cfg['training_stride']}th observation, drop "
              f"points whose aligned reference is more than {cfg['training_label_max_distance_m']:.0f} m away, and fit any "
              "feature scaler inside each cross-validation fold. Nothing is imputed. The details and the reason for each "
              "step follow.")]
    out += H3("2.10.1 Validation (pathsnap/data.py)")
    out += [P("Each file must have the expected column count and only finite values. Coordinates must be plausible, "
              "timestamps strictly increasing, arc endpoints integral and inside the node table, and there must be at "
              "least two GPS points. Invalid input raises an error instead of being repaired silently, because a silent "
              "repair would hide a data problem inside the results. <b>The prediction loader never opens the route "
              "file</b>; route loading is a separate function used only by label construction and server-side scoring. A "
              "unit test makes any attempt to read a route file during prediction fail, and prediction still completes "
              "(Table 7.3).")]
    out += H3("2.10.2 Projection to metres (pathsnap/geometry.py)")
    out += [P("Each trip uses the projection of Section 2.5, centred on the median node coordinate. GPS points, nodes and "
              "arcs are all projected with the same origin, so distances and headings between them are consistent. Arc "
              "lengths and unit direction vectors are computed once per trip.")]
    out += H3("2.10.3 Road index and candidate generation (pathsnap/features.py)")
    out += [P("Checking every GPS point against every arc would cost (points × arcs); for the largest training trip "
              "that is billions of distance computations. Instead we sample every arc at intervals of at most "
              f"{cfg['index_spacing_m']:.0f} m and store the samples in a KD-tree. For each GPS point we query a radius of "
              f"{cfg['candidate_radius_m']:.0f} + {cfg['index_spacing_m'] / 2:.0f} = "
              f"{cfg['candidate_radius_m'] + cfg['index_spacing_m'] / 2:.0f} m. The extra half-spacing guarantees that any "
              f"arc passing within {cfg['candidate_radius_m']:.0f} m has at least one sample inside the query, since no "
              f"point of an arc is more than {cfg['index_spacing_m'] / 2:.0f} m from its nearest sample. For the arcs found "
              f"we compute the exact distance from the point to the finite segment (not the infinite line), keep arcs within "
              f"{cfg['candidate_radius_m']:.0f} m, and retain the {cfg['candidate_count']} closest. A unit test compares this "
              "against brute force, including very long arcs.")]
    out += [P(f"<b>Why {cfg['candidate_radius_m']:.0f} m and {cfg['candidate_count']} candidates.</b> These were fixed "
              f"design choices, not tuned values. Typical GPS-to-reference offsets have a 95th percentile near "
              f"{tr['label_p95_med']:.0f} m, so {cfg['candidate_radius_m']:.0f} m leaves wide margin for poor fixes. Eight "
              "slots hold a twin pair plus several parallel or crossing streets in dense areas while keeping the decoder's "
              "state space small (the Viterbi step costs grow with the square of the candidate count). The measured "
              "consequence: the aligned reference arc is missing from the candidate set for only "
              f"<b>{pct(st['miss_pooled'])}</b> of student-test points. In training the pooled miss rate is "
              f"{pct(tr['miss_pooled'])}, but that is driven almost entirely by trips 00000054, 00000033 and 00000002, whose "
              f"references lie far from the GPS; without them it is {pct(tr['miss_pooled_clean'])}.")]
    out += [P("<b>No candidate within the radius.</b> The point receives the nearest arcs found by a k-nearest-sample "
              "search and is flagged as a fallback. This affected "
              f"{pct(tr['fallback_pooled'])} of training points ({tr['fallback_trips']} trips) and no student-test point. "
              "The reference arc is never inserted into a candidate set: doing so would leak the answer and inflate "
              "accuracy.")]
    out += H3("2.10.4 Proxy labels by monotone alignment (pathsnap/reference.py)")
    out += [P("Training needs a label for every candidate: 1 if it is the arc the vehicle was on at that moment, else 0. "
              "Because the route has no timestamps, we align the GPS sequence to the route sequence. Let "
              "d(t, j) be the distance from GPS point t to the arc at route position j, and let D[t, j] be the cheapest "
              "way to explain points 0..t with point t on route position j:")]
    out.append(eq("D[t, j] = d(t, j)<super>2</super> + min<sub>i ≤ j</sub> D[t − 1, i]"))
    out += [P("The minimum over i ≤ j makes the alignment <i>monotone</i>: the vehicle can stay on a route position or "
              "move forward, skipping short arcs that the GPS missed, but it cannot go back. States are route "
              "<i>positions</i>, not arc IDs, so a street driven twice is two different positions, and loops work. A "
              "running prefix minimum makes each step linear in route length, so a trip costs O(points × route "
              "length). Backtracking gives one route position per GPS point and hence one proxy arc. Squared distance "
              "penalises large offsets strongly, so a point is not assigned to a distant arc when a nearby route position "
              "is available.")]
    out += [P("<b>Limitations, stated plainly.</b> The alignment assumes the route is in driving order. Near "
              "junctions and U-turns, two route positions can be almost equally close and the choice is a tie-break, not "
              "knowledge. Start and end positions are not forced. These labels are therefore a defensible supervision "
              "convention, not ground truth.")]
    out += H3("2.10.5 Selecting training rows")
    m = D["manifest"]
    out += [P(f"(i) <b>Quality filter:</b> observations whose aligned reference arc is more than "
              f"{cfg['training_label_max_distance_m']:.0f} m away are excluded from fitting only: {n(tr['excluded'])} of "
              f"{n(tr['obs_total'])} training observations ({pct(tr['excluded'] / tr['obs_total'])}), mostly from trips "
              "54, 33 and 2 (Figure 2.5b). Such labels are more likely wrong than informative. The filter was fixed before "
              "any model comparison and applied identically to every model. Student-test and reserved points are never "
              f"filtered. (ii) <b>Stride:</b> we keep observations whose index is divisible by {cfg['training_stride']}. At "
              "1 Hz, neighbouring points are near-duplicates, so the stride cuts fitting time about five-fold with "
              f"little loss of information. Result: <b>{n(m['candidate_training_rows'])} candidate rows, of which "
              f"{n(m['positive_rows'])} ({pct(m['positive_rows'] / m['candidate_training_rows'])}) are positive</b>, from "
              f"all {m['training_trips_with_usable_rows']} training trips.")]
    out += H3("2.10.6 Scaling and leakage control")
    out += [P("Logistic regression and Gaussian Naive Bayes are wrapped in a scikit-learn Pipeline with a StandardScaler, "
              "so the scaler is fitted on the training folds only, inside each cross-validation split. Tree models are "
              "scale-invariant and receive raw features. Arc IDs, raw coordinates, route membership and any statistic "
              "derived from labels are absent from the features.")]
    out += H3("2.10.7 Caching and provenance")
    out += [P(f"<font name='Mono'>prepare.py</font> stores features, candidates and labels per trip in compressed "
              "files keyed by a signature of the raw-file hashes, the configuration and the feature list, so a cache can "
              "never be reused after the data or features change. Preparing all 90 development trips took "
              f"{st['prep_seconds'] + tr['prep_seconds']:.0f} s on a CPU laptop.")]

    out += H2("2.11 Patterns observed and features extracted")
    out += question("Deliverables 2.2, Q2", "What patterns did you observe during data exploration? What features did "
                                            "you extract and why did the team choose those features?")
    out += [P("<b>Patterns observed</b> (each with its evidence above):")]
    out += bullets([
        "GPS points usually lie a few metres from the correct road (median about 4.6 m), but the nearest arc is not "
        "always right: parallel streets, junctions and twins sit within the same few metres (Sections 2.1 and 2.7).",
        "Two-way roads appear as twin arcs with identical geometry, so only the direction of motion separates them.",
        "Heading is reliable when the vehicle moves and unreliable when it is stopped or crawling.",
        "Sampling is mostly 1 Hz but has gaps, after which position evidence is less connected to the previous point "
        "(Section 2.4).",
        "Arc lengths vary widely, from short shape segments to long rural links, so where a point projects along an arc "
        "is informative.",
        "Junction complexity varies: some nodes join many arcs, which raises ambiguity.",
    ])
    rows = [["#", "Feature", "Definition", "Pattern it addresses / why chosen"],
            ["1", "distance_m", "Distance from GPS point to the finite arc segment", "Closer arcs are more likely"],
            ["2", "log_distance", "log(1 + distance)", "Compresses the long tail; gives linear models a usable scale"],
            ["3", "distance_above_nearest_m", "distance minus the nearest candidate's distance",
             "Relative evidence: being 12 m away matters less if every option is 12 m away"],
            ["4", "heading_cos", "cos of the angle between motion direction and arc direction",
             "Separates twins: +1 for the arc driven forwards, −1 for its reverse"],
            ["5", "abs_heading_cos", "|heading_cos|", "Road alignment regardless of direction (a road crossing at "
             "90° scores 0)"],
            ["6", "length_m", "Arc length", "Short shape segments vs long links behave differently"],
            ["7", "log_length", "log(1 + length)", "Same, on a compressed scale"],
            ["8", "fraction", "Projection position along the arc, 0 to 1", "Points near endpoints are junction-ambiguous"],
            ["9", "speed_mps", "Speed from the ±3-point window", "Tells the model when heading is trustworthy"],
            ["10", "log_previous_gap_s", "log(1 + time since previous point)", "Weaker evidence after gaps"],
            ["11", "source_indegree", "Arcs entering the arc's start node", "Junction complexity"],
            ["12", "target_outdegree", "Arcs leaving the arc's end node", "Junction complexity, dead ends"],
            ["13", "stationary", "1 if speed &lt; 0.5 m/s", "Explicit switch for unreliable heading"]]
    out += [table(rows, [0.3, 1.8, 2.8, 3.1]), P("Table 2.5. The 13 features computed for every (GPS point, candidate arc) "
                                                 "pair (pathsnap/features.py).", "caption")]
    out += [P("<b>Why these and not others.</b> Every feature uses only the GPS track and the road network, so the same "
              "code runs at prediction time without the reference route. Every feature means the same thing on every map, "
              "which matters because the five model families must generalise to trips and countries they have never seen. "
              "We excluded arc IDs (meaningless across trips), raw coordinates (they would let a model memorise places "
              "rather than learn geometry), and anything derived from the route or labels (leakage). Heading uses future "
              "points within ±3 observations. That is legitimate here because the whole trip is available when "
              "matching, but it makes the method offline rather than real-time.")]
    out += [P("<b>Known weakness of this set.</b> Several features are deliberately redundant (distance and "
              "log-distance; length and log-length; heading and its absolute value). Tree models are unaffected, but the "
              "redundancy breaks Naive Bayes' independence assumption (Section 3.2.2). The features also describe one point "
              "at a time; context over neighbouring points is left to the HMM. The opt-in variant in Section 9 tests adding "
              "such context directly.")]
    return out
