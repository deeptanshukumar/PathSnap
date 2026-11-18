# PathSnap viva and demonstration preparation

Read the code and actual report before rehearsing. These are AI-assisted prompts for the team to explain in its own words, not a substitute for understanding.

## A short opening

“We score nearby directed road candidates for each noisy GPS observation, then decode a coherent route using directed network distances. We compared five models on whole-trip folds, ran 12 Bayesian tuning trials and tested every supplied development trip. The reconstructed route improves route overlap and continuity, with a measured trade-off in inferred pointwise accuracy.”

## Questions every member should be able to answer

1. **What is the label?** A route file is a chronological sequence of arc IDs, not one label per GPS row. We derive proxy observation labels by monotone minimum-distance alignment to that sequence. We explicitly disclose the assumption and also compare predicted/reference road sequences directly.
2. **Why not classify arc ID directly?** Arc IDs are local to each trip and maps have different networks. A binary observation/candidate scorer learns geometric and motion relationships that transfer between maps. The arc ID remains metadata linking a score to its road geometry.
3. **Why use metres?** Longitude degrees have latitude-dependent length. Our local projection uses R cos(latitude) for longitude and R for latitude, then exact projection onto finite straight road segments. It is an approximation over local networks, with the largest extent about 77.6 km.
4. **How do candidates work?** Sample every road edge at <=30 m intervals. A 115 m sample search includes every segment intersecting the 100 m search radius. Recompute exact segment distances and keep eight directed candidates. Missing-radius cases receive a flagged nearest-sample fallback; the reference never repairs candidate misses.
5. **Where can leakage occur?** Randomly splitting candidate rows would put observations from the same trip in training and validation. We use three whole-trip folds and fit scalers inside each fold. Prediction has no reference field; changing `.route` changes scoring but not predictions, which an API test verifies. The evaluator folder is never used during development.
6. **Which models did you actually train?** Logistic regression, Gaussian Naive Bayes, decision tree, random forest and histogram gradient boosting. Show `artifacts/model_comparison.csv` for configurations, CV, precision/recall, log loss, Top-1, continuity and timings. Do not claim neural-network training or GPU experiments.
7. **Why histogram boosting?** It has the highest mean grouped-CV candidate Macro F1 among our baselines: 0.9278. It captures nonlinear feature interactions at modest CPU cost. It was selected by CV, not by the highest training accuracy or a reserved-test score.
8. **What is Bayesian about the search?** Optuna TPE models parameter distributions for good and other trials and samples promising settings. We used four startup random trials followed by eight adaptive trials, seed 357, maximising mean grouped-CV candidate Macro F1. A search with only four startup trials would not have demonstrated adaptive TPE; ours completed 12.
9. **Did tuning improve everything?** No. CV candidate Macro F1 rises 0.9278 -> 0.9290, while development proxy Top-1 changes 87.63% -> 87.46%. We retained that discrepancy. Candidate threshold classification and observation ranking are different objectives.
10. **What are the HMM states, emissions and transitions?** States are projected directed candidate arcs. Emissions normalise positive classifier probabilities across a GPS observation's valid candidates. Transition log compatibility is -abs(network distance - GPS displacement)/beta. Directed network distance includes partial start/end arcs and an actual shortest path between them. This is an HMM-style conditional decoder with discriminative emissions.
11. **What is the Viterbi recurrence?** For candidate j at time t, add its log emission to the maximum over previous candidate i of previous path score plus log transition(i,j). Store the winning predecessor and backtrack. Complexity of the recurrence is O(T K squared); routing is an additional cost. Show `viterbi()` and its brute-force correctness test.
12. **How do you handle gaps and disconnections?** A gap >60 s starts a new sequence. If no legal directed path survives within 55 m/s times elapsed seconds plus 100 m slack, reset and flag it. We do not imply the vehicle travelled a straight line over a missing connection. A small same-edge backwards projection of <=3 m is treated as stationary jitter.
13. **Did decoding help?** Route overlap F1 rises 0.9339 -> 0.9570; ordered route LCS F1 rises 0.7332 -> 0.8012. HMM-state continuity is 0.8822 and bridging raises it to 0.9961, from classifier-only 0.7432. Proxy pointwise accuracy falls 87.46% -> 85.42%, improving on five trips and hurting five. Bridging and Viterbi are distinct contributors; do not attribute all continuity gains to Viterbi.
14. **What does route overlap miss?** It compares unique directed arc sets weighted by length, so repeated travel on a correct edge is not penalised. Ordered LCS F1 adds ordering/repetition sensitivity. Neither establishes physical ground truth if the supplied reference is wrong.
15. **Why can your scores look high on one example?** Trip 00000052 is easy: HMM proxy accuracy is 99.62%. Trip 00000077 is harder: 64.41%. Show both or accept any selected trip. Never present the easy example as the overall result.
16. **What did the API actually do?** The Mac sends a trip ID. The Linux server reads the track/network, builds candidates, runs the model/Viterbi, reconstructs roads, then loads reference data separately to score. It returns predictions and aggregate metrics, never reference labels/geometry. The UI result's request ID appears in the receiving terminal log.
17. **What is still unverified?** The physical Mac-to-Linux connection, genuine terminal screenshot and the team's final report review. Linux CPU operation, all ten development HTTP requests and 12 automated tests are verified. Browser screenshots show a real local cross-origin run.
18. **What is novel?** No novelty is claimed for this required baseline. Bhuvan's extension is pending. A later novelty needs a clear hypothesis, a measured baseline ablation and permitted development selection. Do not relabel standard HMM decoding or Bayesian tuning as an invention.

## A practical 3 PM rehearsal

- Spend about 8 minutes on the eleven-slide deck; leave time for a live trip selection and questions.
- Chiranth practises the Mac UI and API address. Deeptanshu practises server startup and matching request IDs. Darshan can lead model-table interpretation; Bhuvan can lead method limitations and any extension he actually supplies. These are suggested speaking roles, not claims about past contribution.
- Every member should trace one observation through raw GPS -> candidates -> features -> model scores -> Viterbi -> returned arc ID.
- Compare `00000052` with `00000077`; explain the trade-off without inventing a cause.
- Capture actual two-laptop evidence and review the PDF's open-evidence section before submission. Keep AI assistance disclosed.

## Reserved results added after the freeze

The supplied ten evaluator trips contain 19,674 observations. The unchanged predictor achieved classifier/HMM proxy accuracy 88.44% / 77.78%; reconstructed-route overlap F1 is 0.9250 versus classifier-only 0.9160. Bridged ordered LCS F1 falls 0.7630 -> 0.7547. These failures are final results, not a tuning signal. Commit a61ce50 and the freeze hashes establish the boundary.
