# Project state

## Target and plan
Team PT-088: Chiranth R (PES1UG24AM354), Darshan S Govekar (PES1UG24AM355), Daruru Bhuvan Sai (PES1UG24AM356), Deeptanshu Kumar (PES1UG24AM357). Confirm final spelling before submission.

Hard deadline supplied by team: 18 November 2026, 15:00 IST. Faculty email lists later evaluation; use the team's earlier deadline.

1. Inspect rules, raw formats, geographic calculations and label ambiguity.
2. Implement and verify modular loading, candidate features, reference alignment and grouped experiments.
3. Execute five-model comparison, TPE tuning and classifier/HMM comparison.
4. Integrate server, Mac UI and Folium maps; exercise all ten development trips.
5. Produce evidence-backed PDF, presentation and viva preparation. Publish source and required artifacts after checking repository contents.

## Verified exploration
- All five official PDFs read. AI assistance is allowed subject to truthful attribution and team understanding; cross-team sharing and fabricated experiments are prohibited.
- 80 training trips / 208,417 GPS observations; 10 student-test trips / 19,160 GPS observations.
- Four headerless text files per trip; route lengths differ from GPS counts.
- No nonfinite GPS values or non-increasing timestamps in this package. Training maximum gap 153,064 s; student-test maximum gap 210 s.
- Weakly disconnected road networks and 13 training / 1 development reference-route endpoint breaks exist.
- Initial geometry/alignment probes ran on four real trips. Training trip 00000054 has severe GPS/reference disagreement; preserve and quantify this limitation.
- Repository exists at github.com/deeptanshukumar/PathSnap and was private at initial inspection.

## Current milestone
Research and required baseline plan recorded; isolated Python 3.12 environment installation underway. No trained-model result claimed yet.

## External dependencies
- Ten reserved evaluator trips have not been supplied. Do not generate replacement evaluation trips or relabel development trips.
- Chiranth's Mac is not controlled here. Two-laptop operation and receiving-laptop evidence must be verified with the team.
- Bhuvan's novelty is pending; keep extension points modular.
