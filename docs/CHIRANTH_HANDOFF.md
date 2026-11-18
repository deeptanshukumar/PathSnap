# Chiranth's Mac handoff

**Your role: run the UI on your Mac and connect it to Deeptanshu's Linux laptop, which hosts the trained model and API.**

Repository: https://github.com/deeptanshukumar/PathSnap

You need Git, Python 3 and a browser. The UI uses Python's built-in web server; it needs no pip packages, raw datasets, GPU or retraining on your Mac. The report, presentation and final model are already in Git.

## 1. Clone and start the UI

Run in Terminal:

```bash
git clone https://github.com/deeptanshukumar/PathSnap.git
cd PathSnap
python3 --version
python3 -m http.server 8010 --bind 127.0.0.1 --directory ui
```

Keep this Terminal running. Open **http://localhost:8010** in your browser.

If you have already cloned the repo, run `git pull --ff-only` from its root before starting the UI. If that command reports local changes or diverged history, keep those changes and resolve them with the team; do not reset the checkout blindly.

## 2. Connect to the Linux API

Both laptops must be on a network that permits device-to-device connections.

1. In the UI's **Model server · Laptop 1** field, enter **`http://10.20.206.56:8000`**.
2. Click **Connect to API**.
3. Confirm the badge says **Reserved trips · Evaluation mode** and the dropdown contains ten trips.
4. Select a trip and click **Reconstruct route**.
5. Confirm the map, observation count, metrics and request ID appear.

The address above responded successfully from the Linux laptop when this handoff was prepared. It may change after switching Wi-Fi or hotspots; ask Deeptanshu for the current Linux LAN IP if needed. Physical Mac-to-Linux connectivity still needs your test.

**Use the Linux LAN address in the Model server field.** `localhost` or `127.0.0.1` on your Mac refers to your Mac, where the model API is not running. The UI's own browser address remains `http://localhost:8010`.

Quick connectivity check from the Mac browser:

```text
http://10.20.206.56:8000/health
```

Expected response:

```json
{"status":"ready","mode":"evaluation","trip_count":10,"model":"hist_gradient_boosting","tuned":true}
```

The ten reserved IDs are:

```text
00000015  00000017  00000023  00000030  00000049
00000059  00000067  00000073  00000087  00000094
```

Each selection sends only its trip ID. The Linux API performs live candidate scoring, Viterbi decoding, route reconstruction and separate reference scoring. Reference labels and reference geometry stay on the server.

## 3. Complete the real two-laptop evidence

- Capture your Mac's working UI, including the evaluation-mode badge, selected trip, results and request ID.
- On the Linux laptop, capture the actual receiving terminal showing the matching `PATHSNAP` request ID and successful request log. The current background server writes to `/tmp/pathsnap-api.log`; Deeptanshu can display it with `tail -f /tmp/pathsnap-api.log` in a visible terminal.
- Save the actual receiving-terminal screenshot as `reports/evidence/api_terminal.png` on the Linux checkout. Keep the Mac screenshot separately, for example `reports/evidence/ui_mac.png`.
- Tell Deeptanshu once the Mac connection works so the project state and report can record genuine two-device verification.

The existing report openly marks the physical Mac test and receiving-terminal screenshot as pending. Existing browser screenshots prove local cross-origin operation only. After capturing the real evidence, regenerate the report on Linux using the README's report command, then commit and push the updates.

## 4. If something fails

| Symptom | Next step |
|---|---|
| `python3` or `git` is missing | Resolve that prerequisite on the Mac before starting. No ML environment is needed for the UI. |
| Port 8010 is already in use | Use `python3 -m http.server 8012 --bind 127.0.0.1 --directory ui`, then open `http://localhost:8012`. Keep the Model server URL on port 8000. |
| `/health` does not open on the Mac | Check the Linux laptop is awake, the API is running, and its LAN IP is current. Check both devices' network. Campus Wi-Fi may isolate clients; a shared network/hotspot can be tested with the team. |
| `/health` works but the UI cannot connect | Re-enter the same API URL including `http://` and `:8000`, then click Connect. Check browser Console/Network errors if it still fails. |
| Wrong trip IDs or development-mode badge | Ask Deeptanshu to run the API in evaluation mode, then reconnect the UI. |
| Metrics load but map tiles are blank | Check internet access. Folium's display libraries and OpenStreetMap tiles use external resources. |
| An old map URL returns 404 | Reconstruct the trip again; the server retains only a bounded number of recent maps. |
| A trip scores poorly or has many resets | Keep the actual result. Resets and failures are measured behaviour; do not alter predictions or tune against evaluator routes. |

If the Linux API needs restarting, Deeptanshu should run this **from the Linux repository root**, keeping the receiving terminal visible:

```bash
.venv/bin/python -m pathsnap.cli serve \
  --mode evaluation --evaluation-root data/evaluator_test \
  --host 0.0.0.0 --port 8000
```

Cloning on the Mac does not copy `data/evaluator_test/`, `.venv/` or caches; they are deliberately excluded from Git. Those files are already present on the Linux host. Do not start another API on port 8000 if the existing one is healthy.

## 5. Presentation and viva

- [Presentation: 11 editable slides](../reports/PathSnap_Presentation_Final.pptx)
- [Technical report: 18 pages](../reports/PathSnap_Technical_Report.pdf)
- [Viva questions and explanations](VIVA.md)
- [Current implementation and verification](PROJECT_STATE.md)

Practise selecting any reserved trip and explaining the request ID, GPS layer, predicted route, resets and metrics. A high score on one trip is not the overall result.

The system has five compared models, 12 Bayesian search trials and 12 passing tests. All ten development trips and ten reserved trips succeeded over HTTP. On reserved trips, classifier/HMM **proxy** pointwise accuracy is **88.44% / 77.78%**; reconstructed-route overlap F1 is **0.9250**. Pointwise labels are inferred from ordered reference routes. HMM improves some route metrics and hurts others; the report retains that trade-off.

The predictor was frozen in commit `a61ce50` before the reserved archive was opened. Do not retrain, tune or change features/decoder settings using evaluator results. Bhuvan's novelty is pending; no novelty is claimed for the baseline. Retain the report's AI assistance disclosure and review the explanations as a team.

## 6. If you use a coding agent on the Mac

Have it read `AGENTS.md`, this handoff, `docs/PROJECT_STATE.md` and `docs/DECISIONS.md` first. The immediate task is Mac UI startup, connectivity and genuine demonstration evidence. Preserve the frozen predictor and existing results. Use a branch for code changes and coordinate with the team before touching shared experiment artifacts.

The repository is public for cloning. Pushing changes requires GitHub write access; cloning alone does not grant it. Do not put credentials in the repo.
