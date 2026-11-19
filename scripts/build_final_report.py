"""Build reports/PathSnap_Final_Report.pdf from committed artifacts only.

    python scripts/build_final_report.py

Steps: regenerate figures (scripts/report_figures.py) from artifacts/*.json, then lay out the
consolidated report. The raw dataset is not needed. Optional evidence images are inserted
automatically when present: reports/evidence/api_terminal.png, ui_current.png, ui_trace.png.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import report_figures  # noqa: E402
from final_report import s_data, s_front, s_hpo_hmm, s_models, s_rest  # noqa: E402
from final_report.data import ROOT, build  # noqa: E402
from final_report.layout import ReportDoc  # noqa: E402

OUTPUT = ROOT / "reports" / "PathSnap_Final_Report.pdf"


def main() -> None:
    report_figures.main()
    fig = report_figures.OUT
    D = build()
    story = []
    story += s_front.cover(D, fig)
    story += s_front.contents()
    story += s_front.overview(D, fig)
    story += s_data.section(D, fig)
    story += s_models.section(D, fig)
    story += s_hpo_hmm.tuning(D, fig)
    story += s_hpo_hmm.hmm(D, fig)
    story += s_rest.folium_section(D, fig)
    story += s_rest.deployment_section(D, fig)
    story += s_rest.integrity_section(D, fig)
    story += s_rest.extension_section(D, fig)
    story += s_rest.limitations_section(D, fig)
    story += s_rest.reproducibility_section(D, fig)
    story += s_rest.appendix_a(D, fig)
    story += s_rest.appendix_b(D, fig)
    doc = ReportDoc(OUTPUT)
    doc.multiBuild(story)
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({doc.page} pages)")


if __name__ == "__main__":
    main()
