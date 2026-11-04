"""Run every example study and assemble the consolidated reports.

    python examples/run_all.py            # all studies (~20 min on 4 cores)
    python examples/run_all.py 01 03 07   # selected studies

Outputs: results/<study>/*.png, results/<study>/study.json,
         results/REPORT.md (renders on GitHub) and results/report.html.
"""
import os
import runpy
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import RESULTS  # noqa: E402

from msrnx.report import build_html_report, build_markdown_report  # noqa: E402

STUDIES = ["01_nuclear_data", "02_lattice_physics", "03_core_analysis", "04_reactivity_control",
           "05_kinetics_transients", "06_xenon_samarium", "07_burnup_depletion",
           "08_ml_surrogates", "09_parameter_reduction", "10_uq_optimization"]


def main(selected):
    todo = [s for s in STUDIES if not selected or any(s.startswith(x) for x in selected)]
    timings = {}
    for s in todo:
        t = time.time()
        runpy.run_path(os.path.join(HERE, f"{s}.py"), run_name="__main__")
        timings[s] = time.time() - t
        print(f"\n>>> {s} finished in {timings[s]:.1f} s\n")
    md = build_markdown_report(RESULTS, os.path.join(RESULTS, "REPORT.md"))
    html = build_html_report(RESULTS, os.path.join(RESULTS, "report.html"))
    print("\nStudy run times:")
    for s, t in timings.items():
        print(f"  {s:26s} {t:7.1f} s")
    print(f"\nReports written:\n  {md}\n  {html}")


if __name__ == "__main__":
    main(sys.argv[1:])
