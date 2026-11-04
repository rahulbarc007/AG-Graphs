"""Shared helpers for the example studies (paths, result bundles)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RESULTS = os.environ.get("MSRNX_RESULTS", os.path.join(ROOT, "results"))

from msrnx.report import Study  # noqa: E402


def study(name: str, title: str, description: str = "") -> Study:
    print("=" * 78)
    print(f"{title}")
    print("=" * 78)
    return Study(name, title, os.path.join(RESULTS, name), description)
