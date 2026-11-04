"""Console tables and result bundles (figures + key numbers) for every study."""
from __future__ import annotations

import base64
import datetime as _dt
import html
import json
import os
from dataclasses import dataclass, field


def print_table(title: str, rows, headers=None, floatfmt: str = "{:.5g}") -> str:
    """Pretty-print a table to stdout and return it as text."""
    def fmt(v):
        if isinstance(v, float):
            return floatfmt.format(v)
        return str(v)

    rows = [[fmt(c) for c in r] for r in rows]
    if headers:
        rows = [list(map(str, headers))] + rows
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [f"\n{title}", line]
    for j, r in enumerate(rows):
        out.append("| " + " | ".join(c.rjust(w) if j or not headers else c.ljust(w)
                                     for c, w in zip(r, widths)) + " |")
        if j == 0 and headers:
            out.append(line)
    out.append(line)
    text = "\n".join(out)
    print(text)
    return text


@dataclass
class Study:
    """Collects the outputs of one example: figures, tables and key results."""
    name: str
    title: str
    outdir: str
    description: str = ""
    figures: list = field(default_factory=list)      # (path, caption)
    tables: list = field(default_factory=list)       # (title, headers, rows)
    results: dict = field(default_factory=dict)

    def __post_init__(self):
        os.makedirs(self.outdir, exist_ok=True)

    def path(self, fname: str) -> str:
        return os.path.join(self.outdir, fname)

    def add_figure(self, path: str, caption: str):
        self.figures.append((os.path.relpath(path, self.outdir), caption))
        print(f"  [figure] {path}  -  {caption}")

    def add_table(self, title: str, headers, rows, floatfmt: str = "{:.5g}"):
        print_table(title, rows, headers, floatfmt)
        conv = [[(floatfmt.format(c) if isinstance(c, float) else str(c)) for c in r] for r in rows]
        self.tables.append((title, list(map(str, headers)), conv))

    def set(self, **kw):
        for k, v in kw.items():
            try:
                json.dumps(v)
                self.results[k] = v
            except TypeError:
                self.results[k] = float(v) if hasattr(v, "__float__") else str(v)

    def save(self):
        data = {"name": self.name, "title": self.title, "description": self.description,
                "figures": self.figures, "tables": self.tables, "results": self.results}
        with open(os.path.join(self.outdir, "study.json"), "w") as f:
            json.dump(data, f, indent=2, default=float)
        print(f"  [saved] {os.path.join(self.outdir, 'study.json')}")


def build_html_report(results_dir: str, out_path: str, title: str = "MSR-NX Neutronics Report"):
    """Assemble every study.json under ``results_dir`` into one HTML report."""
    studies = []
    for d in sorted(os.listdir(results_dir)):
        p = os.path.join(results_dir, d, "study.json")
        if os.path.exists(p):
            with open(p) as f:
                s = json.load(f)
            s["_dir"] = os.path.join(results_dir, d)
            studies.append(s)
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>",
             "<style>body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;background:#f9f9f7;"
             "color:#0b0b0b;margin:0;padding:24px;max-width:1100px;margin:auto}"
             "h1{font-size:26px}h2{border-bottom:1px solid #e1e0d9;padding-bottom:6px;margin-top:40px}"
             "p,li{color:#52514e;line-height:1.5}figure{margin:18px 0;background:#fcfcfb;padding:8px;"
             "border:1px solid rgba(11,11,11,.1);border-radius:8px}img{max-width:100%}"
             "figcaption{color:#52514e;font-size:13px;margin-top:4px}"
             "table{border-collapse:collapse;font-size:13px;margin:10px 0;background:#fcfcfb}"
             "td,th{border:1px solid #e1e0d9;padding:4px 8px;text-align:right;font-variant-numeric:tabular-nums}"
             "th{background:#f0efec}</style></head><body>",
             f"<h1>{html.escape(title)}</h1><p>Generated {_dt.datetime.now():%Y-%m-%d %H:%M}.</p>"]
    for s in studies:
        parts.append(f"<h2>{html.escape(s['title'])}</h2><p>{html.escape(s.get('description', ''))}</p>")
        for t_title, headers, rows in s["tables"]:
            parts.append(f"<p><b>{html.escape(t_title)}</b></p><table><tr>"
                         + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr>")
            for r in rows:
                parts.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>")
            parts.append("</table>")
        for rel, cap in s["figures"]:
            fp = os.path.join(s["_dir"], rel)
            with open(fp, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            parts.append(f"<figure><img src='data:image/png;base64,{b64}' alt='{html.escape(cap)}'>"
                         f"<figcaption>{html.escape(cap)}</figcaption></figure>")
    parts.append("</body></html>")
    with open(out_path, "w") as f:
        f.write("\n".join(parts))
    return out_path


def _md_escape(text: str) -> str:
    return str(text).replace("|", "\\|")


def build_markdown_report(results_dir: str, out_path: str,
                          title: str = "MSR-NX Neutronics Design Report") -> str:
    """Assemble every study.json under ``results_dir`` into a Markdown report
    (figures linked relatively, so it renders on GitHub)."""
    base = os.path.dirname(os.path.abspath(out_path))
    lines = [f"# {title}", "",
             f"_Generated {_dt.datetime.now():%Y-%m-%d %H:%M} by `examples/run_all.py`._", ""]
    studies = []
    for d in sorted(os.listdir(results_dir)):
        p = os.path.join(results_dir, d, "study.json")
        if os.path.exists(p):
            with open(p) as f:
                s = json.load(f)
            s["_dir"] = os.path.join(results_dir, d)
            studies.append(s)
    lines.append("## Contents")
    for s in studies:
        anchor = s["title"].lower()
        anchor = "".join(c for c in anchor if c.isalnum() or c in " -").replace(" ", "-")
        lines.append(f"- [{s['title']}](#{anchor})")
    lines.append("")
    for s in studies:
        lines += [f"## {s['title']}", "", s.get("description", ""), ""]
        for t_title, headers, rows in s["tables"]:
            lines += [f"**{_md_escape(t_title)}**", "",
                      "| " + " | ".join(_md_escape(h) for h in headers) + " |",
                      "|" + "|".join("---" for _ in headers) + "|"]
            lines += ["| " + " | ".join(_md_escape(c) for c in r) + " |" for r in rows]
            lines.append("")
        for rel, cap in s["figures"]:
            img = os.path.relpath(os.path.join(s["_dir"], rel), base)
            lines += [f"![{_md_escape(cap)}]({img})", f"*{cap}*", ""]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    return out_path
