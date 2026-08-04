"""Phase G — Report Export Engine (Day-13, Layer 13).

Render any report (the uniform typed-section structure from :mod:`src.reporting.base`) to multiple
formats. Because sections are *typed* (``text`` / ``metrics`` / ``list`` / ``table``), the exporter
is a single **fold over the sections dispatching on ``kind``** — one code path renders every report
type. JSON / Markdown / HTML are **deps-free** (stdlib string templates); PDF is **optional and
graceful** (rendered only if a PDF library is importable, else skipped — the HTML is print-ready).

Deterministic (the rendered content is a pure function of the report). No new dependency required.

Example
-------
>>> exp = ReportExporter()
>>> md = exp.to_markdown(report)
>>> md.startswith("# ")
True
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Optional

from src.utils.config import load_config
from src.utils.logger import get_logger


class ReportExporter:
    """Render reports to JSON / Markdown / HTML (PDF optional-graceful).

    Parameters
    ----------
    config, logger:
        Standard config + logger.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None, logger=None) -> None:
        self.config = config if config is not None else load_config()
        self.logger = logger or get_logger("avip.reporting.export")

        c = self.config.get("day13", {}).get("export", {})
        self.out_dir = Path(c.get("out_dir", "outputs/reports"))
        self.formats = list(c.get("formats", ["json", "markdown", "html"]))

    # ------------------------------------------------------------------ formats
    def to_json(self, report: dict[str, Any]) -> str:
        return json.dumps(report, indent=2, default=str)

    def to_markdown(self, report: dict[str, Any]) -> str:
        camp = report.get("campaign") or {}
        lines = [f"# {report['report_type']}",
                 f"*{camp.get('name', '')}* — generated {report.get('generated_utc', '')}", ""]
        for s in report.get("sections", []):
            lines += [f"## {s['title']}", f"*source: {s['source']}*", ""]
            lines += self._md_body(s["kind"], s["body"]) + [""]
        dfr = report.get("deferred_appendix")
        if dfr:
            lines += ["### Not covered (no data — deferred, not fabricated)",
                      f"*{dfr.get('note', '')}*", ""]
            lines += [", ".join(dfr.get("items", []))] + [""]
        if report.get("honesty_note"):
            lines += [f"> {report['honesty_note']}"]
        return "\n".join(lines)

    def to_html(self, report: dict[str, Any]) -> str:
        camp = report.get("campaign") or {}
        parts = [f"<h1>{_e(report['report_type'])}</h1>",
                 f"<p class='sub'>{_e(str(camp.get('name', '')))} — generated "
                 f"{_e(str(report.get('generated_utc', '')))}</p>"]
        for s in report.get("sections", []):
            parts += [f"<section><h2>{_e(s['title'])}</h2>",
                      f"<p class='src'>source: {_e(s['source'])}</p>"]
            parts += self._html_body(s["kind"], s["body"]) + ["</section>"]
        dfr = report.get("deferred_appendix")
        if dfr:
            parts += ["<section class='deferred'><h3>Not covered (no data — deferred, not fabricated)</h3>",
                      f"<p class='src'>{_e(dfr.get('note', ''))}</p>",
                      "<p>" + _e(", ".join(dfr.get("items", []))) + "</p></section>"]
        if report.get("honesty_note"):
            parts += [f"<footer>{_e(report['honesty_note'])}</footer>"]
        return _HTML_SHELL.format(title=_e(report["report_type"]), body="\n".join(parts))

    def to_pdf(self, report: dict[str, Any], path: Path) -> Optional[Path]:
        """Render to PDF only if a PDF library is importable; otherwise skip gracefully."""
        html_str = self.to_html(report)
        try:
            from weasyprint import HTML  # type: ignore
            HTML(string=html_str).write_pdf(str(path))
            return path
        except Exception:
            self.logger.info("PDF export skipped (no PDF library available); HTML is print-ready.")
            return None

    # ------------------------------------------------------------------ export
    def export(self, report: dict[str, Any], formats: Optional[list[str]] = None,
               out_dir: Optional[Path] = None, basename: Optional[str] = None) -> dict[str, str]:
        """Write the report in each requested format; return {format: path}. PDF added if available."""
        formats = formats if formats is not None else self.formats
        out_dir = Path(out_dir) if out_dir is not None else self.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        base = basename or self._basename(report)

        written: dict[str, str] = {}
        renderers = {"json": self.to_json, "markdown": self.to_markdown, "html": self.to_html}
        ext = {"json": "json", "markdown": "md", "html": "html"}
        for fmt in formats:
            if fmt in renderers:
                p = out_dir / f"{base}.{ext[fmt]}"
                p.write_text(renderers[fmt](report))
                written[fmt] = str(p)
        if "pdf" in formats or self._pdf_available():
            p = self.to_pdf(report, out_dir / f"{base}.pdf")
            if p:
                written["pdf"] = str(p)
        self.logger.info("Exported %s -> %s", report["report_type"], list(written))
        return written

    # --------------------------------------------------------------- internals
    @staticmethod
    def _md_body(kind: str, body: Any) -> list[str]:
        if kind == "text":
            return [str(body)]
        if kind == "metrics":
            return [f"- **{k}:** {v}" for k, v in body.items()]
        if kind == "list":
            return [f"- {it}" for it in body]
        if kind == "table":
            cols = body["columns"]
            out = ["| " + " | ".join(str(c) for c in cols) + " |",
                   "| " + " | ".join("---" for _ in cols) + " |"]
            out += ["| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |"
                    for row in body["rows"]]
            return out
        return [str(body)]

    @staticmethod
    def _html_body(kind: str, body: Any) -> list[str]:
        if kind == "text":
            return [f"<p>{_e(str(body))}</p>"]
        if kind == "metrics":
            rows = "".join(f"<tr><th>{_e(str(k))}</th><td>{_e(str(v))}</td></tr>" for k, v in body.items())
            return [f"<table class='metrics'>{rows}</table>"]
        if kind == "list":
            return ["<ul>" + "".join(f"<li>{_e(str(it))}</li>" for it in body) + "</ul>"]
        if kind == "table":
            head = "".join(f"<th>{_e(str(c))}</th>" for c in body["columns"])
            rows = "".join("<tr>" + "".join(f"<td>{_e(str(x))}</td>" for x in row) + "</tr>"
                           for row in body["rows"])
            return [f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"]
        return [f"<p>{_e(str(body))}</p>"]

    @staticmethod
    def _basename(report: dict[str, Any]) -> str:
        rt = re.sub(r"[^a-z0-9]+", "_", report["report_type"].lower()).strip("_")
        name = re.sub(r"[^a-z0-9]+", "_", str((report.get("campaign") or {}).get("name", "")).lower()).strip("_")
        return f"{rt}__{name}"[:120] if name else rt

    @staticmethod
    def _pdf_available() -> bool:
        try:
            import weasyprint  # type: ignore  # noqa: F401
            return True
        except Exception:
            return False


def _e(s: str) -> str:
    return html.escape(str(s))


_HTML_SHELL = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 820px; margin: 2rem auto;
         padding: 0 1rem; color: #1a1a1a; line-height: 1.5; }}
 h1 {{ border-bottom: 3px solid #2b6cb0; padding-bottom: .3rem; }}
 h2 {{ margin-top: 1.6rem; color: #2b6cb0; }}
 .sub {{ color: #666; }}
 .src {{ color: #999; font-size: .8rem; margin: -.4rem 0 .5rem; }}
 table {{ border-collapse: collapse; width: 100%; margin: .4rem 0; }}
 th, td {{ border: 1px solid #ddd; padding: .35rem .5rem; text-align: left; font-size: .92rem; }}
 table.metrics th {{ width: 40%; background: #f7fafc; }}
 .deferred {{ background: #fffaf0; border-left: 4px solid #dd6b20; padding: .5rem 1rem; }}
 footer {{ margin-top: 2rem; color: #666; font-size: .85rem; border-top: 1px solid #eee; padding-top: .6rem; }}
</style></head><body>
{body}
</body></html>"""
