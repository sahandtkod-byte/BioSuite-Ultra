"""
HTML report generator — combine results, plots, and tables into styled reports.
"""
import os
import base64
import io
from datetime import datetime


def _img_to_base64(fig):
    """Convert a matplotlib figure to base64 PNG string."""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=plt.rcParams.get('figure.facecolor', 'white'))
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def _df_to_html_table(df, max_rows=50):
    """Convert a pandas DataFrame to styled HTML table."""
    if df is None or (hasattr(df, 'empty') and df.empty):
        return "<p>No data</p>"
    if hasattr(df, 'head'):
        display_df = df.head(max_rows)
    else:
        display_df = df
    return display_df.to_html(classes='data-table', index=False, border=0)


class ReportSection:
    """A section in an HTML report."""

    def __init__(self, title, content="", level=2):
        self.title = title
        self.content = content
        self.level = level

    def to_html(self):
        return f'<h{self.level}>{self.title}</h{self.level}>\n{self.content}\n'


class HTMLReport:
    """Build a styled HTML report."""

    CSS = """
    body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f8f9fa; color: #333; }
    h1 { color: #00ff88; border-bottom: 2px solid #00ff88; padding-bottom: 10px; }
    h2 { color: #00cc6a; margin-top: 30px; }
    h3 { color: #555; }
    .meta { color: #888; font-size: 0.9em; margin-bottom: 20px; }
    .card { background: white; border-radius: 8px; padding: 20px; margin: 15px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .data-table { border-collapse: collapse; width: 100%; font-size: 0.85em; }
    .data-table th { background: #00ff88; color: black; padding: 8px 12px; text-align: left; }
    .data-table td { padding: 6px 12px; border-bottom: 1px solid #eee; }
    .data-table tr:hover { background: #f0f0f0; }
    .plot { text-align: center; margin: 20px 0; }
    .plot img { max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .stat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; }
    .stat-box { background: white; border-radius: 8px; padding: 15px; text-align: center;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .stat-value { font-size: 2em; font-weight: bold; color: #00ff88; }
    .stat-label { color: #888; font-size: 0.85em; }
    .toc { background: white; border-radius: 8px; padding: 15px 25px; margin: 20px 0; }
    .toc a { text-decoration: none; color: #00cc6a; }
    .toc a:hover { text-decoration: underline; }
    .toc ul { list-style: none; padding-left: 0; }
    .toc li { padding: 3px 0; }
    .error { background: #ffe0e0; border-left: 4px solid red; padding: 10px 15px; border-radius: 4px; }
    .success { background: #e0ffe0; border-left: 4px solid green; padding: 10px 15px; border-radius: 4px; }
    """

    def __init__(self, title="BioSuite Report", subtitle=""):
        self.title = title
        self.subtitle = subtitle
        self.sections = []
        self.toc_entries = []
        self.stats = {}

    def add_section(self, title, content="", level=2):
        section = ReportSection(title, content, level)
        self.sections.append(section)
        self.toc_entries.append((title, level))
        return self

    def add_plot(self, fig, title="", caption=""):
        b64 = _img_to_base64(fig)
        html = f'<div class="plot"><img src="data:image/png;base64,{b64}" alt="{title}">'
        if caption:
            html += f'<p><em>{caption}</em></p>'
        html += '</div>'
        self.add_section(title, html)
        return self

    def add_table(self, df, title="", max_rows=50):
        html = _df_to_html_table(df, max_rows)
        self.add_section(title, html)
        return self

    def add_text(self, text, title=""):
        if title:
            self.add_section(title, f"<p>{text}</p>")
        else:
            if self.sections:
                self.sections[-1].content += f"\n<p>{text}</p>"
            else:
                self.add_section("Report", f"<p>{text}</p>")
        return self

    def add_stats(self, stats_dict):
        self.stats.update(stats_dict)
        return self

    def add_error(self, message):
        self.add_section("Error", f'<div class="error">{message}</div>')
        return self

    def add_success(self, message):
        self.add_section("Status", f'<div class="success">{message}</div>')
        return self

    def _build_toc(self):
        if not self.toc_entries:
            return ""
        items = []
        for title, level in self.toc_entries:
            anchor = title.lower().replace(" ", "-").replace("/", "")
            items.append(f'<li>{"&nbsp;" * (level-2) * 4}<a href="#{anchor}">{title}</a></li>')
        return f'<div class="toc"><h3>Contents</h3><ul>{"".join(items)}</ul></div>'

    def _build_stats_grid(self):
        if not self.stats:
            return ""
        boxes = []
        for k, v in self.stats.items():
            boxes.append(f'<div class="stat-box"><div class="stat-value">{v}</div>'
                         f'<div class="stat-label">{k}</div></div>')
        return f'<div class="stat-grid">{"".join(boxes)}</div>'

    def to_html(self):
        sections_html = []
        for section in self.sections:
            anchor = section.title.lower().replace(" ", "-").replace("/", "")
            sections_html.append(f'<div id="{anchor}">{section.to_html()}</div>')

        html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>{self.title}</title>
<style>{self.CSS}</style>
</head><body>
<h1>{self.title}</h1>
<div class="meta">{self.subtitle} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
{self._build_stats_grid()}
{self._build_toc()}
<div class="card">
{"".join(sections_html)}
</div>
</body></html>"""
        return html

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_html())
        return path


def create_report(title="BioSuite Report", subtitle=""):
    return HTMLReport(title, subtitle)


def generate_pipeline_report(pipeline, output_path="pipeline_report.html"):
    """Generate an HTML report from a pipeline run."""
    report = HTMLReport(f"Pipeline: {pipeline.name}", "Automated pipeline execution")
    report.add_stats({
        "Steps": len(pipeline.steps),
        "Results": len(pipeline.results),
    })
    for step in pipeline.steps:
        status = "success" if step.status == "done" else "error"
        content = f'<div class="{status}">Status: {step.status} | Time: {step.elapsed:.2f}s</div>'
        if step.error:
            content += f'<div class="error">{step.error}</div>'
        report.add_section(step.name, content)
    report.add_text("\n".join(pipeline._log), "Execution Log")
    report.save(output_path)
    return output_path


def generate_batch_report(processor, output_path="batch_report.html"):
    """Generate an HTML report from batch processing results."""
    report = HTMLReport(f"Batch: {processor.name}", "Batch processing report")
    n_done = sum(1 for j in processor.jobs if j.status == "done")
    n_fail = sum(1 for j in processor.jobs if j.status == "failed")
    report.add_stats({"Total": len(processor.jobs), "Done": n_done, "Failed": n_fail})
    for job in processor.jobs:
        status = "success" if job.status == "done" else "error"
        content = f'<div class="{status}">{job.sample_id} — {job.status} ({job.elapsed:.2f}s)</div>'
        if job.error:
            content += f'<div class="error">{job.error}</div>'
        report.add_section(job.sample_id, content, level=3)
    report.save(output_path)
    return output_path
