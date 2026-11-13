"""
HTML report generator — combine results, plots, and tables into styled reports.
"""
import base64
import io
import html
import re
from datetime import datetime


def _esc(value) -> str:
    """Escape report data for HTML text and attribute contexts (N14)."""
    return html.escape(str(value), quote=True)


def _slug(title) -> str:
    """An id safe to put in an href, whatever the title contains."""
    slug = re.sub(r"[^a-z0-9_-]", "-", str(title).lower()).strip("-")
    return slug or "section"


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
    """Convert a pandas DataFrame to a styled HTML table.

    Raises:
        ValueError: if *max_rows* is not a positive count — ``head(0)`` used to
            render a header with no rows and no word about it.
        TypeError: if the object is not a table; pandas' own AttributeError used
            to be the report's answer to a list.
    """
    if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows < 1:
        raise ValueError(f"max_rows must be a positive integer, got {max_rows!r}.")
    if df is None or (hasattr(df, 'empty') and df.empty):
        return "<p>No data</p>"
    if not hasattr(df, 'to_html'):
        raise TypeError(
            f"add_table expects a pandas DataFrame, got {type(df).__name__}: {df!r}")
    if not hasattr(df, 'head'):
        # A duck-typed table object is emitted as it stands: there is no row
        # count to slice and none to announce.
        return df.to_html(classes='data-table', index=False, border=0)
    total = len(df)
    display_df = df.head(max_rows)
    table = display_df.to_html(classes='data-table', index=False, border=0)
    if total > len(display_df):
        table += (f'<p class="meta">Showing {len(display_df)} of {total} rows; '
                  f'the table was truncated at max_rows={max_rows}.</p>')
    return table


class ReportSection:
    """A section in an HTML report.

    ``content`` is markup on purpose — ``add_plot`` and ``add_table`` build it —
    while ``title`` is data and therefore escaped.

    ``level`` is a heading *number*, validated as one: it is interpolated into
    ``<h{level}>``, so a string reached the tag unescaped and any number outside
    1-6 produced a tag no browser knows.
    """

    def __init__(self, title, content="", level=2):
        if not isinstance(level, int) or isinstance(level, bool) or not 1 <= level <= 6:
            raise ValueError(
                f"section level must be an integer from 1 to 6, got {level!r} "
                f"({type(level).__name__}).")
        self.title = title
        self.content = content
        self.level = level
        # The id HTMLReport gives this section inside the document; assigned by
        # add_section, which is the only place that knows about the other
        # sections and can therefore keep the ids unique.
        self.anchor = _slug(title)

    def to_html(self):
        return f'<h{self.level}>{_esc(self.title)}</h{self.level}>\n{self.content}\n'


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
        self.stats = {}
        self._anchors = {}

    def add_section(self, title, content="", level=2):
        section = ReportSection(title, content, level)
        section.anchor = self._unique_anchor(title)
        self.sections.append(section)
        return self

    def _unique_anchor(self, title):
        """One id per section, because every section gets ``id=`` and the TOC a link.

        Two sections called "Results" used to share `id="results"`, so both
        contents were reachable from the same anchor and only the first was —
        invalid HTML, and every link in the table of contents jumped to the same
        place.  A captionless plot had the empty title, which slugged to
        "section" for every one of them.
        """
        base = _slug(title)
        count = self._anchors.get(base, 0) + 1
        self._anchors[base] = count
        return base if count == 1 else f"{base}-{count}"

    def add_plot(self, fig, title="", caption=""):
        b64 = _img_to_base64(fig)
        html_out = f'<div class="plot"><img src="data:image/png;base64,{b64}" ' \
                   f'alt="{_esc(title)}">'
        if caption:
            html_out += f'<p><em>{_esc(caption)}</em></p>'
        html_out += '</div>'
        self.add_section(title, html_out)
        return self

    def add_table(self, df, title="", max_rows=50):
        html = _df_to_html_table(df, max_rows)
        self.add_section(title, html)
        return self

    def add_text(self, text, title=""):
        if title:
            self.add_section(title, f"<p>{_esc(text)}</p>")
        else:
            if self.sections:
                self.sections[-1].content += f"\n<p>{_esc(text)}</p>"
            else:
                self.add_section("Report", f"<p>{_esc(text)}</p>")
        return self

    def add_stats(self, stats_dict):
        self.stats.update(stats_dict)
        return self

    def add_error(self, message):
        self.add_section("Error", f'<div class="error">{_esc(message)}</div>')
        return self

    def add_success(self, message):
        self.add_section("Status", f'<div class="success">{_esc(message)}</div>')
        return self

    def _build_toc(self):
        if not self.sections:
            return ""
        items = []
        for section in self.sections:
            items.append(f'<li>{"&nbsp;" * (section.level-2) * 4}'
                         f'<a href="#{section.anchor}">{_esc(section.title)}</a></li>')
        return f'<div class="toc"><h3>Contents</h3><ul>{"".join(items)}</ul></div>'

    def _build_stats_grid(self):
        if not self.stats:
            return ""
        boxes = []
        for k, v in self.stats.items():
            boxes.append(f'<div class="stat-box"><div class="stat-value">{_esc(v)}</div>'
                         f'<div class="stat-label">{_esc(k)}</div></div>')
        return f'<div class="stat-grid">{"".join(boxes)}</div>'

    def to_html(self):
        sections_html = []
        for section in self.sections:
            sections_html.append(f'<div id="{section.anchor}">'
                                 f'{section.to_html()}</div>')

        document = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>{_esc(self.title)}</title>
<style>{self.CSS}</style>
</head><body>
<h1>{_esc(self.title)}</h1>
<div class="meta">{_esc(self.subtitle)} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
{self._build_stats_grid()}
{self._build_toc()}
<div class="card">
{"".join(sections_html)}
</div>
</body></html>"""
        return document

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_html())
        return path


def create_report(title="BioSuite Report", subtitle=""):
    return HTMLReport(title, subtitle)


def _status_block(status, elapsed, error):
    """One step/job row: green for done, red for failed, plain for never ran.

    Anything that was not ``done`` used to be coloured ``error``, so a pipeline
    stopped on its second step reported the remaining steps as errors and the
    batch report called a queued job a failure.
    """
    style = "success" if status == "done" else "error" if status == "failed" else ""
    klass = f' class="{style}"' if style else ""
    block = (f'<div{klass}>Status: {_esc(status)} '
             f'| Time: {elapsed:.2f}s</div>')
    if error:
        block += f'<div class="error">{_esc(str(error))}</div>'
    return block


def generate_pipeline_report(pipeline, output_path="pipeline_report.html"):
    """Generate an HTML report from a pipeline run."""
    report = HTMLReport(f"Pipeline: {pipeline.name}", "Automated pipeline execution")
    report.add_stats({
        "Steps": len(pipeline.steps),
        "Results": len(pipeline.results),
    })
    for step in pipeline.steps:
        # The title is passed as data: ReportSection escapes it once.  Escaping
        # it here as well turned "alpha & co" into "alpha &amp;amp; co" on the
        # page and an insertion code into markup.
        report.add_section(str(step.name), _status_block(step.status, step.elapsed,
                                                         step.error))
    # The log is markup with data inside it, so each line is escaped and joined
    # with a real break; add_text would have escaped the breaks themselves and
    # the whole log would render as one line of "&lt;br&gt;".
    report.add_section("Execution Log",
                       "<p>" + "<br>".join(_esc(line) for line in pipeline._log) + "</p>")
    report.save(output_path)
    return output_path


def generate_batch_report(processor, output_path="batch_report.html"):
    """Generate an HTML report from batch processing results."""
    report = HTMLReport(f"Batch: {processor.name}", "Batch processing report")
    n_done = sum(1 for j in processor.jobs if j.status == "done")
    n_fail = sum(1 for j in processor.jobs if j.status == "failed")
    report.add_stats({"Total": len(processor.jobs), "Done": n_done, "Failed": n_fail})
    for job in processor.jobs:
        report.add_section(str(job.sample_id),
                           _status_block(job.status, job.elapsed, job.error),
                           level=3)
    report.save(output_path)
    return output_path
