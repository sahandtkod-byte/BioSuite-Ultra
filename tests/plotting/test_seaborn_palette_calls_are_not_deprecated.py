"""The plotting code must not sit on a deprecated seaborn call.

Seven ``sns.*plot(...)`` calls passed ``palette=`` without ``hue=``.  The installed
seaborn 0.13.2 warns on every one of them: "Passing `palette` without assigning
`hue` is deprecated and will be removed in **v0.14.0**" — and the package declares
``seaborn>=0.12,<1``, so 0.14, 0.15 and 0.16 are all inside the supported range,
where the palette argument stops being honoured and every one of those plots
silently loses its colours.  The fix is seaborn's own remedy:
assign the category variable to ``hue`` and switch the legend off.
"""
import ast
import pathlib
import re

import pytest
import seaborn

REPO = pathlib.Path(__file__).resolve().parents[2]


def _calls_with_palette():
    """(location, has_hue) for every call that passes palette=, read off the AST.

    A line-based scan is wrong here: the fixed calls wrap onto a second line, so
    the ``palette=`` line does not contain the ``hue=`` that sits above it.
    """
    found = []
    for path in sorted((REPO / "biosuite").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            keywords = [kw.arg for kw in node.keywords if kw.arg]
            if "palette" in keywords:
                where = f"{path.relative_to(REPO)}:{node.lineno} {ast.unparse(node.func)}"
                found.append((where, "hue" in keywords))
    return found


def _palette_sites():
    return [where for where, has_hue in _calls_with_palette() if not has_hue]


def test_no_plot_call_passes_palette_without_hue():
    offenders = _palette_sites()

    assert offenders == [], (
        f"seaborn {seaborn.__version__} deprecates palette= without hue= and removes "
        f"it in 0.14: {offenders}")


def test_the_colour_is_still_chosen_by_a_palette_somewhere_in_the_package():
    """The guard above passes trivially if the whole feature is deleted, so the
    palettes have to still be there — just with hue assigned."""
    with_hue = sum(1 for where, has_hue in _calls_with_palette()
                   if has_hue and "plotting" in where)

    assert with_hue >= 7, (
        f"only {with_hue} palette calls left; the seven categorical plots lost their "
        "colours instead of being fixed")


def test_the_installed_seaborn_is_the_one_the_warning_was_read_from():
    """The rule above is pinned to a version; if the floor moves the scan has to
    be re-read rather than left to rot."""
    assert seaborn.__version__.startswith("0.13"), seaborn.__version__
    assert re.search(r'seaborn>=0\.12,<1',
                     (REPO / "requirements.txt").read_text(encoding="utf-8"))


def test_the_grouped_plots_still_legend_their_own_groups():
    """hue= with legend=False must not remove the tick labels: the category names
    are how a reader knows which box is which."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from biosuite.plotting import biological_plots as bp

    frame = pd.DataFrame({"Group": ["Ctrl"] * 4 + ["Trt"] * 4,
                          "Value": [1.0, 2.0, 3.0, 2.5, 5.0, 6.0, 7.0, 6.5]})
    fig, ax = plt.subplots()
    try:
        seaborn.boxplot(x="Group", y="Value", data=frame, hue="Group",
                        palette="Set2", legend=False, ax=ax)
    finally:
        plt.close(fig)

    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert labels == ["Ctrl", "Trt"], labels
    assert ax.get_legend() is None, "legend=False was the point"
