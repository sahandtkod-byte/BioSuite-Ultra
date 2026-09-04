"""Interactive biological_plots regression: input mocked, Agg figures closed."""
import builtins

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from biosuite.plotting import biological_plots as bp


def _wrap_inputs(*answers):
    answers = list(answers)
    def fake_input(prompt=''):
        if answers:
            return answers.pop(0)
        raise EOFError
    return fake_input


FNS = {
    'volcano_plot': ['n'],
    'pca_plot': [''],
    'ma_plot': [''],
    'venn_diagram': ['y', 'n'],
    'barplot_custom': ['', ''],
    'boxplot_custom': ['', ''],
    'heatmap_custom': ['', ''],
    'scatter_custom': ['', ''],
    'timeseries_plot': ['', ''],
    'qq_plot': ['', ''],
    'clustered_heatmap': ['', ''],
    'circos_plot': ['y'],
    'alignment_viewer': ['n'],
    'violin_plot': [''],
    'raincloud_plot': [''],
    'ridge_plot': [''],
    'dot_plot': [''],
    'manhattan_plot': [''],
}


@pytest.mark.parametrize('name,answers', list(FNS.items()),
                         ids=list(FNS.keys()))
def test_interactive_fn_runs(monkeypatch, name, answers):
    monkeypatch.setattr(builtins, 'input', _wrap_inputs(*answers))
    getattr(bp, name)()
    plt.close('all')


def test_export_all_to_folder(monkeypatch, tmp_path):
    outdir = str(tmp_path / 'exports')
    monkeypatch.setattr(builtins, 'input', _wrap_inputs(outdir))
    monkeypatch.chdir(tmp_path)
    bp.export_all_to_folder()
    planner = tmp_path / 'biosuite_export'
    src = planner if planner.exists() else tmp_path
    assert any(src.glob('*.png')) or any(tmp_path.glob('**/*.png'))
    plt.close('all')


def test_batch_export_to_pdf(monkeypatch):
    monkeypatch.setattr(builtins, 'input', _wrap_inputs(''))
    bp.batch_export_to_pdf(['volcano_plot', 'ma_plot'])
    plt.close('all')


def test_generate_markdown_story(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(builtins, 'input', _wrap_inputs('y'))
    bp.generate_markdown_story(['volcano_plot'])   # returns None; writes md
    plt.close('all')
