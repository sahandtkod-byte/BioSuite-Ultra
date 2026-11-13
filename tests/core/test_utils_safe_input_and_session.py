"""utils safe-input helpers, session persistence, dataframe loaders, plot saving.

Every test that can reach ``save_config``, ``save_session`` or ``autosave_session``
redirects the paths into ``tmp_path`` (see the ``iso`` fixture) and asserts on the
JSON that was actually written; ``builtins.input`` is always monkeypatched, so
nothing here waits on a terminal, and nothing here touches a network or a real
external tool.
"""
import copy
import json
import logging
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytest

from biosuite.core import utils as ut


class ScriptedInput:
    """Answers prompts from a list; records every prompt it was asked."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt=''):
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError("no more scripted answers")
        return self.answers.pop(0)

    @property
    def n_prompts(self):
        return len(self.prompts)


def _never_called(_prompt=''):
    raise AssertionError("the user should not have been prompted")


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    """Redirect config/session into tmp_path; restore the module state after."""
    home = tmp_path / "state"
    home.mkdir()
    cfg_file = home / "biosuite_config.json"
    sess_file = home / "biosuite_session.json"
    monkeypatch.setattr(ut, "get_config_file", lambda: str(cfg_file))
    monkeypatch.setattr(ut, "get_session_file", lambda: str(sess_file))
    monkeypatch.setattr(ut, "user_config_dir", lambda: str(home))

    cfg_snapshot = copy.deepcopy(ut.config)
    session_snapshot = copy.deepcopy(ut.session)
    counter_snapshot = ut._interaction_counter
    config_keys_snapshot = copy.deepcopy(ut.config.get("api_keys", {}))
    others = {p.name for p in tmp_path.rglob("*") if p.is_file() and home not in p.parents}

    import types
    yield types.SimpleNamespace(home=home, cfg=cfg_file, session=sess_file, tmp=tmp_path)

    ut._interaction_counter = counter_snapshot
    ut.config.clear()
    ut.config.update(cfg_snapshot)
    ut.session.clear()
    ut.session.update(session_snapshot)
    # The API-key sub-dict is shared with callers that read it in place.
    ut.config["api_keys"] = config_keys_snapshot
    written = {p.name for p in tmp_path.rglob("*") if p.is_file()
               and home not in p.parents}
    assert written == others, "files were written outside %s: %s" % (home, written ^ others)


@pytest.fixture()
def interactive(monkeypatch):
    """Quiet mode off, so the prompting path is the one under test."""
    ut._interaction_counter = 0
    monkeypatch.setitem(ut.config, "quiet", False)
    return ut.config


# ── safe_float_input ────────────────────────────────────────────────────────

def test_safe_float_valid_and_invalid(interactive, monkeypatch):
    prompts = ScriptedInput(['3.25', 'oops'])
    monkeypatch.setattr('builtins.input', prompts)
    assert ut.safe_float_input('f?', 1.0) == 3.25
    assert ut.safe_float_input('f?', 1.0) == 1.0
    assert prompts.prompts == ['f?', 'f?']


def test_safe_float_empty_answer_keeps_the_default(interactive, monkeypatch):
    monkeypatch.setattr('builtins.input', ScriptedInput(['   ']))
    assert ut.safe_float_input('f?', 2.75) == 2.75


def test_safe_float_remembers_the_previous_answer(interactive, monkeypatch, iso):
    """Only a call with a ``key`` is remembered: the key is what the session stores
    under and what the next prompt is pre-filled from."""
    ut.session.clear()
    ut.session['cutoff'] = 0.5
    prompts = ScriptedInput([''])
    monkeypatch.setattr('builtins.input', prompts)

    assert ut.safe_float_input('cutoff?', 9.0, key='cutoff') == 0.5
    assert prompts.prompts == ['cutoff?'], "the prompt itself is not rewritten"
    assert ut.session['cutoff'] == 0.5, "the empty answer keeps what the session held"

    monkeypatch.setattr('builtins.input', ScriptedInput(['1.5']))
    assert ut.safe_float_input('cutoff?', 9.0, key='cutoff') == 1.5
    assert ut.session['cutoff'] == 1.5
    assert ut._interaction_counter == 2
    assert not iso.session.exists(), "a session is only written on every tenth answer"


def test_an_answer_without_a_key_is_neither_remembered_nor_counted(interactive, monkeypatch, iso):
    ut.session.clear()
    ut.session['cutoff'] = 0.5
    monkeypatch.setattr('builtins.input', ScriptedInput(['']))

    assert ut.safe_float_input('cutoff?', 9.0) == 9.0, "without a key the session cannot be consulted"
    assert ut._interaction_counter == 0
    assert ut.session == {'cutoff': 0.5}


def test_safe_float_autosaves_on_every_tenth_interaction(interactive, monkeypatch, iso):
    ut.session.clear()
    ut._interaction_counter = 9
    ut.session['evalue'] = 1.0
    monkeypatch.setattr('builtins.input', ScriptedInput(['2.0']))

    assert ut.safe_float_input('e?', 0.0, key='evalue') == 2.0

    assert ut._interaction_counter == 10
    assert json.loads(iso.session.read_text(encoding='utf-8')) == {'evalue': 2.0}
    assert not iso.cfg.exists(), "autosaving a session does not touch the config"


def test_safe_float_quiet_never_prompts(monkeypatch, iso):
    monkeypatch.setitem(ut.config, 'quiet', True)
    monkeypatch.setattr('builtins.input', _never_called)
    ut.session['ignored'] = 42.0
    assert ut.safe_float_input('p?', 2.5, key='ignored') == 2.5
    assert not iso.session.exists()


# ── safe_int_input ──────────────────────────────────────────────────────────

def test_safe_int_valid_empty_invalid(interactive, monkeypatch):
    prompts = ScriptedInput(['7', '', 'no'])
    monkeypatch.setattr('builtins.input', prompts)
    assert ut.safe_int_input('i?', 5) == 7
    assert ut.safe_int_input('i?', 5) == 5
    assert ut.safe_int_input('i?', 5) == 5
    assert prompts.n_prompts == 3


def test_safe_int_remembers_and_persists(interactive, monkeypatch, iso):
    ut.session.clear()
    ut.session['reads'] = 3
    monkeypatch.setattr('builtins.input', ScriptedInput(['']))
    assert ut.safe_int_input('n?', 99, key='reads') == 3

    monkeypatch.setattr('builtins.input', ScriptedInput(['11']))
    assert ut.safe_int_input('n?', 99, key='reads') == 11
    assert ut.session['reads'] == 11
    assert not iso.session.exists()

    ut._interaction_counter = 9
    monkeypatch.setattr('builtins.input', ScriptedInput(['12']))
    assert ut.safe_int_input('n?', 99, key='reads') == 12
    assert json.loads(iso.session.read_text(encoding='utf-8')) == {'reads': 12}


def test_safe_int_quiet_never_prompts(monkeypatch):
    monkeypatch.setitem(ut.config, 'quiet', True)
    monkeypatch.setattr('builtins.input', _never_called)
    assert ut.safe_int_input('n?', 3) == 3


# ── safe_list_input ─────────────────────────────────────────────────────────

def test_safe_list_without_a_key_is_not_persisted(interactive, monkeypatch, iso):
    prompts = ScriptedInput(['1, 2, 3'])
    monkeypatch.setattr('builtins.input', prompts)
    assert ut.safe_list_input('ls', float) == [1.0, 2.0, 3.0]
    assert prompts.prompts == ['ls: ']
    assert ut.session == {}
    assert not iso.session.exists()


def test_safe_list_reuses_and_refills_the_session(interactive, monkeypatch, iso):
    monkeypatch.setattr('builtins.input', ScriptedInput(['1,2,3', '']))
    assert ut.safe_list_input('ls', float, key='sizes') == [1.0, 2.0, 3.0]
    assert ut.safe_list_input('ls', float, key='sizes') == [1.0, 2.0, 3.0]
    assert ut.session['sizes'] == [1.0, 2.0, 3.0]


def test_safe_list_invalid_cast_returns_none(interactive, monkeypatch, caplog):
    monkeypatch.setattr('builtins.input', ScriptedInput(['a,b']))
    with caplog.at_level(logging.INFO, logger='biosuite.core.utils'):
        assert ut.safe_list_input('ls', int) is None
    # The old message promised "Using default." while the function returned None —
    # there is no default to use. It names the value that failed now.
    assert [r.getMessage() for r in caplog.records] == [
        "'a' is not a valid int; no list was returned."]


@pytest.mark.parametrize("typed, message", [
    pytest.param('inf', "Not a finite number. Using default: 1.0", id="infinity"),
    pytest.param('-inf', "Not a finite number. Using default: 1.0", id="negative"),
    pytest.param('nan', "Not a finite number. Using default: 1.0", id="nan"),
    pytest.param('1e999', "Not a finite number. Using default: 1.0", id="overflow"),
])
def test_an_infinite_threshold_is_refused_like_a_typo(interactive, monkeypatch, caplog,
                                                     typed, message):
    """Measured on the old code: `float(input())` accepted all four, and a cutoff of
    `nan` makes every comparison on the far side of it False — the analysis reads as
    "nothing passed" rather than "you typed nonsense"."""
    monkeypatch.setattr('builtins.input', ScriptedInput([typed]))
    with caplog.at_level(logging.INFO, logger='biosuite.core.utils'):
        assert ut.safe_float_input('f', 1.0) == 1.0
    assert [r.getMessage() for r in caplog.records] == [message]


def test_a_stored_session_value_that_is_not_a_number_is_not_offered_as_one(
        interactive, monkeypatch, caplog):
    """The session file is JSON a user can edit; the old code returned whatever was
    in it from a function annotated `-> float`."""
    monkeypatch.setitem(ut.session, 'cutoff', "high")
    monkeypatch.setattr('builtins.input', ScriptedInput(['2.5']))
    with caplog.at_level(logging.INFO, logger='biosuite.core.utils'):
        assert ut.safe_float_input('f', 1.0, key='cutoff') == 2.5
    assert "Stored value for 'cutoff' is 'high', not a finite number; using the " \
        "default instead." in caplog.text
    assert ut.session['cutoff'] == 2.5


def test_a_stored_session_value_that_is_a_number_is_still_offered(interactive,
                                                                 monkeypatch):
    monkeypatch.setitem(ut.session, 'cutoff', 0.25)
    monkeypatch.setattr('builtins.input', ScriptedInput(['']))
    assert ut.safe_float_input('f', 1.0, key='cutoff') == 0.25


def test_a_blank_line_is_said_to_be_blank(interactive, monkeypatch, caplog):
    monkeypatch.setattr('builtins.input', ScriptedInput(['   ']))
    with caplog.at_level(logging.INFO, logger='biosuite.core.utils'):
        assert ut.safe_list_input('ls', float) is None
    assert [r.getMessage() for r in caplog.records] == ["No list entered."]


def test_a_stored_value_that_is_not_a_list_is_reported_and_ignored(interactive,
                                                                   monkeypatch,
                                                                   caplog):
    """The prompt line builds `(last: …)` by joining the stored value.  A session
    file with a number under that key used to raise `TypeError: 'int' object is
    not iterable` out of the input helper, taking the whole module down with it."""
    monkeypatch.setitem(ut.session, 'sizes', 5)
    prompts = ScriptedInput(['2,4'])
    monkeypatch.setattr('builtins.input', prompts)
    with caplog.at_level(logging.INFO, logger='biosuite.core.utils'):
        assert ut.safe_list_input('ls', float, key='sizes') == [2.0, 4.0]
    assert [r.getMessage() for r in caplog.records] == [
        "Stored value for 'sizes' is 5, not a list; no previous answer is shown."]
    assert prompts.prompts == ['ls: '], "no unusable hint is shown"


def test_a_stored_string_is_not_shown_one_character_at_a_time(interactive,
                                                              monkeypatch,
                                                              caplog):
    """`','.join` over a str walks its characters, so a session holding "1,2"
    advertised the previous answer as `1,,,2`."""
    monkeypatch.setitem(ut.session, 'sizes', '1,2')
    monkeypatch.setattr('builtins.input', ScriptedInput(['']))
    with caplog.at_level(logging.INFO, logger='biosuite.core.utils'):
        assert ut.safe_list_input('ls', float, key='sizes') is None
    assert "Stored value for 'sizes' is '1,2', not a list" in caplog.text


def test_a_blank_line_reuses_a_stored_list_without_leaking_it(interactive, monkeypatch):
    """The stored list is handed back as a copy: a caller that sorts or truncates
    the answer it got must not rewrite the session behind the helper's back."""
    monkeypatch.setitem(ut.session, 'sizes', [3.0, 1.0])
    monkeypatch.setattr('builtins.input', ScriptedInput(['']))
    got = ut.safe_list_input('ls', float, key='sizes')
    assert got == [3.0, 1.0]
    got.sort()
    assert ut.session['sizes'] == [3.0, 1.0]


def test_safe_list_empty_without_history_returns_none(interactive, monkeypatch):
    monkeypatch.setattr('builtins.input', ScriptedInput(['']))
    assert ut.safe_list_input('ls', float) is None


def test_safe_list_quiet_returns_none(monkeypatch):
    monkeypatch.setitem(ut.config, 'quiet', True)
    monkeypatch.setattr('builtins.input', _never_called)
    assert ut.safe_list_input('ls', float, key='sizes') is None


def test_safe_list_autosaves_on_every_tenth_interaction(interactive, monkeypatch, iso):
    ut._interaction_counter = 9
    monkeypatch.setattr('builtins.input', ScriptedInput(['4,5']))
    assert ut.safe_list_input('ls', int, key='idx') == [4, 5]
    assert ut._interaction_counter == 10
    assert json.loads(iso.session.read_text(encoding='utf-8')) == {'idx': [4, 5]}


# ── load_dataframe_safe ─────────────────────────────────────────────────────

@pytest.fixture()
def frame():
    return pd.DataFrame({'gene': ['a', 'b'], 'expr': [1.5, 2.5]})


@pytest.mark.parametrize("name,writer", [
    ("d.csv", lambda df, p: df.to_csv(p, index=False)),
    ("d.tsv", lambda df, p: df.to_csv(p, sep='\t', index=False)),
    ("d.json", lambda df, p: df.to_json(p, orient='records')),
    ("d.txt", lambda df, p: df.to_csv(p, sep='\t', index=False)),
])
def test_load_dataframe_safe_reads_each_format(tmp_path, frame, name, writer):
    path = tmp_path / name
    writer(frame, path)
    loaded = ut.load_dataframe_safe(str(path))
    pd.testing.assert_frame_equal(loaded, frame)


@pytest.mark.parametrize("name", ["d.xls", "d.xlsx"])
def test_load_dataframe_safe_dispatches_excel(tmp_path, frame, name, monkeypatch):
    seen = []

    def fake_read_excel(path):
        seen.append(path)
        return frame

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)
    path = tmp_path / name
    path.write_text("placeholder", encoding="utf-8")
    pd.testing.assert_frame_equal(ut.load_dataframe_safe(str(path)), frame)
    assert seen == [str(path)]


def test_load_dataframe_safe_dispatches_parquet_and_hdf(tmp_path, frame, monkeypatch):
    for name, reader, call in [("d.parquet", "read_parquet", frame.to_parquet),
                               ("d.h5", "read_hdf", None)]:
        seen = []

        def fake(path, *_args, **_kwargs):
            seen.append(path)
            return frame

        monkeypatch.setattr(pd, reader, fake)
        path = tmp_path / name
        path.write_text("placeholder", encoding="utf-8")
        pd.testing.assert_frame_equal(ut.load_dataframe_safe(str(path)), frame)
        assert seen == [str(path)]


def test_load_dataframe_safe_unknown_extension_falls_back_to_sniffed_csv(tmp_path, frame):
    path = tmp_path / "matrix.tbl"
    frame.to_csv(path, sep=';', index=False)
    pd.testing.assert_frame_equal(ut.load_dataframe_safe(str(path)), frame)


def test_load_dataframe_safe_header_only_file_is_empty(tmp_path):
    (tmp_path / "only_header.csv").write_text("a,b\n", encoding="utf-8")
    assert ut.load_dataframe_safe(str(tmp_path / "only_header.csv")) is None


def test_load_dataframe_safe_missing_file_is_none(caplog):
    with caplog.at_level(logging.INFO, logger='biosuite.core.utils'):
        assert ut.load_dataframe_safe("/no/such/file.csv") is None
    assert [r.getMessage() for r in caplog.records] == ["File not found: /no/such/file.csv"]


def test_load_dataframe_safe_unparsable_file_is_none(tmp_path, caplog):
    path = tmp_path / "bad.json"
    path.write_text("this is not json at all\n", encoding="utf-8")
    with caplog.at_level(logging.ERROR, logger='biosuite.core.utils'):
        assert ut.load_dataframe_safe(str(path)) is None
    assert [r.levelno for r in caplog.records] == [logging.ERROR]
    assert caplog.records[0].getMessage().startswith("Error reading file: ")


def test_a_row_with_more_fields_than_its_header_is_not_an_error(tmp_path):
    """pandas reads a ragged row as an extra column rather than refusing the file,
    so the loader hands back the frame instead of inventing a failure."""
    path = tmp_path / "wide.csv"
    path.write_text("a,b\n1,2,3\n", encoding="utf-8")

    frame = ut.load_dataframe_safe(str(path))

    assert frame is not None and list(frame.columns) == ["a", "b"]


def test_an_empty_table_is_reported_as_no_data(tmp_path, caplog):
    path = tmp_path / "empty.csv"
    path.write_text("a,b\n", encoding="utf-8")
    with caplog.at_level(logging.INFO, logger='biosuite.core.utils'):
        assert ut.load_dataframe_safe(str(path)) is None
    assert [r.getMessage() for r in caplog.records] == ["File is empty."]


# ── load_dataframe_safe_interactive ────────────────────────────────────────

def test_interactive_loader_offers_the_summary(tmp_path, frame, monkeypatch, caplog):
    path = tmp_path / "d.csv"
    frame.to_csv(path, index=False)
    prompts = ScriptedInput(['Y'])
    monkeypatch.setattr('builtins.input', prompts)
    with caplog.at_level(logging.INFO, logger='biosuite.core.utils'):
        loaded = ut.load_dataframe_safe_interactive(str(path))

    pd.testing.assert_frame_equal(loaded, frame)
    assert prompts.prompts == ["Show data summary (describe)? (y/n): "]
    assert len(caplog.records) == 1
    summary = caplog.records[0].getMessage()
    for token in ("expr", "count", "mean", "2.500000"):
        assert token in summary, summary
    assert "gene" not in summary, "describe() summarises the numeric column only"


def test_interactive_loader_declined_summary_is_not_logged(tmp_path, frame, monkeypatch,
                                                           caplog):
    path = tmp_path / "d.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setattr('builtins.input', ScriptedInput(['n']))
    with caplog.at_level(logging.INFO, logger='biosuite.core.utils'):
        assert ut.load_dataframe_safe_interactive(str(path)) is not None
    assert [r.getMessage() for r in caplog.records] == []


def test_interactive_loader_is_quiet_in_quiet_mode(tmp_path, frame, monkeypatch):
    path = tmp_path / "d.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setitem(ut.config, 'quiet', True)
    monkeypatch.setattr('builtins.input', _never_called)
    pd.testing.assert_frame_equal(ut.load_dataframe_safe_interactive(str(path)), frame)


def test_interactive_loader_on_a_missing_file_does_not_prompt(monkeypatch):
    monkeypatch.setattr('builtins.input', _never_called)
    assert ut.load_dataframe_safe_interactive("/no/such/file.csv") is None


def test_interactive_loader_survives_a_closed_stdin(tmp_path, frame, monkeypatch):
    path = tmp_path / "d.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setattr('builtins.input', _eof_input)
    pd.testing.assert_frame_equal(ut.load_dataframe_safe_interactive(str(path)), frame)


def _eof_input(_prompt=''):
    raise EOFError


# ── maybe_downsample ────────────────────────────────────────────────────────

def test_downsample_rejects_mismatched_lengths():
    with pytest.raises(ValueError,
                       match=r"x and y must be the same length, got 3 and 2"):
        ut.maybe_downsample([1, 2, 3], [1, 2])


@pytest.mark.parametrize("bad", [1, 0, -3])
def test_downsample_rejects_tiny_max_points(bad):
    with pytest.raises(ValueError,
                       match=r"max_points must be at least 2, got %d" % bad):
        ut.maybe_downsample([1, 2, 3], [4, 5, 6], max_points=bad)


def test_downsample_keeps_lists_as_lists_and_tuples_as_tuples():
    x = list(range(11))
    y = tuple(v * 2 for v in range(11))
    xs, ys = ut.maybe_downsample(x, y, max_points=3)
    assert isinstance(xs, list) and isinstance(ys, tuple)
    assert list(xs) == [0, 5, 10]
    assert list(ys) == [0, 10, 20]


def test_downsample_of_a_series_returns_arrays_of_the_same_values():
    x = pd.Series(np.arange(11))
    y = pd.Series(np.arange(11) * 3)
    xs, ys = ut.maybe_downsample(x, y, max_points=3)
    assert list(xs) == [0, 5, 10]
    assert list(ys) == [0, 15, 30]


def test_downsample_returns_the_input_when_it_is_already_short():
    x, y = [1, 2, 3], [4, 5, 6]
    assert ut.maybe_downsample(x, y, max_points=10) == (x, y)


def test_downsample_uses_the_configured_threshold(monkeypatch):
    monkeypatch.setitem(ut.config, 'downsample_threshold', 3)
    x, y = np.arange(10), np.arange(10)
    xs, ys = ut.maybe_downsample(x, y)
    assert len(xs) <= 3 and xs[0] == 0 and xs[-1] == 9


# ── ask_save_plot ───────────────────────────────────────────────────────────

def _new_figure():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    assert plt.get_fignums() == [fig.number]
    return fig


def test_ask_save_plot_saves_under_the_typed_name_and_closes(tmp_path, monkeypatch,
                                                             interactive):
    prompts = ScriptedInput(['y', 'volcano'])
    monkeypatch.setattr('builtins.input', prompts)
    monkeypatch.chdir(tmp_path)
    _new_figure()

    ut.ask_save_plot("default_name", "png", 100)

    assert plt.get_fignums() == [], "the figure leaked"
    assert (tmp_path / "volcano.png").stat().st_size > 0
    assert list(tmp_path.glob("default_name.*")) == []


def test_ask_save_plot_blank_name_falls_back_to_the_default(tmp_path, monkeypatch,
                                                            interactive):
    monkeypatch.setattr('builtins.input', ScriptedInput(['y', '   ']))
    monkeypatch.chdir(tmp_path)
    _new_figure()

    ut.ask_save_plot("fallback", "png", 100)

    assert plt.get_fignums() == []
    assert (tmp_path / "fallback.png").stat().st_size > 0


def test_ask_save_plot_declined_writes_nothing_and_closes(tmp_path, monkeypatch,
                                                          interactive):
    prompts = ScriptedInput(['n'])
    monkeypatch.setattr('builtins.input', prompts)
    monkeypatch.chdir(tmp_path)
    _new_figure()

    ut.ask_save_plot("plot", "png", 100)

    assert plt.get_fignums() == []
    assert list(tmp_path.glob("plot.*")) == []
    assert prompts.prompts == ["Save this plot? (y/n): "]


def test_ask_save_plot_closed_stdin_is_a_decline(tmp_path, monkeypatch, interactive):
    monkeypatch.setattr('builtins.input', _eof_input)
    monkeypatch.chdir(tmp_path)
    _new_figure()

    ut.ask_save_plot("plot", "png", 100)

    assert plt.get_fignums() == []
    assert list(tmp_path.glob("*")) == []


def test_ask_save_plot_releases_the_figure_when_the_disk_refuses(tmp_path, monkeypatch,
                                                                 interactive, caplog):
    def refusing_savefig(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(plt, "savefig", refusing_savefig)
    monkeypatch.setattr('builtins.input', ScriptedInput(['y', 'plot']))
    monkeypatch.chdir(tmp_path)
    _new_figure()

    with caplog.at_level(logging.ERROR, logger='biosuite.core.utils'):
        ut.ask_save_plot("plot", "png", 100)

    assert plt.get_fignums() == []
    # The old text was "Save failed." with no file and no reason; the user could not
    # tell which path the plot did not reach, or why.
    assert [r.getMessage() for r in caplog.records] == [
        "Could not write plot.png: OSError: disk full"]


def test_a_filename_that_already_carries_the_extension_is_not_doubled(tmp_path,
                                                                     monkeypatch,
                                                                     interactive,
                                                                     caplog):
    """Measured on the old code: typing `volume.png` at the prompt wrote
    `volume.png.png`, because the extension was always appended."""
    written = []
    monkeypatch.setattr(plt, "savefig",
                        lambda target, **_kw: written.append(target))
    monkeypatch.setattr('builtins.input', ScriptedInput(['y', 'volume.png']))
    monkeypatch.chdir(tmp_path)
    _new_figure()

    with caplog.at_level(logging.INFO, logger='biosuite.core.utils'):
        ut.ask_save_plot("plot", "png", 100)
    assert written == ["volume.png"]
    assert "Saved as volume.png" in caplog.text


def test_ask_save_plot_adds_a_page_to_the_pdf_and_closes(monkeypatch, interactive):
    recorded = []

    class FakePdf:
        def savefig(self, *args, **kwargs):
            figure = kwargs.get('figure') or (args[0] if args else plt.gcf())
            recorded.append(figure)

    fig = _new_figure()
    ut.ask_save_plot("panel", "png", 100, pdf=FakePdf())

    assert plt.get_fignums() == []
    assert recorded == [fig], "the page added is the figure that was just drawn"


def test_ask_save_plot_survives_a_pdf_that_refuses_the_page(monkeypatch, interactive, caplog):
    class BrokenPdf:
        def savefig(self, *args, **kwargs):
            raise OSError("the disk is full")

    _new_figure()
    with caplog.at_level(logging.WARNING, logger='biosuite.core.utils'):
        assert ut.ask_save_plot("panel", "png", 100, pdf=BrokenPdf()) is None

    assert plt.get_fignums() == [], "a refused page still closes the figure"
    assert "Could not add panel to the PDF report" in caplog.records[-1].getMessage()


def test_ask_save_plot_records_the_panel_in_the_story(monkeypatch, interactive):
    recorded = []

    class FakePdf:
        def savefig(self, *args, **kwargs):
            recorded.append(1)

    story = []
    _new_figure()
    ut.ask_save_plot("panel", "png", 100, pdf=FakePdf(), story=story)

    assert story == ["![panel](panel.png)"]
    assert plt.get_fignums() == []


def test_ask_save_plot_keeps_going_when_the_pdf_is_closed(monkeypatch, interactive,
                                                           caplog):
    class ClosedPdf:
        def savefig(self, *_args, **_kwargs):
            raise ValueError("the PdfPages context has been closed")

    prompts = ScriptedInput(['unexpected'])
    monkeypatch.setattr('builtins.input', prompts)
    _new_figure()

    with caplog.at_level(logging.WARNING, logger='biosuite.core.utils'):
        ut.ask_save_plot("panel", "png", 100, pdf=ClosedPdf())

    assert plt.get_fignums() == []
    assert prompts.prompts == [], "the PDF path must not prompt"
    assert [r.getMessage() for r in caplog.records] == [
        "Could not add panel to the PDF report: the PdfPages context has been closed"]
