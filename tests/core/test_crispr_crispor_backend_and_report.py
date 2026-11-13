"""CRISPR guide design: the optional CRISPOR backend and the text report.

Pins what ``_crispor_api_search`` sends and returns (the request is always
captured — this file never touches the network), that ``tool='crispor'`` can
only warn and fall back to the built-in PAM finder, and the exact column layout
``format_crispr_report`` writes for the top ten guides.
"""
import importlib.util
import json
import pathlib
import sys
import urllib.error
import urllib.request
import warnings

import pytest

from biosuite.core import crispr
from biosuite.core.crispr import (CRISPRResult, GuideRNA, _crispor_api_search,
                                  check_crispr_tools, design_guides,
                                  format_crispr_report)
from biosuite.core.utils import PerformanceWarning

FWD_GUIDE = "ACGTACGTACGTACGTACGT"
TARGET = "TTTT" * 10 + FWD_GUIDE + "AGG" + "CCCC" * 10
API_ROWS = [
    "1    ACGTACGTACGTACGTACGT     AGG    +        1.000  50.0%",
    "2    GGGGGGGGGGGGGGGGGGGG     GGG    -        0.000 100.0%",
    "3    GGGGGGGGGGGGGGGGGGGG     GGG    -        0.000 100.0%",
    "4    GGGGGGGGGGGGGGGGGGGG     GGG    -        0.000 100.0%",
    "5    GGGGGGGGGGGGGGGGGGGG     GGG    -        0.000 100.0%",
    "6    GGGGGGGGGGGGGGGGGGGG     GGG    -        0.000 100.0%",
    "7    GGGGGGGGGGGGGGGGGGGG     GGG    -        0.000 100.0%",
    "8    GGGGGGGGGGGGGGGGGGGG     GGG    -        0.000 100.0%",
    "9    GGGGGGGGGGGGGGGGGGGG     GGG    -        0.000 100.0%",
    "10   GGGGGGGGGGGGGGGGGGGG     GGG    -        0.000 100.0%",
]


class _Response:
    """Stand-in for the object urllib.request.urlopen() returns."""

    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _module_without(blocked, dotted):
    """Re-execute crispr.py with *blocked* stdlib names made unimportable.

    ``importlib.reload`` cannot be used: the already-imported module keeps the
    attributes its first import defined, so HAS_URLLIB would stay True.
    """
    spec = importlib.util.spec_from_file_location(
        dotted, pathlib.Path(crispr.__file__))
    fresh = importlib.util.module_from_spec(spec)
    saved = {name: sys.modules.get(name) for name in blocked}
    try:
        for name in blocked:
            sys.modules[name] = None
        spec.loader.exec_module(fresh)
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return fresh


def test_check_crispr_tools_reports_the_import_time_urllib_flag():
    assert crispr.HAS_URLLIB is True
    assert check_crispr_tools() == {'crispor_api': crispr.HAS_URLLIB}


def test_crispor_api_search_posts_the_sequence_and_returns_parsed_json(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen['url'] = req.full_url
        seen['data'] = req.data
        seen['timeout'] = timeout
        return _Response(json.dumps({"guides": [{"seq": FWD_GUIDE}]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = _crispor_api_search(TARGET[:23])

    assert seen == {'url': 'http://crispor.tefor.net/api/',
                    'data': b'sequence=' + TARGET[:23].encode() + b'&genome=mm10',
                    'timeout': 30}
    assert result == {"guides": [{"seq": FWD_GUIDE}]}


@pytest.mark.parametrize("failure", [
    pytest.param(lambda req, timeout=None: (_ for _ in ()).throw(
        urllib.error.URLError("no route to host")), id="url-error"),
    pytest.param(lambda req, timeout=None: (_ for _ in ()).throw(
        OSError("socket closed")), id="os-error"),
    pytest.param(lambda req, timeout=None: _Response(b"<html>not json</html>"),
                 id="non-json-payload"),
])
def test_crispor_api_search_returns_none_for_every_transport_failure(monkeypatch, failure):
    monkeypatch.setattr(urllib.request, "urlopen", failure)
    assert _crispor_api_search("ACGTACGTACGTACGTACGTAGG") is None


def test_crispor_api_search_passes_an_empty_payload_through_untouched(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: _Response(b"{}"))
    assert _crispor_api_search("ACGTACGTACGTACGTACGTAGG") == {}


def test_an_empty_crispor_payload_designs_guides_without_any_warning(monkeypatch):
    """Measured: a falsy API answer skips the warning, but guides still come."""
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: _Response(b"{}"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        result = design_guides(TARGET, tool='crispor')
    assert result.engine == 'builtin'
    assert result.num_guides == 19


def test_tool_crispor_warns_and_still_designs_built_in_guides(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: _Response(b'{"guides": [1]}'))

    with pytest.warns(PerformanceWarning) as caught:
        result = design_guides(TARGET, tool='crispor')

    message = str(caught[0].message)
    assert message.startswith("CRISPOR API accepted the sequence but result")
    assert "https://crispor.tefor.net/" in message
    assert result.engine == 'builtin'
    assert result.num_guides == 19
    assert result.guides[0].sequence == FWD_GUIDE


def test_tool_crispor_falls_back_even_when_the_api_call_fails(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.URLError("dns failure")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    result = design_guides(TARGET, tool='crispor')

    assert result.engine == 'builtin'
    assert result.num_guides == 19
    assert result.message == "Found 19 guide RNAs for SpCas9"


def test_without_urllib_the_crispor_backend_reports_itself_unavailable():
    fresh = _module_without(["urllib.request"], "biosuite.core.crispr_without_urllib")

    assert fresh.HAS_URLLIB is False
    assert fresh.check_crispr_tools() == {'crispor_api': False}
    assert fresh._crispor_api_search("ACGTAGG") is None

    with pytest.warns(PerformanceWarning) as caught:
        result = fresh.design_guides(TARGET, tool='crispor')
    assert str(caught[0].message).startswith("CRISPOR API not available.")
    assert result.engine == 'builtin'
    assert result.num_guides == 19
    assert crispr.HAS_URLLIB is True, "the shipped module must be untouched"


def test_report_writes_a_fixed_width_table_of_the_ten_best_guides():
    result = design_guides(TARGET, max_guides=20)
    lines = format_crispr_report(result).splitlines()

    assert lines[:4] == ["=== CRISPR Guide RNA Design Report ===",
                         "Engine: builtin",
                         "Target length: 103 bp",
                         "Guides found: 19"]
    assert lines[4] == ""
    assert lines[5] == "#    Guide Sequence           PAM    Strand   Score    GC%"
    assert lines[6] == "-" * 65
    assert lines[7:17] == API_ROWS
    assert len(result.guides) == 19, "the table is truncated, the result is not"
    assert lines[17] == ""
    assert lines[18] == "Note: Found 19 guide RNAs for SpCas9"


def test_report_lists_one_row_per_guide_and_caps_at_ten():
    single = CRISPRResult(
        engine='builtin',
        guides=[GuideRNA(sequence=FWD_GUIDE, pam='AGG', position=40, strand='+',
                         score=1.0, gc_content=50.0, off_target_count=0,
                         on_target_score=1.0)],
        target_sequence=FWD_GUIDE, num_guides=1, message="Found 1 guide RNAs for SpCas9")
    lines = format_crispr_report(single).splitlines()

    assert lines[2] == "Target length: 20 bp"
    assert lines[3] == "Guides found: 1"
    assert lines[7] == "1    ACGTACGTACGTACGTACGT     AGG    +        1.000  50.0%"
    assert len([row for row in lines if row[:1].isdigit()]) == 1


def test_report_omits_the_note_and_table_for_a_result_without_message():
    text = format_crispr_report(CRISPRResult(engine='builtin'))
    assert text.splitlines() == ["=== CRISPR Guide RNA Design Report ===",
                                 "Engine: builtin",
                                 "Target length: 0 bp",
                                 "Guides found: 0"]
    assert "Note:" not in text
