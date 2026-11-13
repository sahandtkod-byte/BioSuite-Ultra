"""Regression tests for the provenance tracker (BSU-007).

Three independent defects made provenance unusable in the very situation it
exists for - recording what a long, concurrent analysis actually did:

* one ``sqlite3.Connection`` was shared across threads without a lock, so
  concurrent ``record()`` calls raised or lost rows;
* numpy scalars/arrays in the parameter dict crashed ``json.dumps``;
* ``export_html`` interpolated parameter text straight into the document.
"""
import concurrent.futures
import json

import numpy as np
import pandas as pd
import pytest

from biosuite.core.provenance import ProvenanceTracker, dumps_params


@pytest.fixture()
def tracker(tmp_path):
    tracked = ProvenanceTracker(db_path=str(tmp_path / "prov.db"))
    yield tracked
    tracked.close()


# ── thread safety ───────────────────────────────────────────────────────────

def test_concurrent_record_keeps_every_step_and_assigns_unique_ids(tracker):
    n = 200

    def record(i):
        return tracker.record(module="m", function=f"f{i}",
                              params={"i": i}, result_summary=str(i))

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        recorded = list(pool.map(record, range(n)))

    ids = [step.step_id for step in recorded]
    assert len(ids) == n
    assert len(set(ids)) == n, "step ids collided across threads"
    steps = tracker.get_steps()
    assert len(steps) == n
    assert {s.function for s in steps} == {f"f{i}" for i in range(n)}


# ── numpy / pandas parameters ───────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    (np.int64(3), 3),
    (np.float32(1.5), 1.5),
    (np.bool_(True), True),
    (np.array([1, 2, 3]), [1, 2, 3]),
    (np.array([[1.0, 2.0], [3.0, 4.0]]), [[1.0, 2.0], [3.0, 4.0]]),
    # Non-finite values have no JSON spelling, so they are recorded as text
    # rather than as the NaN token Python's json module would happily emit.
    (np.float64("nan"), "nan"),
    (np.float64("inf"), "inf"),
])
def test_numpy_parameters_are_serialisable(value, expected):
    text = dumps_params({"value": value})
    assert "NaN" not in text and "Infinity" not in text, text
    assert json.loads(text)["value"] == expected


def test_numpy_parameters_round_trip_through_the_database(tracker):
    tracker.record(module="m", function="f",
                   params={"threshold": np.float64(0.05),
                           "counts": np.array([1, 2, 3]),
                           "flag": np.bool_(False)})
    step = tracker.get_steps()[0]
    params = step.params if isinstance(step.params, dict) else json.loads(step.params)
    assert params["threshold"] == pytest.approx(0.05)
    assert list(params["counts"]) == [1, 2, 3]
    assert params["flag"] is False


def test_pandas_objects_do_not_break_recording(tracker):
    frame = pd.DataFrame({"a": [1, 2]})
    tracker.record(module="m", function="f", params={"df": frame})
    assert len(tracker.get_steps()) == 1


def test_unserialisable_objects_are_recorded_by_class_name():
    class Opaque:
        pass

    text = dumps_params({"obj": Opaque()})
    assert json.loads(text) == {"obj": "<Opaque>"}


# ── HTML export ─────────────────────────────────────────────────────────────

def test_html_export_escapes_untrusted_text(tracker, tmp_path):
    payload = '<script>alert("xss")</script>'
    tracker.record(module=payload, function="f",
                   params={"note": payload}, result_summary=payload)
    out = tmp_path / "prov.html"
    tracker.export_html(str(out))
    html = out.read_text()
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


def test_json_export_is_valid_json_with_numpy_params(tracker, tmp_path):
    tracker.record(module="m", function="f", params={"x": np.arange(3)})
    out = tmp_path / "prov.json"
    tracker.export_json(str(out))
    data = json.loads(out.read_text())
    assert data
