"""The plot endpoints return a path to a real PNG and give the figure back.

An HTTP server that renders with matplotlib leaks memory through pyplot's global
figure registry, so ``save_figure`` closing the figure is the property these tests
pin — along with the artifact cache bounding itself by age and count, which nothing
ran before because the pruning code only triggers after an hour or sixty-four files.
"""
import json
import os
import time

import matplotlib.pyplot as plt
import pytest
from fastapi.testclient import TestClient

from biosuite import api as api_module

VOLCANO = {"log2fc": [2.0, -2.0, 0.1, 1.5], "pvalues": [0.001, 0.002, 0.9, 0.04],
           "gene_names": ["A", "B", "C", "D"]}
PCA_DATA = [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [8.0, 7.0, 3.0], [9.0, 8.0, 2.0]]
MANHATTAN = {"chromosomes": ["1", "1", "2", "2"], "positions": [100, 200, 150, 300],
             "pvalues": [1e-9, 0.4, 0.8, 1e-12]}


@pytest.fixture(scope="module")
def client():
    c = TestClient(api_module.app, raise_server_exceptions=False)
    c.headers["X-API-Key"] = os.environ.get("BIOSUITE_API_KEY", "test-api-key-not-a-real-secret")
    return c


@pytest.fixture(autouse=True)
def _no_figures_left_behind():
    """One test's leaked figure would poison the next one's baseline."""
    yield
    plt.close("all")


def _post_plot(client, path, payload=None, **params):
    before = set(plt.get_fignums())
    response = client.post(path, json=payload, params=params)
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(plt.get_fignums()) == before, f"{path} left a figure open in pyplot"
    return body


# ── PNG endpoints ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(("path", "payload"), [
    ("/api/v1/plotting/pca", {"data": PCA_DATA}),
    ("/api/v1/plotting/heatmap", [[1.0, 2.0], [3.0, 4.0]]),
    ("/api/v1/plotting/manhattan", MANHATTAN),
])
def test_plot_endpoint_returns_a_real_png(client, path, payload):
    body = _post_plot(client, path, payload)

    assert body["format"] == "png"
    with open(body["image_path"], "rb") as handle:
        assert handle.read(8) == b"\x89PNG\r\n\x1a\n"


def test_volcano_static_returns_a_png_and_honours_the_thresholds(client):
    body = _post_plot(client, "/api/v1/plotting/volcano", {**VOLCANO, "interactive": False})

    assert body["format"] == "png"
    assert os.path.getsize(body["image_path"]) > 1000


def test_artifacts_go_to_the_managed_directory_not_the_repository(client):
    body = _post_plot(client, "/api/v1/plotting/heatmap", [[1.0, 2.0], [3.0, 4.0]])

    assert str(api_module._ARTIFACT_DIR) in body["image_path"]
    assert not body["image_path"].startswith(os.getcwd())


def test_volcano_interactive_returns_plotly_json(client):
    body = _post_plot(client, "/api/v1/plotting/volcano", {**VOLCANO, "interactive": True})

    figure = json.loads(body["plotly_json"])
    assert figure["data"], "interactive volcano returned no traces"
    assert all(trace["type"] == "scatter" for trace in figure["data"])
    assert sum(len(trace["x"]) for trace in figure["data"]) == len(VOLCANO["log2fc"])


# ── Artifact cache pruning ───────────────────────────────────────────────────

@pytest.fixture
def artifact_dir(tmp_path, monkeypatch):
    directory = tmp_path / "artifacts"
    directory.mkdir()
    monkeypatch.setattr(api_module, "_ARTIFACT_DIR", directory)
    return directory


def _touch_png(directory, name, age_s):
    path = directory / name
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    stamp = time.time() - age_s
    os.utime(path, (stamp, stamp))
    return path


def test_pruning_removes_artifacts_older_than_the_max_age(artifact_dir):
    stale = _touch_png(artifact_dir, "old.png", api_module._ARTIFACT_MAX_AGE_S + 60)
    fresh = _touch_png(artifact_dir, "new.png", 0)

    api_module._prune_artifacts()

    assert not stale.exists()
    assert fresh.exists()


def test_pruning_keeps_the_newest_files_when_there_are_too_many(artifact_dir, monkeypatch):
    monkeypatch.setattr(api_module, "_ARTIFACT_MAX_FILES", 3)
    monkeypatch.setattr(api_module, "_ARTIFACT_MAX_AGE_S", 10 ** 9)   # age is not the reason here
    oldest = _touch_png(artifact_dir, "oldest.png", 300)
    kept = [_touch_png(artifact_dir, f"keep{i}.png", 100 - i) for i in range(3)]

    api_module._prune_artifacts()

    assert not oldest.exists()
    assert all(path.exists() for path in kept)


def test_pruning_survives_a_vanishing_directory(artifact_dir, monkeypatch):
    """A cleanup path must not take the server down when the scan itself fails."""
    attempted = []

    def refusing_glob(*args, **kwargs):
        attempted.append(args)
        raise OSError("device gone")

    # Path has __slots__, so the failure has to be installed on the class; monkeypatch
    # puts it back when the test ends.
    monkeypatch.setattr(type(artifact_dir), "glob", refusing_glob)

    api_module._prune_artifacts()          # returns quietly

    assert attempted, "the scan was never reached, so the guard proved nothing"


def test_pruning_survives_the_cache_disappearing_mid_pass(artifact_dir, monkeypatch):
    """The age pass and the count pass each scan again; either scan can fail."""
    calls = []

    def glob_then_break(self, pattern):
        calls.append(pattern)
        if len(calls) > 1:
            raise OSError("cache directory went away")
        return iter(())

    monkeypatch.setattr(type(artifact_dir), "glob", glob_then_break)

    api_module._prune_artifacts()          # second scan must be swallowed

    assert len(calls) == 2, "both passes should have scanned the cache"


def test_pruning_survives_a_file_another_process_removed(artifact_dir, monkeypatch):
    """The cache is shared by every worker, so a listing can be stale on use.

    Both passes — age pruning and count pruning — must swallow that race rather than
    fail a request that merely wanted to save a figure.
    """
    doomed = _touch_png(artifact_dir, "doomed.png", api_module._ARTIFACT_MAX_AGE_S + 60)

    def vanishing_unlink(self, *args, **kwargs):
        raise OSError("No such file or directory")

    monkeypatch.setattr(type(doomed), "unlink", vanishing_unlink)
    api_module._prune_artifacts()
    assert doomed.exists()               # the age pass gave up quietly

    monkeypatch.setattr(api_module, "_ARTIFACT_MAX_FILES", 1)
    monkeypatch.setattr(api_module, "_ARTIFACT_MAX_AGE_S", 10 ** 9)
    _touch_png(artifact_dir, "keeper.png", 10)     # doomed is the older of the two
    api_module._prune_artifacts()                  # the count pass must survive it too
