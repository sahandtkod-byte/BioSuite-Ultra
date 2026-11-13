"""biosuite.core.utils: config/session persistence, API keys, theme, window helper.

Everything here writes to the user's configuration directory, so every test
redirects ``get_config_file`` / ``get_session_file`` into ``tmp_path`` first,
snapshots the module-global ``config`` / ``session`` dicts (they are mutated in
place by the product code, so they are restored in place too), and asserts on
the JSON that actually landed on disk.  The fixture's teardown proves nothing
was written anywhere else.
"""
import copy
import json
import logging
import os
import types

import matplotlib.pyplot as plt
import pytest

from biosuite.core import utils as ut

REPO_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(ut.__file__))))
#: Where configuration used to be written (bug BSU-004).
LEGACY_NAMES = ("biosuite_config.json", "biosuite_session.json")
#: Where configuration used to be written (bug BSU-004).
LEGACY_NAMES = ("biosuite_config.json", "biosuite_session.json")


def _listing(directory):
    if not os.path.isdir(directory):
        return set()
    return {name for name in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, name))}


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    """Redirect config/session into tmp_path and restore all module state."""
    home = tmp_path / "state"
    home.mkdir()
    cfg_file = home / "biosuite_config.json"
    sess_file = home / "biosuite_session.json"

    # The location the *unredirected* code would use, kept as a tripwire.
    env_dir = os.environ.get("BIOSUITE_CONFIG_DIR", "")
    untouchable = {
        "env_config_dir": _listing(env_dir) if env_dir else set(),
        "repo_root": {n for n in os.listdir(REPO_ROOT) if n in LEGACY_NAMES},
    }

    monkeypatch.setattr(ut, "get_config_file", lambda: str(cfg_file))
    monkeypatch.setattr(ut, "get_session_file", lambda: str(sess_file))
    monkeypatch.setattr(ut, "user_config_dir", lambda: str(home))
    monkeypatch.setattr(ut, "LEGACY_CONFIG_FILE", str(home / "legacy_config.json"))

    cfg_snapshot = copy.deepcopy(ut.config)
    session_snapshot = copy.deepcopy(ut.session)
    counter_snapshot = ut._interaction_counter

    state = types.SimpleNamespace(home=home, cfg=cfg_file, session=sess_file,
                                  env_dir=env_dir, tmp=tmp_path)
    yield state

    ut._interaction_counter = counter_snapshot
    ut.config.clear()
    ut.config.update(cfg_snapshot)
    ut.session.clear()
    ut.session.update(session_snapshot)

    assert _listing(env_dir) == untouchable["env_config_dir"], \
        "a test wrote into the unredirected config directory"
    assert {n for n in os.listdir(REPO_ROOT) if n in LEGACY_NAMES} == \
        untouchable["repo_root"], "a test wrote config into the source tree"


@pytest.mark.parametrize("stored, kind", [
    pytest.param('["theme", "dpi"]', "list", id="list"),
    pytest.param('"dark-green"', "str", id="bare-string"),
    pytest.param('null', "NoneType", id="null"),
    pytest.param('42', "int", id="number"),
])
def test_a_config_that_is_not_an_object_is_reported_and_the_defaults_survive(
        iso, stored, kind, caplog):
    """`config = load_config()` runs while the package is being imported, so a
    merged non-mapping raised `TypeError: 'list' object is not a mapping` and took
    every entry point down with it — a hand-edited or half-written file bricked
    BioSuite at startup."""
    iso.cfg.write_text(stored, encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        loaded = ut.load_config()
    assert loaded == ut.DEFAULT_CONFIG
    assert loaded is not ut.DEFAULT_CONFIG, "the defaults must not be handed out live"
    assert [r.getMessage() for r in caplog.records] == [
        f"{iso.cfg} holds a JSON {kind}, not an object; ignoring it and using the "
        f"defaults."]


def test_a_corrupt_config_file_is_named_instead_of_vanishing(iso, caplog):
    """The old code skipped a broken file in silence; the user then wondered which
    of their settings had been forgotten."""
    iso.cfg.write_text('{"theme": "dark-green"', encoding="utf-8")   # truncated write
    with caplog.at_level(logging.WARNING):
        assert ut.load_config() == ut.DEFAULT_CONFIG
    message = caplog.records[-1].getMessage()
    assert message.startswith(f"Could not read {iso.cfg} (JSONDecodeError: ")
    assert "trying the next location." in message


def test_a_legacy_config_is_still_read_when_the_user_dir_has_nothing(iso):
    """The one-time migration path, behind the same object check."""
    (iso.home / "legacy_config.json").write_text('{"default_dpi": 96}', encoding="utf-8")
    assert ut.load_config()['default_dpi'] == 96

    (iso.home / "legacy_config.json").write_text('[1, 2]', encoding="utf-8")
    assert ut.load_config() == ut.DEFAULT_CONFIG


@pytest.fixture()
def restore_rcparams():
    with plt.rc_context():
        yield


class _RecordingInput:
    """Answers prompts and records them; any unexpected prompt is a failure."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt=""):
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError("unexpected prompt: %r" % (prompt,))
        return self.answers.pop(0)

    @property
    def called(self):
        return bool(self.prompts)


def _never_called(_prompt=""):
    raise AssertionError("the user should not have been prompted")


# ── user_config_dir / the two accessors ─────────────────────────────────────

def _os_shim(name, environ):
    return types.SimpleNamespace(name=name, environ=environ, path=os.path)


def test_config_and_session_paths_join_into_the_config_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("BIOSUITE_CONFIG_DIR", str(tmp_path))
    assert ut.get_config_file() == os.path.join(str(tmp_path), "biosuite_config.json")
    assert ut.get_session_file() == os.path.join(str(tmp_path), "biosuite_session.json")


def test_the_config_dir_override_is_expanded_and_absolute(monkeypatch, tmp_path):
    target = tmp_path / "nested" / ".." / "expanded"
    monkeypatch.setenv("BIOSUITE_CONFIG_DIR", str(target))
    assert ut.user_config_dir() == str(tmp_path / "expanded")


def test_override_env_var_wins_and_is_absolute(monkeypatch):
    monkeypatch.setattr(ut, "os", _os_shim("nt", {"BIOSUITE_CONFIG_DIR": "relative/dir"}))
    assert ut.user_config_dir() == os.path.abspath("relative/dir")


def test_windows_uses_appdata(monkeypatch, tmp_path):
    monkeypatch.setattr(ut, "os", _os_shim("nt", {"APPDATA": str(tmp_path)}))
    assert ut.user_config_dir() == os.path.join(str(tmp_path), "BioSuite")


def test_windows_without_appdata_uses_the_home(monkeypatch):
    monkeypatch.setattr(ut, "os", _os_shim("nt", {}))
    assert ut.user_config_dir() == os.path.join(os.path.expanduser("~"), "BioSuite")


def test_posix_uses_xdg_config_home(monkeypatch, tmp_path):
    monkeypatch.setattr(ut, "os", _os_shim("posix", {"XDG_CONFIG_HOME": str(tmp_path)}))
    assert ut.user_config_dir() == os.path.join(str(tmp_path), "biosuite")


def test_posix_falls_back_to_dot_config(monkeypatch):
    monkeypatch.setattr(ut, "os", _os_shim("posix", {}))
    assert ut.user_config_dir() == os.path.join(
        os.path.expanduser("~"), ".config", "biosuite")


def test_get_app_dir_is_the_repository_root():
    assert ut.get_app_dir() == REPO_ROOT


def test_get_app_dir_uses_the_executable_when_frozen(monkeypatch, tmp_path):
    frozen = types.SimpleNamespace(frozen=True,
                                   executable=str(tmp_path / "dist" / "BioSuite.exe"))
    monkeypatch.setattr(ut, "sys", frozen)
    assert ut.get_app_dir() == str(tmp_path / "dist")


# ── load_config ─────────────────────────────────────────────────────────────

def test_load_config_merges_the_file_over_the_defaults(iso):
    iso.cfg.write_text(json.dumps({"theme": "solar", "default_dpi": 96}),
                       encoding="utf-8")
    loaded = ut.load_config()
    expected = {**ut.DEFAULT_CONFIG, "theme": "solar", "default_dpi": 96}
    assert loaded == expected


def test_load_config_falls_back_to_the_legacy_location(iso, monkeypatch):
    legacy = iso.home / "legacy_config.json"
    legacy.write_text(json.dumps({"quiet": True}), encoding="utf-8")
    monkeypatch.setattr(ut, "LEGACY_CONFIG_FILE", str(legacy))
    assert ut.load_config()["quiet"] is True
    assert ut.load_config()["theme"] == ut.DEFAULT_CONFIG["theme"]


def test_a_corrupt_config_file_is_not_fatal(iso):
    iso.cfg.write_text("{not json at all", encoding="utf-8")
    assert ut.load_config() == ut.DEFAULT_CONFIG


def test_the_fallback_config_is_a_copy(iso):
    loaded = ut.load_config()
    assert loaded is not ut.DEFAULT_CONFIG
    loaded["theme"] = "mutated-by-a-caller"
    assert ut.DEFAULT_CONFIG["theme"] == "dark-green"


# ── save_config ─────────────────────────────────────────────────────────────

def test_save_config_writes_the_exact_json(iso):
    payload = {**ut.DEFAULT_CONFIG, "marker": "abc123"}
    ut.save_config(payload)
    assert json.loads(iso.cfg.read_text(encoding="utf-8")) == payload


def test_save_config_creates_a_missing_directory(iso, monkeypatch):
    nested = iso.home / "deep" / "deeper"
    monkeypatch.setattr(ut, "get_config_file",
                        lambda: str(nested / "biosuite_config.json"))
    ut.save_config({"theme": "nested"})
    assert json.loads((nested / "biosuite_config.json").read_text(
        encoding="utf-8"))["theme"] == "nested"


def test_save_config_leaves_no_temp_file_when_encoding_fails(iso):
    """The partial temp file must be cleaned up and the error reported."""
    with pytest.raises(TypeError):
        ut.save_config({"obj": object()})
    assert not iso.cfg.exists()
    assert [p.name for p in iso.home.iterdir()] == []


def test_a_failed_temp_cleanup_does_not_mask_the_real_error(iso, monkeypatch):
    def refusing_unlink(_path):
        raise OSError("unlink refused")

    monkeypatch.setattr(ut.os, "unlink", refusing_unlink)
    with pytest.raises(TypeError):
        ut.save_config({"obj": object()})


def test_a_failed_chmod_is_reported_but_keeps_the_file(iso, monkeypatch, caplog):
    def refusing_chmod(_path, _mode):
        raise OSError("chmod not supported")

    monkeypatch.setattr(ut.os, "chmod", refusing_chmod)
    with caplog.at_level("WARNING", logger="biosuite.core.utils"):
        ut.save_config({"theme": "still-saved"})

    assert json.loads(iso.cfg.read_text(encoding="utf-8"))["theme"] == "still-saved"
    messages = [r.getMessage() for r in caplog.records]
    assert any("could not restrict its permissions" in m and "chmod not supported" in m
               for m in messages), messages


def test_save_config_raises_when_the_directory_is_unwritable(iso, monkeypatch):
    blocked = iso.home / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(ut, "get_config_file",
                        lambda: str(blocked / "sub" / "biosuite_config.json"))
    with pytest.raises(OSError):
        ut.save_config({"theme": "dark"})


# ── session ─────────────────────────────────────────────────────────────────

def test_session_round_trip_writes_the_exact_json(iso):
    payload = {"recent": ["a.fasta", "b.csv"], "tab": 3}
    ut.save_session(payload)
    assert json.loads(iso.session.read_text(encoding="utf-8")) == payload
    assert ut.load_session() == payload


def test_load_session_missing_file_is_empty(iso):
    assert ut.load_session() == {}


def test_load_session_corrupt_file_is_empty(iso):
    iso.session.write_text("[busted", encoding="utf-8")
    assert ut.load_session() == {}


def test_save_session_reports_and_swallows_a_write_error(iso, monkeypatch, caplog):
    blocked = iso.home / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(ut, "get_session_file",
                        lambda: str(blocked / "biosuite_session.json"))
    path = str(blocked / "biosuite_session.json")
    try:
        open(path, "w", encoding="utf-8").close()
    except OSError as exc:
        expected = "Could not save the session to %s: %s" % (path, exc)
    else:
        raise AssertionError("the blocked path unexpectedly opened: " + path)

    with caplog.at_level(logging.WARNING, logger="biosuite.core.utils"):
        result = ut.save_session({"tab": 1})

    assert result is None and not (blocked / "biosuite_session.json").exists()
    records = [r for r in caplog.records if r.name == "biosuite.core.utils"]
    assert [(r.levelno, r.getMessage()) for r in records] == [(logging.WARNING, expected)]


def test_autosave_session_uses_the_module_session_dict(iso):
    ut.session.clear()
    ut.session["last_tab"] = "plots"
    ut.autosave_session()
    assert json.loads(iso.session.read_text(encoding="utf-8")) == {"last_tab": "plots"}


def test_autosave_session_accepts_an_explicit_payload(iso):
    ut.session["untouched"] = 1
    ut.autosave_session({"only": "this"})
    assert json.loads(iso.session.read_text(encoding="utf-8")) == {"only": "this"}


# ── API keys ────────────────────────────────────────────────────────────────

def test_set_api_key_recreates_a_missing_api_keys_section(iso):
    iso.cfg.write_text(json.dumps({"theme": "dark"}), encoding="utf-8")
    ut.config.clear()
    ut.config.update(json.loads(iso.cfg.read_text(encoding="utf-8")))
    assert "api_keys" not in ut.config

    ut.set_api_key("ncbi_api_key", "SECRET-1")

    assert ut.config["api_keys"]["ncbi_api_key"] == "SECRET-1"
    on_disk = json.loads(iso.cfg.read_text(encoding="utf-8"))
    assert on_disk == {"theme": "dark", "api_keys": {"ncbi_api_key": "SECRET-1"}}


def test_get_api_key_returns_empty_for_an_unknown_service(iso):
    ut.config["api_keys"] = {"kegg_email": "me@example.org"}
    assert ut.get_api_key("kegg_email") == "me@example.org"
    assert ut.get_api_key("uniprot_email") == ""


@pytest.fixture()
def no_existing_key(iso):
    ut.config["api_keys"] = {}
    ut.config["quiet"] = False
    return iso


def test_prompt_api_key_returns_the_stored_key_without_asking(no_existing_key, monkeypatch):
    ut.config["api_keys"]["ncbi_api_key"] = "SAVED"
    monkeypatch.setattr("builtins.input", _never_called)
    assert ut.prompt_api_key("ncbi_api_key") == "SAVED"


def test_prompt_api_key_is_silent_when_quiet(no_existing_key, monkeypatch):
    ut.config["quiet"] = True
    monkeypatch.setattr("builtins.input", _never_called)
    assert ut.prompt_api_key("ncbi_api_key") == ""
    assert not no_existing_key.cfg.exists()


def test_prompt_api_key_skips_a_non_interactive_session(no_existing_key, monkeypatch,
                                                        caplog):
    monkeypatch.setattr(ut.sys, "stdin", types.SimpleNamespace(isatty=lambda: False))
    prompts = _RecordingInput([])
    monkeypatch.setattr("builtins.input", prompts)
    assert ut.prompt_api_key("ncbi_api_key") == ""
    assert prompts.prompts == []
    assert [r.getMessage() for r in caplog.records] == [
        "\n  API key for ncbi_api_key not set (non-interactive session)."]


def test_prompt_api_key_stores_a_typed_key(no_existing_key, monkeypatch, caplog):
    monkeypatch.setattr(ut.sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
    prompts = _RecordingInput(["   TYPED-KEY   "])
    monkeypatch.setattr("builtins.input", prompts)

    with caplog.at_level(logging.INFO, logger="biosuite.core.utils"):
        got = ut.prompt_api_key("uniprot_email", "Get one from uniprot.org")

    assert got == "TYPED-KEY"
    assert prompts.prompts == ["  Enter API key (or blank to skip): "]
    assert [r.getMessage() for r in caplog.records] == [
        "\n  API key required for: uniprot_email",
        "  Get one from uniprot.org",
        "  API key saved for uniprot_email."]
    assert ut.config["api_keys"]["uniprot_email"] == "TYPED-KEY"
    assert json.loads(no_existing_key.cfg.read_text(encoding="utf-8"))["api_keys"] == \
        {"uniprot_email": "TYPED-KEY"}


def test_prompt_api_key_blank_answer_saves_nothing(no_existing_key, monkeypatch, caplog):
    monkeypatch.setattr(ut.sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
    prompts = _RecordingInput(["   "])
    monkeypatch.setattr("builtins.input", prompts)

    with caplog.at_level(logging.INFO, logger="biosuite.core.utils"):
        got = ut.prompt_api_key("kegg_email")

    assert got == ""
    assert prompts.prompts == ["  Enter API key (or blank to skip): "]
    assert [r.getMessage() for r in caplog.records] == [
        "\n  API key required for: kegg_email"]
    assert ut.config["api_keys"] == {}
    assert not no_existing_key.cfg.exists()


# ── set_theme ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("choice", ["dark", "dark-green", "dark-purple"])
def test_dark_theme_applies_the_dark_palette(restore_rcparams, choice):
    ut.set_theme(choice)
    assert plt.rcParams["figure.facecolor"] == "#121212"
    assert plt.rcParams["axes.facecolor"] == "#1a1a1a"
    assert plt.rcParams["text.color"] == "#00ff99"
    assert plt.rcParams["grid.color"] == "#333333"
    assert plt.rcParams["grid.alpha"] == 0.4
    assert plt.rcParams["axes.titlesize"] == 13


def test_light_theme_applies_the_teal_palette(restore_rcparams):
    ut.set_theme("light")
    assert plt.rcParams["figure.facecolor"] == "#e6f7fa"
    assert plt.rcParams["axes.facecolor"] == "#f0fcfd"
    assert plt.rcParams["text.color"] == "#006064"
    assert plt.rcParams["grid.color"] == "#b2ebf2"
    assert plt.rcParams["grid.alpha"] == 0.3
    assert plt.rcParams["legend.fontsize"] == 10


@pytest.mark.parametrize("choice,unavailable,attempted", [
    ("light", frozenset(), ["seaborn-v0_8-whitegrid"]),
    ("light", frozenset({"seaborn-v0_8-whitegrid"}),
     ["seaborn-v0_8-whitegrid", "seaborn-whitegrid"]),
    ("light", frozenset({"seaborn-v0_8-whitegrid", "seaborn-whitegrid"}),
     ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "default"]),
    ("dark", frozenset(), ["seaborn-v0_8-darkgrid"]),
    ("dark", frozenset({"seaborn-v0_8-darkgrid"}),
     ["seaborn-v0_8-darkgrid", "seaborn-darkgrid"]),
    ("dark", frozenset({"seaborn-v0_8-darkgrid", "seaborn-darkgrid"}),
     ["seaborn-v0_8-darkgrid", "seaborn-darkgrid", "dark_background"]),
])
def test_theme_style_fallback_chain(restore_rcparams, monkeypatch, choice,
                                    unavailable, attempted):
    """Each matplotlib style generation is tried before giving up."""
    seen = []

    def fake_use(name):
        seen.append(name)
        if name in unavailable:
            raise OSError("%s is not a valid style library" % name)

    monkeypatch.setattr(plt.style, "use", fake_use)
    ut.set_theme(choice)
    assert seen == attempted
    facecolor = "#121212" if choice == "dark" else "#e6f7fa"
    assert plt.rcParams["figure.facecolor"] == facecolor


# ── center_window ───────────────────────────────────────────────────────────

class _FakeWindow:
    def __init__(self, width, height):
        self.screen = (width, height)
        self.update_calls = 0
        self.geometry_string = None

    def update_idletasks(self):
        self.update_calls += 1

    def winfo_screenwidth(self):
        return self.screen[0]

    def winfo_screenheight(self):
        return self.screen[1]

    def geometry(self, spec):
        self.geometry_string = spec
        return spec


@pytest.mark.parametrize("screen,size,expected", [
    ((1920, 1080), (800, 600), "800x600+560+240"),
    ((1000, 700), (640, 480), "640x480+180+110"),
    ((1001, 701), (333, 111), "333x111+334+295"),
])
def test_center_window_positions_by_screen_centre(screen, size, expected):
    win = _FakeWindow(*screen)
    ut.center_window(win, size[0], size[1])
    assert win.update_calls == 1
    assert win.geometry_string == expected


def test_center_window_handles_an_odd_small_request():
    win = _FakeWindow(100, 100)
    ut.center_window(win, 150, 150)
    assert win.geometry_string == "150x150+-25+-25"
