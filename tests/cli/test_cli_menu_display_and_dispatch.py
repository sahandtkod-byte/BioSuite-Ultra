"""The menu's own rendering, the entry flags, and the safe-callable resolver.

Three parts of the CLI are not menu *options*, so the two sweeps in
``test_cli_menu_option_sessions*.py`` never reach them: the banner that lists the
options, the ``--gui`` / ``--api`` / ``--theme`` entry flags, and the resolver that
decides which callables the interactive pipeline builder may run.  That last one is a
security boundary — the builders used to ``eval()`` whatever the user typed — so its
refusals matter as much as its acceptances.
"""
import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from biosuite.cli import menu

ANSI = re.compile(r"\x1b\[[0-9;]*m")

#: The option keys the dispatcher answers, read from the source.
DISPATCHED = sorted(set(re.findall(
    r"(?:el)?if choice == '([a-z0-9]+)'",
    Path(menu.__file__).read_text(encoding="utf-8"))))


@pytest.fixture(scope="module")
def banner():
    """``print_menu()`` as a user sees it, with the colour codes stripped off."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        menu.print_menu()
    return ANSI.sub("", buffer.getvalue())


@pytest.fixture
def captured():
    """An in-test stdout sink, so the launchers can be exercised directly."""
    return io.StringIO()


# ── the banner ───────────────────────────────────────────────────────────────

def test_the_banner_shows_every_option_the_dispatcher_answers(banner):
    """An option missing from the banner is unreachable, however well it works."""
    missing = [key for key in DISPATCHED if not re.search(rf"\b{re.escape(key)}\b", banner)]

    assert not missing, f"options dispatched but never listed: {missing}"


def test_the_banner_names_its_sections(banner):
    for section in ("SEQUENCE ANALYSIS", "TRANSCRIPTOMICS", "MOLECULAR CLONING", "SYSTEM"):
        assert section in banner


def test_the_banner_leaves_no_unrendered_placeholders(banner):
    assert "{B}" not in banner and "{R}" not in banner, "an f-string prefix went missing"
    assert "\x1b[" not in banner            # stripped above; proves codes were the only noise


def test_the_exit_option_is_listed_with_the_system_block(banner):
    assert re.search(r"99\s+Change Theme\s+0\s+Exit", banner)


def test_an_item_without_a_description_still_aligns():
    """The banner has both shapes; the short one is a separate formatting path."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        menu._item("7", "Short")
        menu._item("8", "With description", "explains itself")
    plain = ANSI.sub("", buffer.getvalue())

    assert "Short" in plain and "With description" in plain
    assert "explains itself" in plain


def test_header_section_and_item_helpers_compose_a_block():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        menu._header()
        menu._section("DEMO SECTION", menu.C)
        menu._item("42", "Thing", "does a thing")
        menu._col2("7", "8", "Left", "Right")
        menu._end_section()
    plain = ANSI.sub("", buffer.getvalue())

    assert "DEMO SECTION" in plain and "42" in plain and "Thing" in plain
    assert "Left" in plain and "Right" in plain
    assert buffer.getvalue().count("\x1b[") > 4, "the helpers must still emit colour"


# ── entry flags ──────────────────────────────────────────────────────────────

def test_the_gui_flag_launches_the_gui_and_skips_the_menu(monkeypatch):
    calls = []
    monkeypatch.setattr(menu, "_launch_gui", lambda: calls.append("gui"))

    assert menu.main_cli(["--gui"]) == 0
    assert calls == ["gui"]


def test_the_api_flag_passes_the_port_through(monkeypatch):
    calls = []
    monkeypatch.setattr(menu, "_launch_api", lambda port=None: calls.append(port))

    assert menu.main_cli(["--api", "--port", "9123"]) == 0
    assert calls == [9123], "--port is parsed as an integer"


def test_an_unknown_theme_under_the_flags_reports_failure(monkeypatch):
    monkeypatch.setitem(menu.config, "theme", "dark-green")

    assert menu.main_cli(["--theme", "nonsense"]) == 2
    assert menu.config["theme"] == "dark-green"


def test_the_theme_flag_changes_the_theme(monkeypatch):
    seen = []
    monkeypatch.setattr(menu, "set_theme", lambda name: seen.append(name))
    monkeypatch.setitem(menu.config, "theme", "dark-green")

    assert menu.main_cli(["--theme", "light-blue"]) == 0
    assert seen == ["light-blue"]
    assert menu.config["theme"] == "light-blue"


def test_a_command_runs_without_entering_the_menu(monkeypatch):
    monkeypatch.setattr(menu, "_launch_gui", lambda: pytest.fail("the menu must not open"))

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = menu.main_cli(["gc", "GGCC"])

    assert (code, buffer.getvalue().strip()) == (0, "100.0")


# ── _launch_api ──────────────────────────────────────────────────────────────

def test_a_port_that_is_not_a_number_is_reported(captured):
    with redirect_stdout(captured):
        menu._launch_api("not-a-port")

    assert "Invalid port" in captured.getvalue()


@pytest.mark.parametrize("port", [0, 70000, -1])
def test_a_port_outside_the_valid_range_is_reported(captured, port):
    with redirect_stdout(captured):
        menu._launch_api(port)

    assert "Invalid port" in captured.getvalue()


def test_a_missing_extra_tells_the_user_what_to_install(monkeypatch, captured):
    import builtins

    real_import = builtins.__import__

    def without_uvicorn(name, *args, **kwargs):
        if name == "uvicorn":
            raise ImportError("no uvicorn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_uvicorn)

    with redirect_stdout(captured):
        menu._launch_api(8000)

    assert "pip install" in captured.getvalue() or "API extra" in captured.getvalue()


def test_an_unsafe_server_configuration_is_shown_rather_than_served(monkeypatch, captured):
    pytest.importorskip("slowapi", reason="the [api] extra is optional")
    import biosuite.api.server as server

    def refuse():
        raise SystemExit("Refusing to start: insecure production configuration.")

    monkeypatch.setattr(server, "ensure_safe_to_serve", refuse)

    with redirect_stdout(captured):
        menu._launch_api(8000)

    assert "Refusing to start" in captured.getvalue()


def test_the_server_starts_on_the_configured_host_with_the_given_port(monkeypatch,
                                                                      captured):
    pytest.importorskip("slowapi", reason="the [api] extra is optional")
    import biosuite.api.server as server
    started = []

    module = type("uvicorn_stub", (), {"run": staticmethod(
        lambda app, host=None, port=None, **kwargs: started.append((host, port)))})
    monkeypatch.setitem(sys.modules, "uvicorn", module)
    monkeypatch.setattr(server, "ensure_safe_to_serve", lambda: None)
    monkeypatch.setenv("BIOSUITE_API_HOST", "127.0.0.1")

    with redirect_stdout(captured):
        menu._launch_api(8050)

    assert started == [("127.0.0.1", 8050)]
    assert "Starting API server on 127.0.0.1:8050" in captured.getvalue()


def test_the_gui_launcher_builds_and_runs_the_app(monkeypatch):
    import biosuite.gui.main_window as main_window

    started = []

    class App:
        def mainloop(self):
            started.append("mainloop")

    monkeypatch.setattr(main_window, "BioSuiteApp", lambda: App())
    menu._launch_gui()

    assert started == ["mainloop"]


# ── the safe-callable resolver ───────────────────────────────────────────────

def test_the_resolver_turns_an_alias_into_the_function_it_advertises():
    import biosuite.core.sequence as sequence

    assert menu.resolve_safe_callable("gc_content") is sequence.gc_content
    assert menu.resolve_safe_callable("revcomp") is sequence.reverse_complement
    assert menu.resolve_safe_callable("length") is len


def test_the_resolver_accepts_a_qualified_biosuite_path():
    import biosuite.core.sequence as sequence

    assert menu.resolve_safe_callable("biosuite.core.sequence:gc_content") \
        is sequence.gc_content


def test_every_alias_points_at_something_importable():
    import importlib

    assert len(menu.SAFE_FUNCTIONS) == 9
    for alias, path in menu.SAFE_FUNCTIONS.items():
        mod_name, _, attr = path.partition(':')
        target = importlib.import_module(mod_name)
        for part in attr.split('.'):
            target = getattr(target, part)
        assert menu._import_func(path) is target, alias


@pytest.mark.parametrize("attempt", ["", "   ", None])
def test_the_resolver_needs_a_name(attempt):
    with pytest.raises(ValueError, match="No function given"):
        menu.resolve_safe_callable(attempt)


def test_the_resolver_refuses_a_bare_name_that_is_not_an_alias():
    with pytest.raises(ValueError, match="Unknown function"):
        menu.resolve_safe_callable("open")


@pytest.mark.parametrize("attempt", ["os:system", "subprocess:run", "builtins:eval",
                                     "pathlib:Path", "shutil:rmtree"])
def test_the_resolver_refuses_everything_outside_the_biosuite_package(attempt):
    with pytest.raises(ValueError):
        menu.resolve_safe_callable(attempt)


@pytest.mark.parametrize("builtin", ["eval", "exec", "open", "__import__"])
def test_the_builtin_allowlist_is_the_whole_story(builtin):
    with pytest.raises(ValueError, match="Unsupported builtin"):
        menu.resolve_safe_callable(f"builtins:{builtin}")


def test_the_resolver_reports_a_module_without_that_function():
    with pytest.raises(ValueError, match="Cannot resolve"):
        menu.resolve_safe_callable("biosuite.core.sequence:not_a_function")


def test_the_resolver_reports_a_name_that_is_not_callable():
    with pytest.raises(ValueError, match="is not callable"):
        menu.resolve_safe_callable("biosuite.cli.menu:SAFE_FUNCTIONS")


def test_import_func_requires_a_module_colon_function_path():
    with pytest.raises(ValueError):
        menu._import_func("no-colon-here")

    assert menu._import_func("biosuite.core.sequence:gc_content")("GGCC") == 100.0
