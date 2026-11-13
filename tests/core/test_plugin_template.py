"""The plugin template `create_plugin_template()` writes, and the names it accepts.

A template is a promise that the thing it generated can be installed and loaded.
The old layout put ``__init__.py`` straight in the dashed project folder and aimed
the entry point at a module name that could never be imported, so a generated
plugin failed at ``ep.load()``; the name argument was interpolated into a path and
into source without a look, which let ``../../outside`` write elsewhere.
"""
import importlib
import os

import pytest

from biosuite.core import plugin as pl


@pytest.fixture
def template(tmp_path):
    def _make(name="my_tool"):
        pl.PluginManager().create_plugin_template(name, output_dir=str(tmp_path))
        return os.path.join(str(tmp_path), f"biosuite-plugin-{name}")
    return _make


# ── the layout ──────────────────────────────────────────────────────────────

def test_the_three_pieces_are_written_in_the_documented_layout(template):
    project = template("my_tool")
    assert sorted(os.listdir(project)) == ["README.md", "my_tool", "pyproject.toml"]
    assert os.listdir(os.path.join(project, "my_tool")) == ["__init__.py"]


def test_the_package_module_is_importable_from_the_project_directory(template,
                                                                     monkeypatch):
    """This is what an entry point needs: the part before the colon has to be a
    module Python can find."""
    project = template("my_tool")
    monkeypatch.syspath_prepend(project)
    module = importlib.import_module("my_tool")
    assert issubclass(module.Plugin, pl.BioSuitePlugin)
    assert module.Plugin().name() == "my_tool"


def test_a_dashed_plugin_name_still_gets_an_importable_module(template, monkeypatch):
    """'my-dashed-tool' is a legal distribution name and an illegal module name;
    the template has to write the module as my_dashed_tool and point the entry
    point there."""
    project = template("my-dashed-tool")
    assert os.path.isdir(os.path.join(project, "my_dashed_tool"))
    toml = open(os.path.join(project, "pyproject.toml"), encoding="utf-8").read()
    assert 'my-dashed-tool = "my_dashed_tool:Plugin"' in toml
    assert 'name = "biosuite-plugin-my-dashed-tool"' in toml
    monkeypatch.syspath_prepend(project)
    import importlib
    module = importlib.import_module("my_dashed_tool")
    assert module.Plugin().name() == "my-dashed-tool"


def test_the_generated_code_is_a_loadable_local_plugin(template, fake_home, monkeypatch):
    """Copying the package folder into ~/.biosuite/plugins is the documented
    no-install workflow, so the generated class has to satisfy the loader."""
    project = template("drop_in")
    plugins_dir = os.path.join(str(fake_home), ".biosuite", "plugins")
    os.makedirs(plugins_dir, exist_ok=True)
    import shutil

    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, "entry_points",
                        lambda: _NoEntryPoints())
    shutil.copytree(os.path.join(project, "drop_in"),
                    os.path.join(plugins_dir, "drop_in"))
    manager = pl.PluginManager()
    assert manager.discover() == ["drop_in"]
    assert manager.plugins["drop_in"].version == "0.1.0"
    assert manager.load_plugin("drop_in") is True
    assert manager.get_plugin("drop_in").description() == \
        "A BioSuite plugin for drop_in analysis"


class _NoEntryPoints:
    def select(self, group):
        return []


def test_the_template_class_answers_the_contract(template):
    init = open(os.path.join(template("my_tool"), "my_tool", "__init__.py"),
               encoding="utf-8").read()
    assert "class Plugin(BioSuitePlugin)" in init
    assert 'return "my_tool"' in init
    assert "def register(self, app):" in init
    assert "def dependencies(self):" in init


# ── the metadata the template declares ──────────────────────────────────────

def test_the_requirements_match_a_python_this_project_runs_on(template):
    toml = open(os.path.join(template("my_tool"), "pyproject.toml"),
                encoding="utf-8").read()
    assert 'requires-python = ">=3.10"' in toml, \
        "the core needs 3.10; a template that allows 3.9 generates an unloadable plugin"


def test_the_biosuite_floor_tracks_the_installed_version(template):
    from biosuite import __version__
    toml = open(os.path.join(template("my_tool"), "pyproject.toml"),
                encoding="utf-8").read()
    floor = f'"biosuite>={__version__.split(".")[0]}.0"'
    assert floor in toml
    assert "biosuite>=4.0" not in toml, "two major versions behind the product"


def test_setuptools_is_told_which_package_to_ship(template):
    toml = open(os.path.join(template("my_tool"), "pyproject.toml"),
                encoding="utf-8").read()
    assert '[tool.setuptools]' in toml
    assert 'packages = ["my_tool"]' in toml


def test_the_readme_documents_both_ways_of_using_it(template):
    readme = open(os.path.join(template("my_tool"), "README.md"),
                  encoding="utf-8").read()
    assert "## Without installing it" in readme
    assert "~/.biosuite/plugins/my_tool" in readme
    assert "my_tool/" in readme and "__init__.py" in readme
    assert 'pm.load_plugin("my_tool")' in readme


# ── the instructions it prints ──────────────────────────────────────────────

def test_the_publish_step_is_the_command_that_publishes(template, capsys):
    project = template("my_tool")
    out = capsys.readouterr().out
    assert "python -m twine upload dist/*" in out
    assert "python -m twine dist/*" not in out
    assert f'pip install -e "{project}"' in out, "a path with a space needs quotes"
    assert "drop \"my_tool\" into ~/.biosuite/plugins/" in out


def test_a_path_with_a_space_is_printed_quoted(tmp_path, capsys):
    spaced = tmp_path / "with space"
    spaced.mkdir()
    pl.PluginManager().create_plugin_template("spaced_tool", output_dir=str(spaced))
    out = capsys.readouterr().out
    expected = os.path.join(str(spaced), "biosuite-plugin-spaced_tool")
    assert f'pip install -e "{expected}"' in out


# ── the name it accepts ─────────────────────────────────────────────────────

@pytest.mark.parametrize("good", ["my_tool", "my-tool", "MyTool2", "_private",
                                  "normal", "tool_name_with_underscores"])
def test_names_that_work_as_directories_and_modules_are_accepted(tmp_path, good):
    pl.PluginManager().create_plugin_template(good, output_dir=str(tmp_path))
    project = tmp_path / f"biosuite-plugin-{good}"
    assert (project / "pyproject.toml").exists()
    assert (project / good.replace("-", "_") / "__init__.py").exists()


@pytest.mark.parametrize("bad", [
    "../evil", "../../evil", "with/slash", "with\\backslash", "", "   ",
    'with"quote', "with space", "1starts_with_digit", "-leading-dash",
    "with;semicolon", "with|pipe", "with`tick", "with$(sub)", "dot.dot",
    None, 7, ["list"],
])
def test_a_name_that_cannot_be_a_module_or_a_path_is_refused(tmp_path, bad):
    """Every one of these either wrote outside the plugin directory, died inside
    the operating system, or produced a package that could not be imported."""
    before = sorted(os.listdir(str(tmp_path)))
    with pytest.raises((ValueError, TypeError)):
        pl.PluginManager().create_plugin_template(bad, output_dir=str(tmp_path))
    assert sorted(os.listdir(str(tmp_path))) == before, "a refusal must not write"


def test_a_name_that_would_be_a_python_keyword_is_refused(tmp_path):
    """'class' passes the pattern and still cannot be a module: the generated
    entry point 'class:Plugin' would never load."""
    with pytest.raises(ValueError, match="Python keyword"):
        pl.PluginManager().create_plugin_template("class", output_dir=str(tmp_path))
    assert os.listdir(str(tmp_path)) == []


def test_a_positive_control_after_the_keyword_refusal(tmp_path):
    pl.PluginManager().create_plugin_template("classified", output_dir=str(tmp_path))
    assert (tmp_path / "biosuite-plugin-classified" / "classified").is_dir()


def test_a_traversal_name_writes_nothing_anywhere(tmp_path):
    """The proof that the refusal above is about the escape, not the message."""
    target = tmp_path / "out"
    target.mkdir()
    with pytest.raises(ValueError, match="must start with a letter"):
        pl.PluginManager().create_plugin_template("../../c23_escape",
                                                  output_dir=str(target))
    assert list(tmp_path.parent.rglob("c23_escape")) == []
    assert os.listdir(str(target)) == []


def test_a_positive_control_after_the_refusals(tmp_path):
    pl.PluginManager().create_plugin_template("allowed_name", output_dir=str(tmp_path))
    assert (tmp_path / "biosuite-plugin-allowed_name").exists()


def test_re_running_a_name_updates_the_template_in_place(tmp_path):
    """A user iterating on a name must not be stopped by the existing folder."""
    manager = pl.PluginManager()
    manager.create_plugin_template("again", output_dir=str(tmp_path))
    manager.create_plugin_template("again", output_dir=str(tmp_path))
    assert (tmp_path / "biosuite-plugin-again" / "again" / "__init__.py").exists()


def test_the_template_is_valid_python_source(tmp_path):
    """Compiling the generated file is the cheapest proof that the name could not
    break out of the string it is interpolated into."""
    project = tmp_path / "p"
    project.mkdir()
    pl.PluginManager().create_plugin_template("compiled_check", output_dir=str(project))
    source = (project / "biosuite-plugin-compiled_check" / "compiled_check" /
              "__init__.py").read_text(encoding="utf-8")
    namespace = compile(source, "generated", "exec")
    assert namespace is not None
    assert "class Plugin(BioSuitePlugin)" in source
