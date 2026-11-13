"""The declared matplotlib floor has to cover the keyword arguments the code uses.

``plot_api.boxplot`` and ``interactive_plots`` both call
``ax.boxplot(..., tick_labels=...)``, and matplotlib's own deprecation text names
the version that introduced it: "The 'labels' parameter of boxplot() has been
renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be
dropped in 3.11."  ``requirements.txt`` and ``pyproject.toml`` declared
``matplotlib>=3.7``, so ``pip install`` on a 3.7 or 3.8 machine gave a package
whose boxplot died with ``TypeError: got an unexpected keyword argument
'tick_labels'``.
"""
import ast
import pathlib
import re
import warnings

import matplotlib
import matplotlib.pyplot as plt
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

#: Keyword arguments with a known matplotlib introduction version, measured from
#: the library's own messages rather than from memory.
VERSION_SENSITIVE = [
    ('tick_labels=', (3, 9), 'Axes.boxplot(tick_labels=) — renamed from labels in 3.9'),
]


def _uses(keyword, package='biosuite'):
    hits = []
    for path in sorted((REPO / package).rglob('*.py')):
        text = path.read_text(encoding='utf-8')
        if keyword in text:
            hits.append(path.relative_to(REPO))
    return hits


def _declared_floor(text, name='matplotlib'):
    match = re.search(rf'{name}>=\s*(\d+)\.(\d+)', text)
    assert match, f"{name}>= is not declared"
    return int(match.group(1)), int(match.group(2))


def test_the_installed_library_names_the_version_that_renamed_the_keyword():
    """The evidence the floor change rests on, taken from matplotlib itself."""
    fig, ax = plt.subplots()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        ax.boxplot([[1.0, 2.0], [3.0, 4.0]], labels=['a', 'b'])
    texts = [str(w.message) for w in caught]
    plt.close(fig)

    assert any("renamed 'tick_labels' since Matplotlib 3.9" in t for t in texts), texts


@pytest.mark.parametrize('keyword,minimum,reason', VERSION_SENSITIVE)
def test_every_declared_floor_covers_the_kwargs_used(keyword, minimum, reason):
    users = _uses(keyword)
    if not users:
        pytest.fail(f"{reason} is no longer used in biosuite/; drop it from "
                    f"VERSION_SENSITIVE instead of letting the pin rot")

    requirements = (REPO / 'requirements.txt').read_text(encoding='utf-8')
    pyproject = (REPO / 'pyproject.toml').read_text(encoding='utf-8')

    assert _declared_floor(requirements) >= minimum, \
        f"requirements.txt allows matplotlib below {minimum} but {reason} is used by {users}"
    assert _declared_floor(pyproject) >= minimum, \
        f"pyproject.toml allows matplotlib below {minimum} but {reason} is used"


def test_requirements_and_pyproject_declare_the_same_runtime_dependencies():
    """requirements.txt says it mirrors the pyproject table; this keeps that
    sentence true, so a floor raised in one file cannot be missed in the other."""
    def pairs(text, pattern):
        return dict(re.findall(pattern, text, re.M))

    requirement_pairs = pairs((REPO / 'requirements.txt').read_text(encoding='utf-8'),
                              r'^([a-zA-Z0-9_.-]+)\s*(>=.*?)$')
    block = re.search(r'dependencies = \[(.*?)\]',
                      (REPO / 'pyproject.toml').read_text(encoding='utf-8'), re.S).group(1)
    pyproject_pairs = pairs(block, r'"([a-zA-Z0-9_.-]+)\s*(>=.*?)"')

    shared = set(requirement_pairs) & set(pyproject_pairs)
    assert shared, "no overlapping dependencies to compare"
    differing = {name: (requirement_pairs[name], pyproject_pairs[name])
                 for name in shared if requirement_pairs[name] != pyproject_pairs[name]}

    assert differing == {}, f"{differing} differ between the two files"


def test_the_matplotlib_floor_is_not_declared_above_the_installed_library():
    """A floor nobody can satisfy because the wheel does not exist is as bad as a
    floor that is too low: the declared minimum must be a release we can run."""
    floor = _declared_floor((REPO / 'requirements.txt').read_text(encoding='utf-8'))
    installed = tuple(int(part) for part in matplotlib.__version__.split('.')[:2])

    assert floor <= installed, f"installed matplotlib {matplotlib.__version__} is below {floor}"
