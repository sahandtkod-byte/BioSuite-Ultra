"""`set_theme` — the names it knows, and what an unknown one does.

Measured before, any string that was not one of the three dark names got the
**light** palette with no word said, so `biosuite theme light_cyan` (or a typo left
in `biosuite_config.json`) recoloured every plot by accident.  An unknown name now
keeps the current colours and lists the names that exist.
"""
import logging

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def restore_rcparams():
    saved = {key: plt.rcParams[key] for key in ('figure.facecolor', 'text.color')}
    yield
    for key, value in saved.items():
        plt.rcParams[key] = value


def _names():
    from biosuite.core import utils
    return utils.DARK_THEMES, utils.LIGHT_THEMES, utils.KNOWN_THEMES


def test_the_theme_names_are_the_ones_the_cli_and_gui_offer():
    dark, light, known = _names()
    assert dark == ('dark', 'dark-green', 'dark-purple')
    assert light == ('light', 'light-blue')
    assert known == frozenset(dark + light)


@pytest.mark.parametrize("choice", ['dark', 'dark-green', 'dark-purple'])
def test_a_dark_theme_paints_the_dark_palette(choice):
    from biosuite.core import utils
    utils.set_theme(choice)
    assert plt.rcParams['figure.facecolor'] == '#121212'
    assert plt.rcParams['text.color'] == '#00ff99'


@pytest.mark.parametrize("choice", ['light', 'light-blue'])
def test_a_light_theme_paints_the_teal_palette(choice):
    from biosuite.core import utils
    utils.set_theme(choice)
    assert plt.rcParams['figure.facecolor'] == '#e6f7fa'
    assert plt.rcParams['text.color'] == '#006064'


@pytest.mark.parametrize("choice", ['light_cyan', 'LIGHT', 'dark_green', '', None, 3])
def test_an_unknown_theme_changes_nothing_and_says_why(choice, caplog):
    """`'LIGHT'` is not `'light'`: the names are matched exactly, and the refusal
    lists them so the caller can see the spelling it needed."""
    from biosuite.core import utils
    utils.set_theme('dark-green')
    before = plt.rcParams['figure.facecolor']
    with caplog.at_level(logging.WARNING):
        utils.set_theme(choice)
    assert plt.rcParams['figure.facecolor'] == before == '#121212'
    records = [r for r in caplog.records if r.name == 'biosuite.core.utils']
    assert len(records) == 1
    assert records[0].getMessage().startswith(f"Unknown theme {choice!r};")
    assert "dark, dark-green, dark-purple, light, light-blue" in records[0].getMessage()
