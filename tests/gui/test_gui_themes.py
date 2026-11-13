"""The three palettes and the plot catalogue agree on their shape.

Every theme is read through the same keys, so a missing colour in one palette is
a KeyError at the moment the user switches to it — on the user's machine, not in
the theme dictionary.  The catalogue's ids become the gallery's dispatch keys, so
a duplicate would silently replace one plot with another.
"""
import re

from biosuite.gui.themes import PLOT_CATEGORIES, THEMES

#: Keys the app reads from ``self.T``; a palette without them cannot be applied.
REQUIRED_KEYS = {
    'name', 'ctk_mode', 'bg', 'bg_secondary', 'card', 'card_hover',
    'accent', 'accent_dim', 'accent_glow', 'text', 'text_dim', 'text_muted',
    'border', 'border_light', 'sidebar_bg', 'sidebar_text', 'sidebar_hover',
    'sidebar_active', 'sidebar_active_text', 'danger', 'success', 'input_bg',
    'scrollbar', 'header_accent', 'overlay', 'dialog_bg', 'dialog_border',
}


def test_every_theme_defines_the_same_palette_keys():
    key_sets = {name: set(theme) for name, theme in THEMES.items()}

    assert len(key_sets) == 3
    assert len({frozenset(keys) for keys in key_sets.values()}) == 1, key_sets
    assert REQUIRED_KEYS <= next(iter(key_sets.values()))


def test_every_colour_is_a_hex_triplet():
    for theme_name, theme in THEMES.items():
        for key, value in theme.items():
            if key in ('name', 'ctk_mode'):
                continue
            assert re.fullmatch(r'#[0-9a-fA-F]{6}', value), (
                f"{theme_name}.{key} is not a #rrggbb colour: {value!r}")


def test_every_theme_declares_a_mode_and_a_display_name():
    for theme_name, theme in THEMES.items():
        assert theme['ctk_mode'] in ('dark', 'light'), theme_name
        assert isinstance(theme['name'], str) and theme['name'].strip(), theme_name


def test_the_plot_catalogue_has_nine_categories_and_forty_plot_types():
    entries = [entry for category in PLOT_CATEGORIES.values() for entry in category]

    assert len(PLOT_CATEGORIES) == 9
    assert len(entries) == 40
    assert len({plot_id for _label, plot_id in entries}) == 40, "a duplicate id hides a plot"


def test_every_catalogue_entry_names_a_label_and_an_id():
    for category, entries in PLOT_CATEGORIES.items():
        assert entries, category
        for label, plot_id in entries:
            assert isinstance(label, str) and label
            assert isinstance(plot_id, str) and plot_id
