"""Behavior contracts for UpSet exclusive intersection computation."""
import matplotlib
matplotlib.use('Agg')

from biosuite.plotting.upset_plots import compute_upset_matrix


def _mask_dict(sets):
    labels, matrix, counts = compute_upset_matrix(sets)
    names = {i: n for i, n in enumerate(labels)}
    out = {}
    for mask, cnt in zip(matrix, counts):
        combo = " + ".join(names[i] for i, b in enumerate(mask) if b)
        out[combo] = cnt
    return out


def test_exclusive_regions_computed():
    sets = {'X': {1, 2}, 'Y': {1, 2, 3}, 'Z': {2, 3}}
    m = _mask_dict(sets)
    assert m.get('X + Y', 0) == 1       # exclusive {1}; {2} excluded (also in Z)
    assert m.get('Y + Z', 0) == 1       # exclusive {3}
    assert m.get('X + Z', 0) == 0       # empty exclusive region dropped
    assert m.get('X + Y + Z', 0) == 1   # {2} shared by all three


def test_non_adjacent_pair_computed():
    sets = {'A': {1, 2, 3}, 'B': {2, 4}, 'C': {3, 5}}
    m = _mask_dict(sets)
    assert m.get('A + C', 0) == 1       # exclusive {3} — every pair is checked
    assert m.get('A + B', 0) == 1
    assert m.get('B + C', 0) == 0


def test_upset_from_single_set():
    m = _mask_dict({'A': {1, 2, 3}})
    assert m['A'] == 3
