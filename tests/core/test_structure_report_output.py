"""The text ``format_structure_report`` returns: which sections appear, and that an
absent section never leaves a blank line behind.

Measured before this cluster, the optional sections contributed ``""`` placeholders to
the line list and the list was joined as it stood, so
``format_structure_report(StructureInfo(name="x"))`` ended with
``…Chains: 0 ()\n\n\n`` - two blank lines, four when the secondary-structure dict
totalled zero.  Both the GUI text widget and the CLI print show them as gaps between the
sections.
"""
import pytest

from biosuite.core import structure as S

FULL_REPORT = [
    "=== Protein Structure Report ===",
    "Name: 1UBQ",
    "Atoms: 118",
    "Residues: 76",
    "Chains: 2 (A, B)",
    "Resolution: 1.80 Å",
    "Method: X-RAY DIFFRACTION",
    "Surface area: 4200 Å²",
    "Secondary structure: 53% helix, 13% sheet, 34% coil",
    "Ramachandran: 3 angles (phi: 1.0±84.2°)",
    "Note: boom",
]


@pytest.mark.parametrize("info", [
    pytest.param(S.StructureInfo(name="x"), id="bare"),
    pytest.param(S.StructureInfo(name="x", secondary_structure={"H": 0, "E": 0, "C": 0}),
                 id="all-zero-secondary"),
    pytest.param(S.StructureInfo(name="x", resolution=0.0, method="", surface_area=0.0,
                                 ramachandran={"phi": [], "psi": []}), id="no-sections"),
    pytest.param(S.StructureInfo(name="x", message=""), id="empty-note"),
])
def test_the_report_never_hands_back_blank_lines(info):
    text = S.format_structure_report(info)
    assert text.splitlines()[0] == "=== Protein Structure Report ==="
    assert all(line for line in text.splitlines()), repr(text)
    assert text == text.strip()


def test_a_full_report_prints_every_section_once():
    info = S.StructureInfo(name="1UBQ", num_atoms=118, num_residues=76, num_chains=2,
                           chains=["A", "B"], resolution=1.8, method="X-RAY DIFFRACTION",
                           surface_area=4200.5, secondary_structure={"H": 40, "E": 10,
                                                                    "C": 26},
                           ramachandran={"phi": [-60.0, -57.0, 120.0],
                                         "psi": [-45.0, -47.0, 130.0]},
                           message="boom")
    assert S.format_structure_report(info).splitlines() == FULL_REPORT


def test_a_report_without_measurements_prints_only_the_five_fixed_lines():
    assert S.format_structure_report(S.StructureInfo(name="x")).splitlines() == [
        "=== Protein Structure Report ===",
        "Name: x",
        "Atoms: 0",
        "Residues: 0",
        "Chains: 0 ()",
    ]


def test_the_secondary_structure_line_needs_usable_counts():
    for secondary, shows_line in [
        ({"H": 0, "E": 0, "C": 0}, False),
        ({"H": 1, "E": 0, "C": 0}, True),
        (["H"], False),
        ({"Q": 3}, False),
    ]:
        text = S.format_structure_report(
            S.StructureInfo(name="x", secondary_structure=secondary))
        assert ("Secondary structure" in text) is shows_line, secondary
        assert "\n\n" not in text, secondary
