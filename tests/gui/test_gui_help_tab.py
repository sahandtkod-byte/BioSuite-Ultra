"""The Help & Guides tab: what each topic button actually puts in the reading pane.

The tab is a list of buttons whose command calls ``_show_guide`` with one documentation
method; the only observable output is the text that lands in ``help_text``.  Each test
therefore clicks the real button and reads the pane, and the expected content is the
guide's own title line plus a line that only that guide carries — so a button wired to
the wrong guide, or a pane that is not cleared between clicks, fails loudly.
"""
import customtkinter as ctk
import pytest

from biosuite import __version__

GUIDES = [
    ("Getting Started", "GETTING STARTED WITH BIOSUITE ULTRA",
     f"Welcome to BioSuite Ultra v{__version__}"),
    ("Sequence Analysis", "SEQUENCE ANALYSIS GUIDE",
     "4. Click \"Stats\" for detailed breakdown"),
    ("Alignment Tools", "ALIGNMENT TOOLS GUIDE",
     "Needleman-Wunsch = global alignment (full sequences)"),
    ("Transcriptomics", "TRANSCRIPTOMICS GUIDE",
     "4. Results show TPM values per transcript"),
    ("Genomics & NGS", "GENOMICS & NGS GUIDE",
     "VARIANT CALLING (VCF):"),
    ("Single-Cell", "SINGLE-CELL RNA-SEQ GUIDE",
     "7. Marker Detection -- Find cluster markers"),
    ("Proteomics", "PROTEOMICS GUIDE",
     "Ramachandran plot data"),
    ("CRISPR Design", "CRISPR GUIDE RNA DESIGN GUIDE",
     "- Cas12a -- Different PAM (TTTV, 23bp guide)"),
    ("Metagenomics", "METAGENOMICS GUIDE",
     "- Classification uses the built-in k-mer classifier"),
    ("Population Genetics", "POPULATION GENETICS GUIDE",
     "1,1,2,2,1,0"),
    ("Machine Learning", "MACHINE LEARNING GUIDE",
     "- Feature importances (RF only)"),
    ("Database Search", "DATABASE SEARCH GUIDE",
     "TIP: Add API keys in \"API Keys Config\" for faster NCBI searches."),
    ("File Formats", "FILE FORMATS GUIDE",
     "(A:0.1,B:0.2)C:0.3;"),
    ("API Keys Setup", "API KEYS SETUP GUIDE",
     "TIP: The NCBI key gives 10x faster searches (10/sec vs 3/sec)."),
    ("External Tools", "EXTERNAL TOOLS GUIDE",
     "- COBRApy -- Metabolic modeling"),
    ("Visualization", "VISUALIZATION GUIDE",
     "- UpSet Plot -- Multi-set intersections"),
    ("Workflow & Pipeline", "WORKFLOW & PIPELINE GUIDE",
     "  revcomp:reverse_complement"),
    ("GO Browser", "GENE ONTOLOGY BROWSER",
     "TIP: The GO browser includes 31 built-in terms for quick reference."),
    ("GWAS Analysis", "GWAS ANALYSIS GUIDE",
     "- p < 5e-8 = Genome-wide significant"),
    ("Epitope Prediction", "EPITOPE PREDICTION GUIDE",
     "- Peptide lengths: 8-11 amino acids"),
    ("16S rRNA Pipeline", "16S rRNA CLASSIFICATION GUIDE",
     "- Helicobacter pylori"),
    ("SV / CNV Detection", "SV / CNV DETECTION GUIDE",
     "- CN=3,4 -- Duplication/amplification"),
    ("BigWig Reader", "BIGWIG READER GUIDE",
     "- ENCODE project data"),
    ("Keyboard Shortcuts", "KEYBOARD SHORTCUTS",
     "- Ctrl+S saves the Sequence tab's results box or the latest plot; other tabs"),
    ("REST API", "REST API SERVER",
     "- /api/v1/database/* -- NCBI, UniProt, PDB, KEGG search"),
]


def _guide_button(app, label):
    """The topic button the user clicks, found by the label printed on it."""
    stack = [app.frames["help"]]
    while stack:
        widget = stack.pop()
        if isinstance(widget, ctk.CTkButton) and widget.cget("text") == label:
            return widget
        stack.extend(widget.winfo_children())
    raise AssertionError(f"the help tab has no {label!r} button")


def _click_guide(ui, label):
    _guide_button(ui.app, label).invoke()
    return ui.text("help_text")


@pytest.mark.parametrize("label,title,marker", GUIDES,
                         ids=[row[0] for row in GUIDES])
def test_the_topic_button_shows_its_own_guide(ui, label, title, marker):
    ui.clear("help_text")

    box = _click_guide(ui, label)

    assert box.splitlines()[0] == title, box[:80]
    assert marker in box, f"{label!r} never reached the reading pane"
    assert ui.messages == []


def test_the_getting_started_guide_carries_the_app_version(ui):
    ui.clear("help_text")

    box = _click_guide(ui, "Getting Started")

    assert f"BioSuite Ultra v{__version__}" in box, "the guide quotes a stale version"
    assert "F1 -- Open help" in box


def test_clicking_another_topic_replaces_the_previous_guide(ui):
    _click_guide(ui, "REST API")
    before = ui.text("help_text")
    assert "REST API SERVER" in before

    _click_guide(ui, "GWAS Analysis")

    after = ui.text("help_text")
    assert "GWAS ANALYSIS GUIDE" in after
    assert "REST API SERVER" not in after, "the pane was not cleared between topics"
    assert after.strip() != before.strip()


def test_every_topic_button_shows_a_different_guide(ui):
    shown = {}
    for label, _title, _marker in GUIDES:
        ui.clear("help_text")
        shown[label] = _click_guide(ui, label).strip()
        assert shown[label], f"clicking {label!r} left the reading pane empty"

    assert len(shown) == 25, "the tab must offer one button per documented topic"
    texts = list(shown.values())
    assert len(set(texts)) == len(texts), "two topic buttons show the same guide"


def test_show_guide_inserts_the_text_once_and_never_twice(ui):
    """``_show_guide`` clears the pane first, so a second call cannot duplicate it."""
    ui.clear("help_text")

    ui.app._show_guide(ui.app._help_epitope)
    once = ui.text("help_text")

    ui.app._show_guide(ui.app._help_epitope)
    assert ui.text("help_text") == once, "the pane is not cleared before a guide is inserted"
    assert once.count("EPITOPE PREDICTION GUIDE") == 1

    assert _click_guide(ui, "Epitope Prediction").strip() == once.strip(), \
        "the topic button and the shared callback must show the same guide"
