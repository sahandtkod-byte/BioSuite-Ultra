"""The menu options have to work when the session actually has data.

``test_cli_menu_option_sessions.py`` drives every option with empty answers; this file
drives the same 116 options with real files — a FASTA, a count matrix, a BED, a VCF, a
Newick string — and keeps the network switched off.  The empty sweep cannot reach the
``if data:`` half of each branch, and that half is where the defects of the
``design_primers(template, start, end)`` kind live: a call site written against a
signature the core module never had, which only fires once the user supplies a
sequence.
"""
import builtins
import io
import re
import socket
from contextlib import redirect_stderr
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from biosuite.cli import menu

MENU_SOURCE = Path(menu.__file__).read_text(encoding="utf-8")
HANDLER_SOURCE = MENU_SOURCE[MENU_SOURCE.index("def _handle_menu_choice"):]
OPTIONS = sorted(set(re.findall(r"(?:el)?if choice == '([a-z0-9]+)'", HANDLER_SOURCE)))

INTERNAL_ERRORS = (NameError, AttributeError, TypeError, IndexError, UnboundLocalError,
                   ZeroDivisionError, KeyError)

#: Text an option must never put in front of a user, however it got there.
TRACEBACK_MARKS = ("Traceback (most recent call last)", "object at 0x")

DNA = "ATGCGTACGTTAGCGTACAGTACTGCATGCATCGTCGGGATCGATCGCGATCGATCGCGATCGATCGATCGATCGATCG"
PROTEIN = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPMSIRSGKYVSAYRNVYTAHESTVVPIFGTAVVGCVIALY"

FIXTURES = {
    "sequences.fasta": f">seq_one\n{DNA}\n>seq_two\n{DNA[::-1]}\n",
    "aligned.fasta": ">a\nACGTACGT\n>b\nACGAACGT\n>c\nACGTACGA\n",
    "counts.csv": "gene,s1,s2,s3,s4\nG1,10,12,100,120\nG2,100,120,10,12\n"
                  "G3,50,50,50,50\nG4,7,9,70,80\n",
    "regions.bed": "chr1\t100\t200\tgeneA\t0.9\nchr1\t300\t400\tgeneB\t0.5\n",
    "annotations.gff": "chr1\tprod\tgene\t101\t200\t.\t+\t.\tID=geneA\n"
                       "chr1\tprod\texon\t101\t150\t.\t+\t.\tParent=geneA\n",
    "variants.vcf": "##fileformat=VCFv4.2\n"
                    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                    "chr1\t1000\trs1\tA\tG\t99\tPASS\tAF=0.3\n"
                    "chr1\t2000\trs2\tC\tT\t55\tPASS\tAF=0.1\n",
    "reads.fastq": "@r1\nACGTACGTAC\n+\nIIIIIIIIII\n@r2\nACGAACGTAC\n+\nIIIIHHIIII\n",
    "alignments.sam": "@HD\tVN:1.6\n@SQ\tSN:chr1\tLN:1000\n"
                      "r1\t0\tchr1\t100\t60\t10M\t*\t0\t0\tACGTACGTAC\tIIIIIIIIII\n",
    "tree.nwk": "((human:0.1,chimp:0.1):0.2,gorilla:0.3);\n",
    "genes.txt": "BRCA1\nTP53\nEGFR\nMYC\n",
    "gwas.csv": "chrom,pos,snp_id,case_alt,case_ref,ctrl_alt,ctrl_ref\n"
                "chr1,1000,rs1,90,10,10,90\nchr1,2000,rs2,50,50,50,50\n"
                "chr2,3000,rs3,80,20,15,85\n",
    "model.xml": """<?xml version="1.0" encoding="UTF-8"?>
<sbml level="2" version="1"><model id="m1" name="demo">\n"""\
                 """<listOfSpecies><species id="A" concentration="1"/></model></sbml>\n""",
    "stockholm.sto": "# STOCKHOLM 1.0\na ACGTACGT\nb ACGAACGT\n#=GF SQ 2\n//\n",
}


def _answer_with_data(prompts_seen):
    """Answer every prompt with something real, chosen from the question's wording."""
    lines = {"rows": iter(["0,1,2", "1,1,0", "2,2,1", ""]),
             "interactions": iter(["A,B,1.0", "B,C,2.0", ""]),
             "genomes": iter(["GeneA,GeneB,GeneC", "GeneA,GeneC,GeneB"])}

    def answer(prompt=""):
        p = prompt.lower()
        prompts_seen.append(prompt)
        for marker, stream in lines.items():
            if marker in p and "one per line" in p:
                return next(stream, "")
        if "newick" in p:
            return FIXTURES["tree.nwk"].strip()
        if "count matrix" in p or "counts" in p:
            return "counts.csv"
        if "database" in p and "fasta" in p:
            return "sequences.fasta"
        if "aligned" in p and "fasta" in p:
            return "aligned.fasta"
        if "query" in p or "search" in p:
            return ""                                  # never reach the network
        if "fasta" in p or ".fa" in p:
            return "sequences.fasta"
        if "bed" in p:
            return "regions.bed"
        if "gff" in p:
            return "annotations.gff"
        if "vcf" in p:
            return "variants.vcf"
        if "fastq" in p or "reads" in p:
            return "reads.fastq"
        if "sam" in p or "bam" in p:
            return "alignments.sam"
        if "stockholm" in p:
            return "stockholm.sto"
        if "sbml" in p:
            return "model.xml"
        if "csv" in p or "file path" in p or "any file" in p or "path:" in p:
            return "regions.bed"
        if "snp" in p and ("csv" in p or "file" in p):
            return "gwas.csv"
        if "conditions" in p:
            return "A,A,B,B"
        if "gene lengths" in p:
            return "1.0,2.0,1.5,2.5"
        if "intensities" in p:
            return "10,25,40,80,60,30"
        if "gene ids" in p:
            return "BRCA1,TP53,EGFR"
        if "protein" in p and "seq" in p:
            return PROTEIN
        if "dna" in p or "sequence" in p or "template" in p or "target" in p:
            return DNA
        if "enzyme" in p:
            return "EcoRI"
        if "primer" in p:
            return DNA[:20]
        if "chromosome" in p:
            return "chr1"
        if "save this plot" in p or "show data summary" in p:
            return "n"
        if "(y/n)" in p or "y/n" in p:
            return "n"
        if "number of sets" in p:
            return "2"
        if "use default" in p:
            return "y"
        return ""

    return answer


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """A unit test that reaches the internet is a defect, not a slow test."""
    def refuse(*args, **kwargs):
        raise RuntimeError("network access is not allowed in the menu sweep")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture
def data_session(tmp_path, monkeypatch, capsys):
    import biosuite.gui.main_window as main_window

    for name, body in FIXTURES.items():
        (tmp_path / name).write_text(body, encoding="utf-8")

    seen: list = []
    monkeypatch.setattr(menu, "_launch_gui", lambda: None)
    monkeypatch.setattr(menu, "_launch_api", lambda port=None: None)
    monkeypatch.setattr(main_window, "BioSuiteApp", lambda *a, **k: SimpleApp())
    monkeypatch.setitem(menu.config, "quiet", True)
    monkeypatch.setattr(builtins, "input", _answer_with_data(seen))
    monkeypatch.chdir(tmp_path)

    def run(choice):
        figures_before = set(plt.get_fignums())
        broken, refused = None, None
        try:
            with redirect_stderr(io.StringIO()):
                menu._handle_menu_choice(choice)
        except INTERNAL_ERRORS as exc:
            broken = exc
        except Exception as exc:                       # noqa: BLE001 - reported below
            refused = exc
        out = capsys.readouterr().out
        written = sorted(str(p.relative_to(Path.cwd()))
                         for p in Path.cwd().rglob("*") if p.is_file())
        return {"out": out, "broken": broken, "refused": refused,
                "written": written, "prompts": list(seen),
                "leaked": set(plt.get_fignums()) - figures_before}

    return run


class SimpleApp:
    def mainloop(self):
        return None


@pytest.mark.parametrize("choice", OPTIONS)
def test_option_survives_a_session_full_of_data(data_session, choice):
    result = data_session(choice)

    if result["broken"] is not None:
        pytest.fail(f"option {choice} raised {type(result['broken']).__name__}: "
                    f"{result['broken']}")
    assert not result["leaked"], \
        f"option {choice} left {len(result['leaked'])} figure(s) open"
    announced = result["out"] + "\n".join(result["prompts"])
    silent = [f for f in result["written"] if f not in announced
              and Path(f).name not in FIXTURES]
    assert not silent, f"option {choice} wrote {silent} without mentioning them"
    for mark in TRACEBACK_MARKS:
        assert mark not in result["out"], \
            f"option {choice} printed {mark!r} to the user"
    assert result["out"].strip() or result["refused"] is not None, \
        f"option {choice} neither printed anything nor raised anything"
