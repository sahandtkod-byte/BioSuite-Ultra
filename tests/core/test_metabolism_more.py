"""Metabolism SBML parsing + knockout dispatch offline tests."""
import pytest

from biosuite.core import metabolism as mb


SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="toy">
    <listOfSpecies>
      <species id="A" name="substrate A" compartment="c" hasOnlySubstanceUnits="false"/>
      <species id="B" name="intermediate B" compartment="c" hasOnlySubstanceUnits="false"/>
      <species id="C" name="product C" compartment="c" hasOnlySubstanceUnits="false"/>
    </listOfSpecies>
    <listOfReactions>
      <reaction id="R1" reversible="false" fast="false">
        <listOfReactants>
          <speciesReference species="A" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfProducts>
      </reaction>
      <reaction id="R2" reversible="false" fast="false">
        <listOfReactants>
          <speciesReference species="B" stoichiometry="1" constant="true"/>
        </listOfReactants>
        <listOfProducts>
          <speciesReference species="C" stoichiometry="1" constant="true"/>
        </listOfProducts>
      </reaction>
    </listOfReactions>
  </model>
</sbml>
"""


def test_sbml_parsing_counts(tmp_path):
    p = tmp_path / 'toy.sbml'
    p.write_text(SBML)
    reactions, metabolites, stoich = mb._parse_sbml_simple(str(p))
    assert len(reactions) == 2
    assert len(metabolites) == 3
    assert stoich['R1']['products'][0][0] == 'B'
    assert stoich['R2']['substrates'][0][0] == 'B'
    assert all(isinstance(m, dict) for m in metabolites)


def test_sbml_parsing_empty_file(tmp_path):
    p = tmp_path / 'empty.sbml'
    p.write_text("")
    reactions, metabolites, stoich = mb._parse_sbml_simple(str(p))
    assert reactions == [] and metabolites == []


def test_run_fba_from_sbml(tmp_path):
    p = tmp_path / 'toy.sbml'
    p.write_text(SBML)
    res = mb.run_fba(model_file=str(p))
    assert res.engine in ('builtin', 'cobra')
    assert res.reaction_count >= 0
    assert isinstance(res.message, str)


def test_knockout_sbml_dispatch_no_cobra(tmp_path):
    p = tmp_path / 'toy.sbml'
    p.write_text(SBML)
    out = mb._cobra_knockout(str(p), ['gene1']);
    assert isinstance(out, list)


def test_format_flux_report_from_results(tmp_path):
    p = tmp_path / 'toy.sbml'
    p.write_text(SBML)
    res = mb.run_fba(model_file=str(p))
    rep = mb.format_flux_report(res)
    assert isinstance(rep, str) and 'Engine' in rep
