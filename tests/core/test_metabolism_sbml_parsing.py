"""Reading an SBML model with the built-in parser.

SBML spells its element names inside a namespace that changes with the level, and
writes its list containers in different cases, so a parser that matches one
flavour exactly reports the others as empty models.
"""
import pytest
import xml.etree.ElementTree as ET

from biosuite.core import metabolism as mb


L3_V1 = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
 <model id="m" name="demo">
  <ListOfSpecies>
   <species id="A" name="A" constant="false" boundaryCondition="false" compartment="c"/>
   <species id="B" name="B" constant="false" boundaryCondition="false" compartment="c"/>
  </ListOfSpecies>
  <ListOfReactions>
   <reaction id="R1" reversible="false">
    <ListOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/></ListOfReactants>
    <ListOfProducts><speciesReference species="B" stoichiometry="2" constant="true"/></ListOfProducts>
   </reaction>
   <reaction id="R2" reversible="true">
    <ListOfReactants><speciesReference species="B" stoichiometry="1"/></ListOfReactants>
    <ListOfProducts><speciesReference species="A" stoichiometry="1"/></ListOfProducts>
   </reaction>
  </ListOfReactions>
 </model>
</sbml>
"""

L2_V4 = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
 <model id="m" name="demo">
  <listOfSpecies>
   <species id="A" name="A"/>
   <species id="B" name="B"/>
  </listOfSpecies>
  <listOfReactions>
   <reaction id="R1" reversible="false">
    <listOfReactants><speciesReference species="A" stoichiometry="1"/></listOfReactants>
    <listOfProducts><speciesReference species="B" stoichiometry="2"/></listOfProducts>
   </reaction>
   <reaction id="R2" reversible="true">
    <listOfReactants><speciesReference species="B" stoichiometry="1"/></listOfReactants>
    <listOfProducts><speciesReference species="A" stoichiometry="1"/></listOfProducts>
   </reaction>
  </listOfReactions>
 </model>
</sbml>
"""

NO_NAMESPACE = L3_V1.replace(' xmlns="http://www.sbml.org/sbml/level3/version1/core"', "")


def _model(tmp_path, text, name="model.xml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("flavour,document", [
    ("SBML L3 v1", L3_V1),
    ("SBML L2 v4", L2_V4),
    ("no namespace at all", NO_NAMESPACE),
])
def test_every_flavour_is_read_for_what_it_contains(tmp_path, flavour, document):
    """Measured before the fix: L2 and un-namespaced files parsed as 0 reactions,
    and even the L3 file produced empty substrate and product lists.

    The version-specific `findall` patterns matched the reaction but not the
    capitalisation of its containers, so a model counted as one reaction carried no
    stoichiometry at all.
    """
    reactions, metabolites, stoich = mb._parse_sbml_simple(_model(tmp_path, document))
    assert [r["id"] for r in reactions] == ["R1", "R2"], flavour
    assert [m["id"] for m in metabolites] == ["A", "B"], flavour
    assert stoich["R1"] == {"substrates": [("A", 1.0)], "products": [("B", 2.0)]}
    assert stoich["R2"] == {"substrates": [("B", 1.0)], "products": [("A", 1.0)]}


def test_the_reversible_flag_is_taken_from_the_attribute_it_is_written_in(tmp_path):
    reactions, _mets, _stoich = mb._parse_sbml_simple(_model(tmp_path, L3_V1))
    assert [(r["id"], r["reversible"]) for r in reactions] == [("R1", False), ("R2", True)]


def test_a_reaction_that_declares_no_name_is_named_by_its_id(tmp_path):
    document = L3_V1.replace('<reaction id="R1" reversible="false">',
                             '<reaction id="R1" reversible="false" name="named R1">')
    reactions, _mets, _stoich = mb._parse_sbml_simple(_model(tmp_path, document))
    assert [r["name"] for r in reactions] == ["named R1", "R2"]


def test_a_species_reference_without_a_stoichiometry_defaults_to_one(tmp_path):
    document = L3_V1.replace('<speciesReference species="A" stoichiometry="1" '
                             'constant="true"/>', '<speciesReference species="A"/>')
    _rxns, _mets, stoich = mb._parse_sbml_simple(_model(tmp_path, document))
    assert stoich["R1"]["substrates"] == [("A", 1.0)]


def test_a_reference_that_names_no_species_is_skipped_not_stored_as_none(tmp_path):
    document = L3_V1.replace('<ListOfProducts><speciesReference species="B" '
                             'stoichiometry="2" constant="true"/></ListOfProducts>',
                             '<ListOfProducts><speciesReference stoichiometry="2"/>'
                             '<speciesReference species="B" stoichiometry="2"/></ListOfProducts>')
    _rxns, _mets, stoich = mb._parse_sbml_simple(_model(tmp_path, document))
    assert stoich["R1"]["products"] == [("B", 2.0)]


def test_a_model_with_no_reactions_is_reported_as_the_empty_model_it_is(tmp_path):
    document = L3_V1.split("<ListOfReactions>")[0] + "</model></sbml>"
    reactions, metabolites, stoich = mb._parse_sbml_simple(_model(tmp_path, document))
    assert (reactions, stoich) == ([], {})
    assert [m["id"] for m in metabolites] == ["A", "B"]


def test_malformed_xml_is_not_reported_as_an_empty_model(tmp_path):
    """Measured before the fix: `except ...: pass` returned ([], [], {}).

    run_fba then said "Parsed 0 reactions, 0 metabolites" for a file that was
    truncated on download — a sentence about the model rather than about the read.
    """
    path = _model(tmp_path, L3_V1[:120], "broken.xml")
    with pytest.raises(ET.ParseError):
        mb._parse_sbml_simple(path)
    assert "is not well-formed XML" in mb.run_fba(path).message


def test_an_unreadable_path_is_named_with_the_error_the_system_gave(tmp_path):
    message = mb.run_fba(str(tmp_path / "absent.xml")).message
    assert "could not be read" in message and "absent.xml" in message


def test_a_directory_path_does_not_reach_the_caller_as_a_traceback(tmp_path):
    message = mb.run_fba(str(tmp_path)).message
    assert "could not be read" in message, message


def test_run_fba_on_a_real_model_says_what_it_did_and_did_not_do(tmp_path):
    """The built-in engine solves a matrix; a file is only parsed.

    Measured before the fix, the same call answered "Parsed 1 reactions, 2
    metabolites" with an objective of 0.0000 printed under it, which reads as a
    solved model that does not grow.
    """
    result = mb.run_fba(_model(tmp_path, L3_V1))
    assert result.reaction_count == 2 and result.metabolite_count == 2
    assert result.objective_value is None
    assert "the built-in engine does not solve a file" in result.message
    assert "pip install cobra" in result.message


def test_run_fba_on_an_empty_model_says_no_model_was_built(tmp_path):
    document = L3_V1.split("<ListOfReactions>")[0] + "</model></sbml>"
    result = mb.run_fba(_model(tmp_path, document))
    assert result.reaction_count == 0 and result.objective_value is None
    assert "No reactions were found" in result.message


def test_a_reaction_element_with_an_analyte_scaffold_is_read_for_its_species(tmp_path):
    """`_sbml_references` only takes speciesReference children, not any element."""
    document = L3_V1.replace(
        '<ListOfReactants><speciesReference species="A" stoichiometry="1" constant="true"/>'
        '</ListOfReactants>',
        '<ListOfReactants><modificationResidue id="mod1"/><speciesReference species="A" '
        'stoichiometry="1"/></ListOfReactants>')
    _rxns, _mets, stoich = mb._parse_sbml_simple(_model(tmp_path, document))
    assert stoich["R1"]["substrates"] == [("A", 1.0)]


def test_a_stoichiometry_that_is_not_a_number_skips_its_reference_with_a_warning(
        tmp_path, caplog):
    """Defaulting to 1.0 here would invent the one number the element states."""
    document = L3_V1.replace('<speciesReference species="B" stoichiometry="2" '
                             'constant="true"/>',
                             '<speciesReference species="B" stoichiometry="two"/>')
    with caplog.at_level("WARNING", logger="biosuite.core.metabolism"):
        _rxns, _mets, stoich = mb._parse_sbml_simple(_model(tmp_path, document))
    assert stoich["R1"]["products"] == []
    assert "carried a stoichiometry that is not a number" in caplog.text, caplog.text
    assert "R1/B='two'" in caplog.text, caplog.text


def test_a_reaction_element_with_no_containers_at_all_yields_empty_lists(tmp_path):
    document = L3_V1.replace('<ListOfReactants><speciesReference species="A" '
                             'stoichiometry="1" constant="true"/></ListOfReactants>',
                             '')
    _rxns, _mets, stoich = mb._parse_sbml_simple(_model(tmp_path, document))
    assert stoich["R1"] == {"substrates": [], "products": [("B", 2.0)]}


def test_the_engine_probe_reports_only_what_it_can_know():
    assert mb.check_metabolism_tools() == {"cobra": mb.HAS_COBRA}
    assert mb.HAS_COBRA is False, "cobra is absent in this environment"
