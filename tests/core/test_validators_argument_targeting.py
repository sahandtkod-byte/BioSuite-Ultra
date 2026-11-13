"""Which argument a ``core/validators.py`` decorator inspects, and when it refuses
to decorate at all.

These four decorators used to scan the call for "the first argument of a matching
type", which produced three wrong answers at once: a function taking a label and a
sequence had its **label** validated, a second path or DataFrame beside the first
was never looked at, and a call carrying no matching value was validated by nobody.
The parameter is now resolved once when the decorator is applied, in a fixed order
— documented name, then type annotation, then the first ordinary parameter — and a
function with nothing to inspect is refused while it is being decorated.

The exact rejection messages of each decorator live in that decorator's own file
(`test_validators_sequence_decorator.py`, `test_validators_range_decorator.py`,
`test_validators_file_extension_decorator.py`,
`test_validators_dataframe_decorator.py`).
"""
import inspect

import pandas as pd
import pytest

from biosuite.core import validators as v


# ── the resolution order ────────────────────────────────────────────────────

def test_a_documented_parameter_name_wins_over_the_one_that_happens_to_be_first():
    """The label is not a sequence, so it must not be the argument under test."""
    @v.validate_sequence(min_length=2, allowed_chars='ATCG')
    def align(name, seq):
        return (name, seq)

    # 'ATCGNN' is a fine label and a bad sequence; the old scan refused the label.
    assert align('ATCGNN', 'ACGT') == ('ATCGNN', 'ACGT')
    with pytest.raises(ValueError) as excinfo:
        align('ATCGNN', 'XX')
    assert str(excinfo.value) == (
        "Sequence in argument 'seq' of align() contains invalid characters: ['X']. "
        "Allowed characters: ['A', 'C', 'G', 'T']"
    )


def test_the_type_annotation_selects_the_parameter_when_no_name_matches():
    @v.validate_sequence(min_length=2)
    def transcribe(primer: str, cycles: int):
        return (primer, cycles)

    assert transcribe('ACGT', 3) == ('ACGT', 3)
    with pytest.raises(ValueError) as excinfo:
        transcribe('AX', 3)
    assert "argument 'primer' of transcribe()" in str(excinfo.value)


def test_the_fallback_target_is_named_in_the_refusal_so_a_wrong_one_is_obvious():
    """With no documented name and no annotation the first ordinary parameter is
    used — which is why naming the parameter is the caller's job."""
    @v.validate_range(min_val=0.0, max_val=1.0)
    def plot(label, fraction):
        return (label, fraction)

    with pytest.raises(ValueError) as excinfo:
        plot('qc', 0.5)
    assert str(excinfo.value) == "argument 'label' of plot() must be a number, got str."

    @v.validate_range(min_val=0.0, max_val=1.0)
    def plotted(label, value):
        return (label, value)

    assert plotted('qc', 0.5) == ('qc', 0.5)


def test_self_and_cls_are_never_the_target():
    class Analyser:
        @v.validate_range(min_val=0.0, max_val=1.0)
        def set_cutoff(self, fraction):
            return (self, fraction)

    host = Analyser()
    assert host.set_cutoff(0.5)[1] == 0.5
    with pytest.raises(ValueError) as excinfo:
        host.set_cutoff(2.0)
    assert str(excinfo.value) == (
        "Value in argument 'fraction' of set_cutoff() (2.0) is above maximum (1.0)"
    )


@pytest.mark.parametrize("builder, signature", [
    pytest.param(lambda: v.validate_sequence()(lambda *seqs: seqs), '(*seqs)', id="var-positional"),
    pytest.param(lambda: v.validate_sequence()(lambda **seqs: seqs), '(**seqs)',
                 id="var-keyword"),
    pytest.param(lambda: v.validate_sequence()(lambda: 1), '()', id="no-parameters"),
])
def test_a_function_with_nothing_to_inspect_is_refused_while_it_is_decorated(builder,
                                                                            signature):
    """Silently validating nothing is worse than refusing the decorator."""
    with pytest.raises(TypeError) as excinfo:
        builder()
    assert str(excinfo.value).startswith(
        "validate_sequence cannot decorate <lambda>(): its signature ")
    assert signature in str(excinfo.value)
    assert str(excinfo.value).endswith(" has no ordinary parameter to inspect.")


def test_a_callable_without_a_signature_is_refused_where_it_is_decorated():
    """`int` and `type` are callables that carry no signature at all, so there is
    no parameter to resolve and the decorator says so while it is applied."""
    with pytest.raises(TypeError) as excinfo:
        v.validate_sequence()(int)
    assert str(excinfo.value) == (
        "validate_sequence cannot read the signature of <class 'int'>: "
        "no signature found for builtin type <class 'int'>"
    )
    with pytest.raises(TypeError) as excinfo:
        v.validate_range(min_val=0)(type)
    assert str(excinfo.value) == (
        "validate_range cannot read the signature of <class 'type'>: "
        "no signature found for builtin <class 'type'>"
    )


# ── what the resolved target accepts ────────────────────────────────────────

def test_a_value_of_the_wrong_kind_is_refused_instead_of_being_handed_on():
    """The old code passed ``f(123)`` through and let the wrapped function crash."""
    @v.validate_sequence(min_length=2)
    def reverse(seq):
        return seq[::-1]

    with pytest.raises(ValueError) as excinfo:
        reverse(123)
    assert str(excinfo.value) == "argument 'seq' of reverse() must be a string, got int."
    with pytest.raises(ValueError) as excinfo:
        reverse(None)
    assert str(excinfo.value) == "argument 'seq' of reverse() must be a string, got NoneType."


@pytest.mark.parametrize(
    "call, expected",
    [
        (lambda f: f('ACGT', 'extra'), "reverse_complement() takes 1 positional "
                                       "argument but 2 were given"),
        (lambda f: f(), "reverse_complement() missing 1 required positional argument: 'seq'"),
    ],
    ids=["too-many", "missing"])
def test_a_call_that_does_not_fit_the_signature_is_left_to_python(call, expected):
    """A validator has nothing useful to add to Python's own argument-count error."""
    @v.validate_sequence(min_length=2)
    def reverse_complement(seq):
        return 'ok'

    assert reverse_complement('AC') == 'ok'
    with pytest.raises(TypeError) as excinfo:
        call(reverse_complement)
    assert str(excinfo.value).endswith(expected)


def test_an_argument_the_caller_left_out_is_nothing_to_validate_but_None_is():
    """`seq=None` as a default means "this call brought no sequence"; passing None
    explicitly is a value of the wrong type and has to be refused."""
    @v.validate_sequence(min_length=2)
    def write(seq=None):
        return seq

    assert write() is None
    with pytest.raises(ValueError) as excinfo:
        write(None)
    assert str(excinfo.value) == "argument 'seq' of write() must be a string, got NoneType."


def test_every_named_parameter_of_a_call_is_checked_not_just_the_first():
    @v.validate_sequence(min_length=2)
    def merge(first_seq=None, seq=None, sequence=None):
        return (first_seq, seq, sequence)

    # 'first_seq' is not in the vocabulary, so 'seq' and 'sequence' are the targets.
    with pytest.raises(ValueError) as excinfo:
        merge(first_seq='A', seq='AC', sequence='T')
    assert str(excinfo.value) == (
        "Sequence in argument 'sequence' of merge() is too short: 1 < 2 (min_length=2)"
    )
    with pytest.raises(ValueError) as excinfo:
        merge(seq='A', sequence='AC')
    assert str(excinfo.value) == (
        "Sequence in argument 'seq' of merge() is too short: 1 < 2 (min_length=2)"
    )
    assert merge(first_seq='A', seq='AC', sequence='ACGT') == ('A', 'AC', 'ACGT')


def test_a_default_is_validated_too_because_it_reaches_the_function():
    @v.validate_sequence(min_length=3)
    def read(seq='AC'):
        return seq

    with pytest.raises(ValueError) as excinfo:
        read()
    assert str(excinfo.value) == (
        "Sequence in argument 'seq' of read() is too short: 2 < 3 (min_length=3)"
    )
    assert read('ACGTA') == 'ACGTA'


def test_a_keyword_only_parameter_is_found_and_its_default_is_checked():
    @v.validate_sequence(min_length=2)
    def write(*, seq='AC'):
        return seq

    assert write() == 'AC'
    assert write(seq='ACGT') == 'ACGT'
    with pytest.raises(ValueError) as excinfo:
        write(seq='A')
    assert str(excinfo.value) == (
        "Sequence in argument 'seq' of write() is too short: 1 < 2 (min_length=2)"
    )


def test_every_decorator_targets_its_own_vocabulary_of_names():
    """The five names each decorator recognises are module constants, so a caller
    can read them; the decorators must agree with their own lists."""
    assert v.SEQUENCE_ARGUMENT_NAMES == ('sequence', 'seq', 'dna', 'rna', 'sequence_str')
    assert v.PATH_ARGUMENT_NAMES == ('filepath', 'path', 'file', 'filename', 'input_file')
    assert v.DATAFRAME_ARGUMENT_NAMES == ('df', 'dataframe', 'data', 'table')
    assert v.NUMBER_ARGUMENT_NAMES == ('value', 'threshold', 'ratio', 'score', 'pvalue',
                                       'cutoff')


@pytest.mark.parametrize("key", v.SEQUENCE_ARGUMENT_NAMES)
def test_the_sequence_vocabulary_names_are_all_recognised(key):
    @v.validate_sequence(min_length=2)
    def run(sequence=None, seq=None, dna=None, rna=None, sequence_str=None):
        return (sequence, seq, dna, rna, sequence_str)

    expected = tuple('AC' if name == key else None for name in v.SEQUENCE_ARGUMENT_NAMES)
    assert run(**{key: 'AC'}) == expected
    with pytest.raises(ValueError) as excinfo:
        run(**{key: 'A'})
    assert f"argument '{key}' of run()" in str(excinfo.value)


@pytest.mark.parametrize("key", v.PATH_ARGUMENT_NAMES)
def test_the_path_vocabulary_names_are_all_recognised(key):
    @v.validate_file_extension(['.fa'])
    def run(filepath=None, path=None, file=None, filename=None, input_file=None):
        return (filepath, path, file, filename, input_file)

    expected = tuple('reads.fa' if name == key else None
                     for name in v.PATH_ARGUMENT_NAMES)
    assert run(**{key: 'reads.fa'}) == expected
    with pytest.raises(ValueError) as excinfo:
        run(**{key: 'reads.txt'})
    assert f"argument '{key}' of run()" in str(excinfo.value)


@pytest.mark.parametrize("key", v.DATAFRAME_ARGUMENT_NAMES)
def test_the_dataframe_vocabulary_names_are_all_recognised(key):
    @v.validate_dataframe_columns(['gene'])
    def run(df=None, dataframe=None, data=None, table=None):
        return (df, dataframe, data, table)

    good = pd.DataFrame({'gene': ['A']})
    assert run(**{key: good})[v.DATAFRAME_ARGUMENT_NAMES.index(key)] is good
    with pytest.raises(ValueError) as excinfo:
        run(**{key: pd.DataFrame({'other': [1]})})
    assert f"DataFrame in argument '{key}' of run()" in str(excinfo.value)


@pytest.mark.parametrize("key", v.NUMBER_ARGUMENT_NAMES)
def test_the_number_vocabulary_names_are_all_recognised(key):
    @v.validate_range(min_val=0.0, max_val=1.0)
    def run(value=None, threshold=None, ratio=None, score=None, pvalue=None, cutoff=None):
        return (value, threshold, ratio, score, pvalue, cutoff)

    assert run(**{key: 0.5})[v.NUMBER_ARGUMENT_NAMES.index(key)] == 0.5
    with pytest.raises(ValueError) as excinfo:
        run(**{key: 2.0})
    assert f"Value in argument '{key}' of run()" in str(excinfo.value)


# ── the decorators keep the function they wrap ──────────────────────────────

def test_a_decorated_function_keeps_its_name_signature_and_docstring():
    @v.validate_sequence(min_length=2)
    def translate(seq):
        """Turn a message into protein."""
        return len(seq)

    assert (translate.__name__, translate.__doc__) == ('translate', 'Turn a message into protein.')
    assert list(inspect.signature(translate).parameters) == ['seq']
    assert translate('ACGT') == 4
