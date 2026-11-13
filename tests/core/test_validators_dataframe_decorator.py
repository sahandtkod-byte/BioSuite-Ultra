"""``@validate_dataframe_columns`` — the frame it inspects and the columns it needs.

The measured failures this file replaces: a second positional DataFrame was never
checked once the first matched, a keyword that held a list instead of a frame was
skipped silently, and `required_cols='gene'` — one string, not a list — was read
one character at a time, so a frame with columns 'g', 'e', 'n' passed.  Which
parameter the decorator picks is the subject of
`test_validators_argument_targeting.py`.
"""
import pandas as pd
import pytest

from biosuite.core import validators as v


@pytest.fixture
def gene_and_expr():
    """``@validate_dataframe_columns(['gene', 'expr'])`` over a shape reader."""
    @v.validate_dataframe_columns(['gene', 'expr'])
    def summarize(df):
        return df.shape

    return summarize


def test_a_frame_with_every_required_column_is_handed_on(gene_and_expr):
    frame = pd.DataFrame({'gene': ['A', 'B'], 'expr': [1.0, 2.0]})
    assert gene_and_expr(frame) == (2, 2)


def test_extra_columns_are_nobody_business(gene_and_expr):
    frame = pd.DataFrame({'expr': [1], 'gene': ['A'], 'batch': ['b1']})
    assert gene_and_expr(frame) == (1, 3)


def test_every_missing_column_is_reported_at_once(gene_and_expr):
    with pytest.raises(ValueError) as excinfo:
        gene_and_expr(pd.DataFrame({'gene': [1], 'other': [2]}))
    assert str(excinfo.value) == (
        "DataFrame in argument 'df' of summarize() is missing required columns: ['expr']. "
        "Available columns: ['gene', 'other']"
    )


def test_the_default_requirement_is_a_gene_column():
    @v.validate_dataframe_columns()
    def summarize(df):
        return df.shape

    assert summarize(pd.DataFrame({'gene': ['A', 'B']})) == (2, 1)
    with pytest.raises(ValueError) as excinfo:
        summarize(pd.DataFrame({'symbol': ['A']}))
    assert str(excinfo.value) == (
        "DataFrame in argument 'df' of summarize() is missing required columns: ['gene']. "
        "Available columns: ['symbol']"
    )


@pytest.mark.parametrize("value, message", [
    pytest.param([1, 2], "argument 'df' of summarize() must be a pandas DataFrame, got list.",
                 id="list"),
    pytest.param({'gene': ['A']},
                 "argument 'df' of summarize() must be a pandas DataFrame, got dict.",
                 id="dict"),
    pytest.param(None,
                 "argument 'df' of summarize() must be a pandas DataFrame, got NoneType.",
                 id="none"),
    pytest.param('gene',
                 "argument 'df' of summarize() must be a pandas DataFrame, got str.",
                 id="string"),
])
def test_a_value_that_is_not_a_frame_is_refused(gene_and_expr, value, message):
    """Measured on the old code: a non-frame was skipped as "not a DataFrame", so
    the function received a list and raised its own AttributeError several frames
    deeper — and `summarize(df=[1, 2])` looked like a passing validation."""
    with pytest.raises(ValueError) as excinfo:
        gene_and_expr(value)
    assert str(excinfo.value) == message


@pytest.mark.parametrize("required", [
    pytest.param(['gene'], id="list"),
    pytest.param(('gene',), id="tuple"),
    pytest.param({'gene'}, id="set"),
    pytest.param(frozenset({'gene'}), id="frozenset"),
])
def test_any_collection_of_names_is_a_valid_requirement(required):
    @v.validate_dataframe_columns(required)
    def summarize(df):
        return df.shape

    assert summarize(pd.DataFrame({'gene': [1]})) == (1, 1)
    with pytest.raises(ValueError):
        summarize(pd.DataFrame({'other': [1]}))


@pytest.mark.parametrize("required_cols, error, message", [
    pytest.param([], ValueError,
                 "validate_dataframe_columns required_cols must list at least one column name.",
                 id="empty"),
    pytest.param('gene', TypeError,
                 "validate_dataframe_columns required_cols must be a list of column names, "
                 "got the single string 'gene'.", id="bare-string"),
    pytest.param([''], ValueError,
                 "validate_dataframe_columns required_cols must contain non-empty strings, "
                 "got ''.", id="blank-entry"),
    pytest.param([1], ValueError,
                 "validate_dataframe_columns required_cols must contain non-empty strings, "
                 "got 1.", id="number-entry"),
    pytest.param(42, TypeError,
                 "validate_dataframe_columns required_cols must be a list of column names, "
                 "got int.", id="number"),
])
def test_the_column_list_is_checked_when_the_decorator_is_built(required_cols, error, message):
    """`required_cols='gene'` used to become `set('gene')` — the four letters
    g, e, n, e — which a frame with those single-letter columns would satisfy while
    the gene column the caller meant stayed unchecked."""
    with pytest.raises(error) as excinfo:
        v.validate_dataframe_columns(required_cols)
    assert str(excinfo.value) == message


def test_duplicated_names_in_the_requirement_are_checked_once():
    @v.validate_dataframe_columns(['gene', 'gene'])
    def summarize(df):
        return df.shape

    assert summarize(pd.DataFrame({'gene': [1]})) == (1, 1)
    with pytest.raises(ValueError) as excinfo:
        summarize(pd.DataFrame({'other': [1]}))
    assert str(excinfo.value) == (
        "DataFrame in argument 'df' of summarize() is missing required columns: ['gene']. "
        "Available columns: ['other']"
    )
