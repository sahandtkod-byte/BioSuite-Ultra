"""
GWAS analysis — genome-wide association studies.
Chi-squared test, Manhattan/QQ plots, lead SNP detection.
Pure Python/numpy/scipy implementation.
"""
import math

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from .expression import _benjamini_hochberg as _bh_shared
from .log import get_logger

logger = get_logger(__name__)

# Column schema of :func:`run_gwas` output.  Declared explicitly so a run that
# tests nothing still returns a typed frame: an empty frame with no columns
# made every consumer below die with ``KeyError: 'p_value'``.
GWAS_RESULT_COLUMNS = ("chrom", "pos", "snp_id", "case_alt", "case_ref",
                       "ctrl_alt", "ctrl_ref", "odds_ratio", "chi2_stat",
                       "p_value", "neg_log10")

# Columns a lead-SNP scan and the report read by name.  ``run_gwas`` produces
# all of them, but a hand-made or filtered frame reaches here from the CLI too.
LEAD_SNP_COLUMNS = ("chrom", "pos", "snp_id", "p_value")
REPORT_LEAD_COLUMNS = LEAD_SNP_COLUMNS + ("odds_ratio",)

# Report thresholds.  The label is kept next to the value it describes so the
# number the report prints cannot drift away from the one it compared.
GENOME_WIDE_THRESHOLD = 5e-8
GENOME_WIDE_LABEL = "5e-8"
SUGGESTIVE_THRESHOLD = 1e-5
SUGGESTIVE_LABEL = "1e-5"

# Floor used for -log10(p): scipy answers 0.0 once a p-value leaves the double
# range, so the plot position of every such SNP is this cap.
NEG_LOG10_FLOOR = 1e-300


def _finite_count(value, label):
    """A count that can go into a 2x2 table, or a ValueError naming the SNP.

    scipy answers ``All values in `observed` must be nonnegative`` for a negative
    count and says nothing at all about NaN, which a JSON body with an omitted
    field turns a count into.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number, got {value!r}.") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number, got {value!r}.")
    if number < 0:
        raise ValueError(f"{label} must be >= 0, got {value!r}.")
    return number


def _require_columns(frame, columns, name):
    """Raise when *frame* lacks one of *columns*, naming what it does carry."""
    present = set(frame.columns)
    missing = [column for column in columns if column not in present]
    if missing:
        raise ValueError(
            f"{name} needs column(s) {', '.join(missing)}; the frame carries "
            f"{', '.join(map(str, frame.columns)) or 'no columns at all'}.")
    return frame


def _integer_argument(value, name, minimum):
    """Validate a count argument against its inclusive lower bound."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


def _row_label(row, index):
    """How to name a SNP row in a message or an output column.

    ``str(row.get("snp_id") or ...)`` would call a legitimate id of ``0`` missing,
    and an id column that is absent or blank made every row share one name — which
    is what collapsed separate loci into a single lead SNP downstream.
    """
    value = row.get("snp_id", "")
    if isinstance(value, float) and math.isnan(value):
        value = ""
    if value is None or value == "":
        return f"row {index}"
    return str(value)


def gwas_chi_squared(controls_alt_count: int, cases_alt_count: int, controls_total: int, cases_total: int) -> dict:
    """Perform chi-squared test for one SNP.

    Args:
        controls_alt_count: number of alt alleles in controls.
        cases_alt_count: number of alt alleles in cases.
        controls_total: total allele count in controls.
        cases_total: total allele count in cases.

    Returns:
        dict with odds_ratio, p_value, chi2_stat.

    Raises:
        ValueError: when a count is not a finite non-negative number, when an
            allele count exceeds its own total, or when the table has no allele
            present in both arms (nothing can then be associated with anything).
    """
    controls_alt = _finite_count(controls_alt_count, "controls_alt_count")
    cases_alt = _finite_count(cases_alt_count, "cases_alt_count")
    controls_n = _finite_count(controls_total, "controls_total")
    cases_n = _finite_count(cases_total, "cases_total")
    if controls_alt > controls_n:
        raise ValueError(f"controls_alt_count ({controls_alt:g}) exceeds "
                         f"controls_total ({controls_n:g}).")
    if cases_alt > cases_n:
        raise ValueError(f"cases_alt_count ({cases_alt:g}) exceeds "
                         f"cases_total ({cases_n:g}).")

    ref_controls = controls_n - controls_alt
    ref_cases = cases_n - cases_alt
    if controls_n == 0 or cases_n == 0:
        raise ValueError(
            "chi-squared test is undefined: the "
            f"{'control' if controls_n == 0 else 'case'} arm has no alleles typed "
            f"(controls {controls_alt:g}/{controls_n:g}, "
            f"cases {cases_alt:g}/{cases_n:g}).")
    if (controls_alt == 0 and cases_alt == 0) or (ref_controls == 0 and ref_cases == 0):
        raise ValueError(
            f"chi-squared test is undefined: the alt allele is "
            f"{'absent' if controls_alt == 0 and cases_alt == 0 else 'fixed'} in "
            f"both arms (controls {controls_alt:g}/{controls_n:g}, "
            f"cases {cases_alt:g}/{cases_n:g}).")
    table = [[controls_alt, ref_controls],
             [cases_alt, ref_cases]]
    chi2_stat, p_val, dof, expected = chi2_contingency(table, correction=True)
    # OR = P(alt | case) / P(alt | ctrl) odds — the old formula was the
    # reciprocal, so genuinely protective/associated alleles were inverted.
    odds_ratio = _odds_ratio(controls_alt, ref_controls, cases_alt, ref_cases)
    return {
        "chi2_stat": chi2_stat,
        "p_value": p_val,
        "odds_ratio": odds_ratio,
    }


def _odds_ratio(ctrl_alt, ctrl_ref, case_alt, case_ref):
    """Odds ratio of the alt allele, 0-corrected when a cell is empty.

    ``max(denominator, 1)`` used to answer a zero denominator with the numerator
    itself, so a SNP carried by 50 of 100 cases and by none of 100 controls was
    reported as ``OR=5000.0`` — a number that grew with the sample size instead
    of describing the sample.  Adding half an observation to every cell
    (Haldane–Anscombe) keeps the estimate finite and shrinks it towards 1 as the
    evidence allows.  A non-degenerate table is untouched by it.
    """
    denominator = case_ref * ctrl_alt
    if denominator > 0:
        return (case_alt * ctrl_ref) / denominator
    return ((case_alt + 0.5) * (ctrl_ref + 0.5)) / ((case_ref + 0.5) * (ctrl_alt + 0.5))


def run_gwas(snp_data, case_col: str = "case", control_col: str = "control") -> pd.DataFrame:
    """Run GWAS on a DataFrame of SNP data.

    Rows that cannot be tested are dropped and counted in a warning, never
    answered with a NaN p-value: one monomorphic SNP used to abort the whole run
    with scipy's ``expected frequencies has a zero element`` message, which names
    neither the SNP nor the file.

    Args:
        snp_data: DataFrame with columns [chrom, pos, snp_id, case_alt, case_ref, ctrl_alt, ctrl_ref]
        case_col: prefix for case columns
        control_col: prefix for control columns

    Returns:
        DataFrame with GWAS results.

    Raises:
        ValueError: when the frame does not carry the columns an association test
            needs.  Silently skipping every row used to return an *empty frame with
            no columns at all*, so the next function in the chain died with
            ``KeyError: 'p_value'`` — far from the file or caller that was wrong.
    """
    columns = set(snp_data.columns)
    counts = [(case_col, "alt", "case_alt"), (case_col, "ref", "case_ref"),
              (control_col, "alt", "ctrl_alt"), (control_col, "ref", "ctrl_ref")]
    missing = []
    for prefix, suffix, fallback in counts:
        wanted = f"{prefix}_{suffix}"
        if wanted in columns or fallback in columns:
            continue
        missing.append(wanted if wanted == fallback else f"{wanted} (or {fallback})")
    missing += [label for label in ("chrom", "pos") if label not in columns]
    if missing:
        # The names quoted here must be the ones the help and the docstring
        # document, otherwise the user is told to invent a different schema.
        raise ValueError(
            f"SNP data is missing required column(s): {', '.join(missing)}. "
            f"Expected chrom, pos and the count columns documented in the help "
            f"({', '.join(fallback for _p, _s, fallback in counts)}).")

    keys = [(f"{prefix}_{suffix}", fallback) for prefix, suffix, fallback in counts]
    results = []
    unreadable = []
    monomorphic = 0
    untyped = 0
    for index, row in snp_data.iterrows():
        label = _row_label(row, index)
        try:
            case_alt, case_ref, ctrl_alt, ctrl_ref = [
                _finite_count(row.get(primary, row.get(fallback, 0)),
                              f"{label}: {primary}")
                for primary, fallback in keys]
        except ValueError as exc:
            unreadable.append(str(exc))
            continue

        case_total = case_alt + case_ref
        ctrl_total = ctrl_alt + ctrl_ref

        if case_total == 0 or ctrl_total == 0:
            untyped += 1
            continue
        if ((case_alt == 0 and ctrl_alt == 0)
                or (case_ref == 0 and ctrl_ref == 0)):
            monomorphic += 1
            continue

        res = gwas_chi_squared(ctrl_alt, case_alt, ctrl_total, case_total)
        results.append({
            "chrom": row.get("chrom", ""),
            "pos": row.get("pos", 0),
            "snp_id": label,
            "case_alt": case_alt,
            "case_ref": case_ref,
            "ctrl_alt": ctrl_alt,
            "ctrl_ref": ctrl_ref,
            "odds_ratio": res["odds_ratio"],
            "chi2_stat": res["chi2_stat"],
            "p_value": res["p_value"],
            "neg_log10": -np.log10(res["p_value"] + NEG_LOG10_FLOOR),
        })

    skipped = len(unreadable) + monomorphic + untyped
    if skipped:
        parts = []
        if unreadable:
            shown = "; ".join(unreadable[:5])
            more = f" (+{len(unreadable) - 5} more)" if len(unreadable) > 5 else ""
            parts.append(f"{len(unreadable)} row(s) with counts that are not "
                         f"usable numbers: {shown}{more}")
        if monomorphic:
            parts.append(f"{monomorphic} monomorphic row(s) whose allele is "
                         f"absent or fixed in both arms")
        if untyped:
            parts.append(f"{untyped} row(s) with no alleles typed in one arm")
        logger.warning("[gwas] tested %d of %d SNP row(s); skipped %s",
                       len(results), len(snp_data), "; ".join(parts))

    df = pd.DataFrame(results, columns=GWAS_RESULT_COLUMNS)
    if not df.empty:
        df = df.sort_values("p_value")
        df["p_adjusted"] = _benjamini_hochberg(df["p_value"].values)
    return df


def _benjamini_hochberg(p_values):
    """Alias of the single shared BH implementation in :mod:`biosuite.core.expression`.

    The local copy that used to live here handled NaN p-values incorrectly: NaN
    sorts last and then propagated backwards through the monotonicity pass,
    turning one missing p-value into a run of meaningless adjusted values.
    """
    return _bh_shared(p_values)


def detect_lead_snps(gwas_results, p_threshold: float = 5e-8, window_kb: int = 500) -> pd.DataFrame:
    """Detect lead (top) SNPs per locus.

    Args:
        gwas_results: DataFrame with chrom, pos, snp_id and p_value columns.
        p_threshold: genome-wide significance threshold.
        window_kb: window size (kilobases, >= 0) to define independent loci.

    Returns:
        DataFrame of lead SNPs, re-indexed 0..n-1 and carrying the input's
        columns — an empty run returns an empty frame *with those columns*,
        because a columnless empty frame made ``leads["snp_id"]`` die in the
        next step, far from the threshold that found nothing.

    Raises:
        ValueError: when a column the scan reads by name is absent, or when
            *window_kb* is not a non-negative number.
    """
    _require_columns(gwas_results, LEAD_SNP_COLUMNS, "detect_lead_snps")
    try:
        window = float(window_kb)
    except (TypeError, ValueError):
        raise ValueError(f"window_kb must be a number of kilobases >= 0, "
                         f"got {window_kb!r}.") from None
    if not math.isfinite(window) or window < 0:
        raise ValueError(f"window_kb must be a finite number >= 0, got {window_kb!r}.")
    window *= 1000

    empty = gwas_results.iloc[0:0].copy()
    sig = gwas_results[gwas_results["p_value"] < p_threshold]
    if sig.empty:
        return empty

    lead_snps = []
    for _chrom, group in sig.groupby("chrom"):
        # Most significant SNP claims the locus — sorting by position made
        # the LEFTMOST SNP the lead even when a neighbour was stronger.
        # Re-indexed first, because a locus is claimed by a *row*, not by a
        # label its input frame happened to carry: identical snp_id values
        # (empty, repeated, or copied from a merged file) used to retire
        # every locus that shared one, leaving one lead SNP per chromosome.
        group = group.sort_values("p_value").reset_index(drop=True)
        used = set()
        for label, row in group.iterrows():
            if label in used:
                continue
            lead_snps.append(row)
            nearby = group[(group["pos"] >= row["pos"] - window) &
                           (group["pos"] <= row["pos"] + window)]
            used.update(nearby.index.tolist())

    return pd.DataFrame(lead_snps).reset_index(drop=True)


def generate_gwas_data(n_snps=5000, n_chromosomes=22, seed=42):
    """Generate simulated GWAS data for testing/demo.

    Returns:
        DataFrame with chrom, pos, snp_id, case_alt, case_ref, ctrl_alt, ctrl_ref,
        holding exactly ``n_snps`` rows: the count is spread with ``divmod`` and the
        remainder goes to the first chromosomes.  ``n_snps // n_chromosomes`` per
        chromosome truncated it silently instead — asked for 250, delivered 242 —
        so no caller could rely on the number it passed in.

    Raises:
        ValueError: when *n_snps* is negative or *n_chromosomes* is not at least
            one.  ``divmod(n_snps, 0)`` used to be a ZeroDivisionError, and a
            negative *n_snps* answered an empty frame instead of the count asked
            for.
    """
    n_snps = _integer_argument(n_snps, "n_snps", 0)
    n_chromosomes = _integer_argument(n_chromosomes, "n_chromosomes", 1)
    rng = np.random.default_rng(seed)  # local: no global RNG side effects
    data = []
    snp_idx = 0
    per_chrom, remainder = divmod(n_snps, n_chromosomes)
    for chrom in range(1, n_chromosomes + 1):
        n_snps_chr = per_chrom + (1 if chrom <= remainder else 0)
        for _i in range(n_snps_chr):
            snp_idx += 1
            pos = rng.integers(1, 250_000_000)
            ctrl_alt = rng.integers(5, 200)
            ctrl_ref = rng.integers(5, 200)
            # Add some signals
            if chrom == 6 and 25_000_000 < pos < 35_000_000:
                case_alt = ctrl_alt + rng.integers(20, 60)
            else:
                case_alt = ctrl_alt + rng.integers(-10, 10)
            case_ref = max(5, ctrl_ref + rng.integers(-10, 10))
            data.append({
                "chrom": f"chr{chrom}", "pos": pos,
                "snp_id": f"rs{snp_idx}",
                "case_alt": max(0, case_alt), "case_ref": max(0, case_ref),
                "ctrl_alt": ctrl_alt, "ctrl_ref": ctrl_ref,
            })
    return pd.DataFrame(data)


def format_gwas_report(gwas_results, lead_snps=None):
    """Format GWAS results as text report.

    ``lead_snps=None`` means the lead-SNP step never ran, which is not the same
    statement as "it ran and found nothing".  The old code printed ``No
    significant loci detected.`` for both, so omitting the argument turned a
    frame carrying two SNPs below 5e-8 into a study with no loci at all.
    """
    _require_columns(gwas_results, ("p_value",), "format_gwas_report")
    n_total = len(gwas_results)
    n_sig = int((gwas_results["p_value"] < GENOME_WIDE_THRESHOLD).sum())
    n_suggestive = int((gwas_results["p_value"] < SUGGESTIVE_THRESHOLD).sum())
    lines = [
        "GWAS Analysis Report",
        "=" * 40,
        f"Total SNPs tested: {n_total}",
        f"Genome-wide significant (p < {GENOME_WIDE_LABEL}): {n_sig}",
        f"Suggestive (p < {SUGGESTIVE_LABEL}): {n_suggestive}",
        "",
    ]
    if lead_snps is None:
        lines.append("Lead SNP detection was not run, so no loci are listed.")
        return "\n".join(lines)

    if not lead_snps.empty:
        _require_columns(lead_snps, REPORT_LEAD_COLUMNS, "format_gwas_report")
    lines.append("Lead SNPs:" if not lead_snps.empty else "No significant loci detected.")
    for _, row in lead_snps.iterrows():
        lines.append(f"  {row['snp_id']} ({row['chrom']}:{row['pos']}) "
                     f"p={row['p_value']:.2e} OR={row['odds_ratio']:.2f}")
    return "\n".join(lines)

