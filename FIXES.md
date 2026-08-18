# BioSuite Ultra — Fix Tracker

> Generated from audit FINDINGS.md | Last updated: 2026-08-18
> **Rule:** Each fix = 1 PR + new test. Test first (red), fix second (green).

---

## 🔴 CRITICAL — Phase 1 (This Week)

| ID | Bug | File | Est. | Status |
|----|-----|------|------|--------|
| T3-C3 | `eval()` on CLI user input → RCE | `cli/menu.py:916,927` | 30m | ⬜ Open |
| T3-C1 | Default admin creds + forgeable JWT | `api/admin.py` | 2h | ⬜ Open |
| T3-C2 | Arbitrary file read as counting oracle | `api/admin.py` | 1h | ⬜ Open |
| T4-C1 | `needleman_wunsch()` wrong scores 97% | `core/alignment.py` | 3h | ⬜ Open |
| T5-C1 | GWAS odds ratios ALL inverted | `core/gwas.py` | 1h | ⬜ Open |
| T5-C2 | Ti/Tv ratio returns 0 for all-transition | `core/variant_calling.py` | 1h | ⬜ Open |
| T8-C1 | Log-rank test ALWAYS p=1.0 | `core/survival.py` | 2h | ⬜ Open |
| T8-C2 | Tajima's D systematically biased negative | `core/popgen.py` | 2h | ⬜ Open |
| T2-C1 | Phylogeny tree heights all wrong | `core/phylogeny.py` | 3h | ⬜ Open |
| T2-C2 | Corrupted benchmark test file | `benchmarks/test_core_benchmarks.py` | 30m | ⬜ Open |

**Phase 1 total: ~16h estimated**

---

## 🟠 HIGH — Phase 2 (Week 2-3)

| ID | Bug | File | Est. | Status |
|----|-----|------|------|--------|
| T5-H1 | Peak caller drops peaks at chr ends | `core/peak_calling.py` | 2h | ⬜ Open |
| T5-H2 | Peak caller fails on sparse ChIP-seq | `core/peak_calling.py` | 2h | ⬜ Open |
| T6-H1 | `read_counts_matrix` destroys sample names | `core/expression.py` | 1h | ⬜ Open |
| T7-H1 | ORF finder `include_start=False` ignored | `core/orf_finder.py` | 1h | ⬜ Open |
| T7-H2 | Trimmer avg_quality_after wrong denominator | `core/trimming.py` | 1h | ⬜ Open |
| Re-H1 | MSA star alignment cascades NW errors | `core/msa.py` | 2h | ⬜ Open |
| Re-H2 | Variant caller missing filter flags | `core/variant_calling.py` | 1h | ⬜ Open |
| Re-H3 | Expression DESeq2 wrapper edge case | `core/expression.py` | 1h | ⬜ Open |
| T1-H1 | Release archives committed to git | repo root | 30m | ⬜ Open |
| T1-H2 | Stray .bak and test PNGs in tree | repo root | 30m | ⬜ Open |
| T1-H3 | main/master branch divergence | git | 1h | ⬜ Open |
| T3-H1 | API rate limiting bypass possible | `api/` | 2h | ⬜ Open |
| T3-H2 | CORS misconfiguration | `api/` | 1h | ⬜ Open |

**Phase 2 total: ~16h estimated**

---

## 🟡 MEDIUM — Phase 3 (Week 4+)

| ID | Bug | File | Est. | Status |
|----|-----|------|------|--------|
| T11-M1 | FBA minimizes instead of maximizes | `core/metabolism.py` | 30m | ⬜ Open |
| T10-M1 | Epigenomics bisulfite conversion bias | `core/epigenomics.py` | 1h | ⬜ Open |
| *(18 more from FINDINGS.md)* | | | | ⬜ Open |

**Phase 3 total: ~20 items**

---

## 🔵 LOW — Phase 4 (Backlog)

| ID | Bug | File | Est. | Status |
|----|-----|------|------|--------|
| T12-L1 | `tempfile.mktemp()` race condition ×3 | `read_aligner.py`, `docking.py` | 30m | ⬜ Open |
| T13-L1 | Bare `except:` in GUI cleanup | `gui/main_window.py:300` | 15m | ⬜ Open |
| T1-L1..L6 | Repo hygiene (tags, topics, ORCID) | repo config | 2h | ⬜ Open |
| T2-L1..L4 | Code quality notes | various | 1h | ⬜ Open |
| T3-L1..L6 | Security hardening notes | various | 2h | ⬜ Open |

**Phase 4 total: ~18 items**

---

## ✅ DONE

*(Move items here after fix + test pass + PR merged)*

---

## 📊 Progress

| Phase | Total | Done | Remaining |
|-------|-------|------|-----------|
| 🔴 Critical | 10 | 0 | 10 |
| 🟠 High | 13 | 0 | 13 |
| 🟡 Medium | 20 | 0 | 20 |
| 🔵 Low | 18 | 0 | 18 |
| **ALL** | **61** | **0** | **61** |
