# BioSuite-Ultra — Review Tracker

> آخرین به‌روزرسانی: اسپرینت دوم (۲۰۲۶-۰۹-۰۴). این فایل یک بار به‌اشتباه از درخت گیت حذف شده بود و الان دوباره بازسازی شده است.
> وضعیت کدبیس: شاخه‌ی `arena/improvements`، ۸۵ کامیت جلوتر از upstream main (bd670f0).

## روند کلی
- Sprint I: مرور خط‌به‌خط **۸۷/۸۷ فایل** پایتون، رفع باگ‌های اسپرینت اول (بلست هابکتها، MSA consensus، genome_browser و غیره).
- Sprint II: بازبینی عمیق ماژول‌های هسته + افزودن ۱۴ فایل تست جدید. اصلاحات تازه: menu.py (UnboundLocalError)، interactive_plots.py (numpy truthiness)، md_simulation.py (عناصر دوبخشی).

## پوشش تست (coverage branch-mode)
- کل: **62%** — 16067 دستور، 5927 پوشش‌نداده، 4814 شاخه.
- توزیع ۷۸ ماژول: ≥۹۵٪ ‏= ۸ | ۸۰–۹۵٪ ‏= ۲۵ | ۷۰–۸۰٪ ‏= ۱۶ | <۷۰٪ ‏= ۲۹ (بیشتر GUI/وب‌API).

## وضعیت ماژول‌به‌ماژول
| فایل | مرور | پوشش | وضع |
|------|------|-------|------|
| biosuite/__init__.py | REVIEWED (line-by-line, Sprint I) | n/a | 🔴 |
| biosuite/api/__init__.py | REVIEWED (line-by-line, Sprint I) | n/a | 🔴 |
| biosuite/api/auth.py | REVIEWED (line-by-line, Sprint I) | 94% | 🟢 |
| biosuite/api/security.py | REVIEWED (line-by-line, Sprint I) | 97% | 🟢 |
| biosuite/api/server.py | REVIEWED (line-by-line, Sprint I) | 0% | 🔴 |
| biosuite/cli/__init__.py | REVIEWED (line-by-line, Sprint I) | n/a | 🔴 |
| biosuite/cli/menu.py | REVIEWED (line-by-line, Sprint I) | 35% | 🔴 |
| biosuite/core/__init__.py | REVIEWED (line-by-line, Sprint I) | n/a | 🔴 |
| biosuite/core/alignment.py | REVIEWED (line-by-line, Sprint I) | 94% | 🟢 |
| biosuite/core/assembly.py | REVIEWED (line-by-line, Sprint I) | 81% | 🟡 |
| biosuite/core/bayesian_phylogeny.py | REVIEWED (line-by-line, Sprint I) | 83% | 🟡 |
| biosuite/core/bio_ml.py | REVIEWED (line-by-line, Sprint I) | 82% | 🟡 |
| biosuite/core/blast.py | REVIEWED (line-by-line, Sprint I) | 70% | 🟡 |
| biosuite/core/cloning.py | REVIEWED (line-by-line, Sprint I) | 72% | 🟡 |
| biosuite/core/codon_usage.py | REVIEWED (line-by-line, Sprint I) | 97% | 🟢 |
| biosuite/core/crispr.py | REVIEWED (line-by-line, Sprint I) | 79% | 🟡 |
| biosuite/core/databases.py | REVIEWED (line-by-line, Sprint I) | 75% | 🟡 |
| biosuite/core/docking.py | REVIEWED (line-by-line, Sprint I) | 87% | 🟡 |
| biosuite/core/enrichment.py | REVIEWED (line-by-line, Sprint I) | 65% | 🔴 |
| biosuite/core/epigenomics.py | REVIEWED (line-by-line, Sprint I) | 93% | 🟢 |
| biosuite/core/epitope.py | REVIEWED (line-by-line, Sprint I) | 99% | 🟢 |
| biosuite/core/expression.py | REVIEWED (line-by-line, Sprint I) | 74% | 🟡 |
| biosuite/core/file_formats.py | REVIEWED (line-by-line, Sprint I) | 75% | 🟡 |
| biosuite/core/go_browser.py | REVIEWED (line-by-line, Sprint I) | 93% | 🟢 |
| biosuite/core/gwas.py | REVIEWED (line-by-line, Sprint I) | 97% | 🟢 |
| biosuite/core/log.py | REVIEWED (line-by-line, Sprint I) | 92% | 🟢 |
| biosuite/core/md_simulation.py | REVIEWED (line-by-line, Sprint I) | 83% | 🟡 |
| biosuite/core/metabolism.py | REVIEWED (line-by-line, Sprint I) | 75% | 🟡 |
| biosuite/core/metabolomics.py | REVIEWED (line-by-line, Sprint I) | 96% | 🟢 |
| biosuite/core/metagenomics.py | REVIEWED (line-by-line, Sprint I) | 70% | 🟡 |
| biosuite/core/ml_phylogeny.py | REVIEWED (line-by-line, Sprint I) | 70% | 🟡 |
| biosuite/core/msa.py | REVIEWED (line-by-line, Sprint I) | 68% | 🔴 |
| biosuite/core/ngs.py | REVIEWED (line-by-line, Sprint I) | 67% | 🔴 |
| biosuite/core/orf_finder.py | REVIEWED (line-by-line, Sprint I) | 82% | 🟡 |
| biosuite/core/parallel.py | REVIEWED (line-by-line, Sprint I) | 68% | 🔴 |
| biosuite/core/pathway_viz.py | REVIEWED (line-by-line, Sprint I) | 95% | 🟢 |
| biosuite/core/peak_calling.py | REVIEWED (line-by-line, Sprint I) | 84% | 🟡 |
| biosuite/core/phylogeny.py | REVIEWED (line-by-line, Sprint I) | 84% | 🟡 |
| biosuite/core/plugin.py | REVIEWED (line-by-line, Sprint I) | 70% | 🟡 |
| biosuite/core/popgen.py | REVIEWED (line-by-line, Sprint I) | 88% | 🟡 |
| biosuite/core/provenance.py | REVIEWED (line-by-line, Sprint I) | 89% | 🟡 |
| biosuite/core/quantification.py | REVIEWED (line-by-line, Sprint I) | 62% | 🔴 |
| biosuite/core/read_aligner.py | REVIEWED (line-by-line, Sprint I) | 80% | 🟡 |
| biosuite/core/sequence.py | REVIEWED (line-by-line, Sprint I) | 74% | 🟡 |
| biosuite/core/single_cell.py | REVIEWED (line-by-line, Sprint I) | 34% | 🔴 |
| biosuite/core/structure.py | REVIEWED (line-by-line, Sprint I) | 61% | 🔴 |
| biosuite/core/structure_prediction.py | REVIEWED (line-by-line, Sprint I) | 69% | 🔴 |
| biosuite/core/survival.py | REVIEWED (line-by-line, Sprint I) | 92% | 🟢 |
| biosuite/core/trimming.py | REVIEWED (line-by-line, Sprint I) | 76% | 🟡 |
| biosuite/core/utils.py | REVIEWED (line-by-line, Sprint I) | 69% | 🔴 |
| biosuite/core/validators.py | REVIEWED (line-by-line, Sprint I) | 77% | 🟡 |
| biosuite/core/variant_calling.py | REVIEWED (line-by-line, Sprint I) | 76% | 🟡 |
| biosuite/core/workflow/__init__.py | REVIEWED (line-by-line, Sprint I) | n/a | 🔴 |
| biosuite/core/workflow/batch.py | REVIEWED (line-by-line, Sprint I) | 95% | 🟢 |
| biosuite/core/workflow/pipeline.py | REVIEWED (line-by-line, Sprint I) | 67% | 🔴 |
| biosuite/core/workflow/report.py | REVIEWED (line-by-line, Sprint I) | 89% | 🟡 |
| biosuite/gui/__init__.py | REVIEWED (line-by-line, Sprint I) | n/a | 🔴 |
| biosuite/gui/dialogs.py | REVIEWED (line-by-line, Sprint I) | 13% | 🔴 |
| biosuite/gui/main_window.py | REVIEWED (line-by-line, Sprint I) | 12% | 🔴 |
| biosuite/gui/tabs/__init__.py | REVIEWED (line-by-line, Sprint I) | n/a | 🔴 |
| biosuite/gui/tabs/advanced.py | REVIEWED (line-by-line, Sprint I) | 6% | 🔴 |
| biosuite/gui/tabs/cloning.py | REVIEWED (line-by-line, Sprint I) | 8% | 🔴 |
| biosuite/gui/tabs/databases.py | REVIEWED (line-by-line, Sprint I) | 6% | 🔴 |
| biosuite/gui/tabs/genomics.py | REVIEWED (line-by-line, Sprint I) | 6% | 🔴 |
| biosuite/gui/tabs/help.py | REVIEWED (line-by-line, Sprint I) | 53% | 🔴 |
| biosuite/gui/tabs/metabolomics.py | REVIEWED (line-by-line, Sprint I) | 10% | 🔴 |
| biosuite/gui/tabs/sequence_analysis.py | REVIEWED (line-by-line, Sprint I) | 17% | 🔴 |
| biosuite/gui/tabs/survival.py | REVIEWED (line-by-line, Sprint I) | 12% | 🔴 |
| biosuite/gui/tabs/transcriptomics.py | REVIEWED (line-by-line, Sprint I) | 7% | 🔴 |
| biosuite/gui/tabs/visualization.py | REVIEWED (line-by-line, Sprint I) | 7% | 🔴 |
| biosuite/gui/tabs/workflow.py | REVIEWED (line-by-line, Sprint I) | 7% | 🔴 |
| biosuite/gui/themes.py | REVIEWED (line-by-line, Sprint I) | 100% | 🟢 |
| biosuite/gui/widgets.py | REVIEWED (line-by-line, Sprint I) | 14% | 🔴 |
| biosuite/notebook/__init__.py | REVIEWED (line-by-line, Sprint I) | n/a | 🔴 |
| biosuite/plotting/__init__.py | REVIEWED (line-by-line, Sprint I) | n/a | 🔴 |
| biosuite/plotting/biological_plots.py | REVIEWED (line-by-line, Sprint I) | 60% | 🔴 |
| biosuite/plotting/conservation_plots.py | REVIEWED (line-by-line, Sprint I) | 95% | 🟢 |
| biosuite/plotting/genome_browser.py | REVIEWED (line-by-line, Sprint I) | 71% | 🟡 |
| biosuite/plotting/interactive_plots.py | REVIEWED (line-by-line, Sprint I) | 82% | 🟡 |
| biosuite/plotting/math_plots.py | REVIEWED (line-by-line, Sprint I) | 88% | 🟡 |
| biosuite/plotting/network_plots.py | REVIEWED (line-by-line, Sprint I) | 86% | 🟡 |
| biosuite/plotting/plasmid_map.py | REVIEWED (line-by-line, Sprint I) | 93% | 🟢 |
| biosuite/plotting/plot_api.py | REVIEWED (line-by-line, Sprint I) | 64% | 🔴 |
| biosuite/plotting/sequence_viewer.py | REVIEWED (line-by-line, Sprint I) | 93% | 🟢 |
| biosuite/plotting/specialized_plots.py | REVIEWED (line-by-line, Sprint I) | 61% | 🔴 |
| biosuite/plotting/synteny.py | REVIEWED (line-by-line, Sprint I) | 88% | 🟡 |
| biosuite/plotting/upset_plots.py | REVIEWED (line-by-line, Sprint I) | 93% | 🟢 |
