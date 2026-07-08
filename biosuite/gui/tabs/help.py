"""
Help & Guides tab — all help text for every module.
"""
import customtkinter as ctk

from ..themes import FONT_FAMILY, FONT_SMALL


class HelpTabMixin:
    """Provides the Help & Guides tab with all documentation text."""

    def _build_help_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['help'] = f
        self._section_header(f, "Help & Guides")
        card = self._card(f)
        card.pack(fill='both', expand=True)

        left = ctk.CTkFrame(card, fg_color='transparent', width=220)
        left.pack(side='left', fill='y', padx=(12, 0), pady=12)
        left.pack_propagate(False)

        self._label(left, 'Topics', 'sub').pack(anchor='w', pady=(0, 8))

        self.help_text = self._text_box(card, height=400)
        self.help_text.pack(side='right', fill='both', expand=True, padx=(0, 12), pady=12)

        guides = {
            'Getting Started': self._help_getting_started,
            'Sequence Analysis': self._help_sequence,
            'Alignment Tools': self._help_alignment,
            'Transcriptomics': self._help_transcriptomics,
            'Genomics & NGS': self._help_genomics,
            'Single-Cell': self._help_singlecell,
            'Proteomics': self._help_proteomics,
            'CRISPR Design': self._help_crispr,
            'Metagenomics': self._help_metagenomics,
            'Population Genetics': self._help_popgen,
            'Machine Learning': self._help_ml,
            'Database Search': self._help_databases,
            'File Formats': self._help_fileformats,
            'API Keys Setup': self._help_apikeys,
            'External Tools': self._help_external,
            'Visualization': self._help_visualization,
            'Workflow & Pipeline': self._help_workflow,
            'GO Browser': self._help_go,
            'GWAS Analysis': self._help_gwas,
            'Epitope Prediction': self._help_epitope,
            '16S rRNA Pipeline': self._help_16s,
            'SV / CNV Detection': self._help_svcnv,
            'BigWig Reader': self._help_bigwig,
            'Keyboard Shortcuts': self._help_shortcuts,
            'REST API': self._help_api,
        }

        for name, func in guides.items():
            btn = ctk.CTkButton(left, text=name, anchor='w', height=30, corner_radius=6,
                                font=(FONT_FAMILY, 11), fg_color='transparent',
                                text_color=T['text'], hover_color=T['border'],
                                command=lambda f=func: self._show_guide(f))
            btn.pack(fill='x', pady=1)

        self._show_guide(self._help_getting_started)

    def _show_guide(self, func):
        self.help_text.delete("1.0", "end")
        self.help_text.insert("end", func())

    def _help_getting_started(self):
        return """GETTING STARTED WITH BIOSUITE ULTRA
====================================

Welcome to BioSuite Ultra v4.0 -- a complete bioinformatics platform with 52+ analysis modules, molecular cloning toolkit, plasmid maps, and virtual gel electrophoresis.

QUICK START:
1. Click any tab in the left sidebar to access that analysis tool
2. Each tab has input fields, buttons, and a results area
3. Most tools work immediately with just pip install

NAVIGATION:
- Plots Gallery -- 36+ visualization types (UpSet, Genome Browser, Interactive, Logos)
- Sequence Analysis -- FASTA/FASTQ I/O, GC%, translation, ORF finding
- Alignments -- BLAST search, multiple alignment, pairwise
- Phylogenetics -- Tree building (NJ, ML, Bayesian)
- Expression -- Differential expression, normalization
- Genomics -- Variant calling, SV/CNV detection, read alignment
- Single-Cell -- Full scRNA-seq pipeline
- Protein Structure -- PDB analysis, prediction, docking
- Workflow -- Pipeline builder, batch processor, HTML reports
- GO Browser -- Browse Gene Ontology hierarchy
- GWAS -- Genome-wide association studies
- Epitope -- T-cell and B-cell epitope prediction
- 16S rRNA -- Taxonomic classification pipeline
- SV/CNV -- Structural variant and copy number detection
- BigWig -- Read continuous track data files
- Database Search -- Query NCBI, UniProt, PDB, KEGG

KEYBOARD SHORTCUTS:
- Ctrl+S -- Save current work
- Ctrl+Q -- Quit application
- F1 -- Open help
- F5 -- Refresh current tab
- Escape -- Go to plots gallery

FIRST-TIME SETUP:
No setup required! Everything works out of the box.
Optional: Add API keys in "API Keys Config" for faster database searches."""

    def _help_sequence(self):
        return """SEQUENCE ANALYSIS GUIDE
========================

INPUT: Paste FASTA or FASTQ sequences into the text box, or load from file.

FEATURES:
- Load File -- Read FASTA/FASTQ/GenBank files
- GC% -- Calculate GC content percentage
- Rev Comp -- Get reverse complement
- Translate -- Translate DNA to protein (6 frames)
- Stats -- Full base composition statistics

EXAMPLE WORKFLOW:
1. Paste: >my_seq\\nACGTACGTACGT...
2. Click "GC%" to see composition
3. Click "Translate" to get protein sequence
4. Click "Stats" for detailed breakdown

OUTPUT FORMAT:
Results appear in the right panel with formatted statistics."""

    def _help_alignment(self):
        return """ALIGNMENT TOOLS GUIDE
======================

PAIRWISE ALIGNMENT:
- Enter two sequences in the text boxes
- Needleman-Wunsch = global alignment (full sequences)
- Smith-Waterman = local alignment (best matching region)
- Results show aligned sequences and alignment score

MULTIPLE SEQUENCE ALIGNMENT:
- Paste multiple FASTA sequences
- Click "Auto Align" to run progressive alignment
- Consensus sequence shown below alignment
- Conservation bars indicate similarity

BLAST SEARCH:
- Enter query sequence
- Select or create a database FASTA file
- Results ranked by E-value and score
- Pure Python search works without BLAST+ installed

TIP: For large databases (>10,000 sequences), install BLAST+ for faster results."""

    def _help_transcriptomics(self):
        return """TRANSCRIPTOMICS GUIDE
=======================

RNA-seq QUANTIFICATION:
1. Select your reads FASTQ file
2. Select your transcriptome FASTA file
3. Click "Run Quantification"
4. Results show TPM values per transcript

DIFFERENTIAL EXPRESSION:
1. Load a count matrix (CSV with gene columns as samples)
2. Enter conditions (e.g., control,control,treat,treat)
3. Click "Run DE Analysis"
4. Results show log2FC, p-values, adjusted p-values

NORMALIZATION:
- CPM -- Counts Per Million (depth normalization)
- TPM -- Transcripts Per Million (length + depth normalized)

GO/KEGG ENRICHMENT:
- Enter gene list (comma-separated)
- Select ORA or GSEA method
- Results show enriched pathways with p-values

TIP: For large datasets, install Salmon for faster quantification."""

    def _help_genomics(self):
        return """GENOMICS & NGS GUIDE
======================

READ ALIGNMENT:
1. Select reference FASTA file
2. Select reads FASTQ file
3. Results show mapping statistics

VARIANT CALLING:
1. Select a SAM/BAM file (aligned reads)
2. Click "Run Variant Calling"
3. Results show SNPs and indels with quality scores

PEAK CALLING (ChIP-seq):
1. Select your ChIP-seq BAM/SAM file
2. Results show peaks with positions and scores

GENOME ASSEMBLY:
1. Select reads FASTQ file
2. Results show contig statistics (N50, total length)

PHYLOGENETIC TREES:
- Paste aligned sequences in FASTA format
- Select tree method (NJ, ML, or Bayesian)
- Results show Newick tree and visualization

TIP: Install BWA for faster read alignment on large genomes."""

    def _help_singlecell(self):
        return """SINGLE-CELL RNA-SEQ GUIDE
===========================

INPUT: h5ad file, CSV count matrix, or 10x directory

PIPELINE STEPS:
1. QC Filtering -- Remove low-quality cells
2. Normalization -- Library size correction
3. HVG Detection -- Find variable genes
4. PCA -- Dimensionality reduction
5. UMAP -- 2D visualization
6. Leiden Clustering -- Group cells
7. Marker Detection -- Find cluster markers

PARAMETERS:
- Min genes per cell: 200 (default)
- Max mito%: 20% (default)
- Leiden resolution: 0.5 (higher = more clusters)

OUTPUT:
- Number of cells and clusters
- Cluster sizes
- Top marker genes per cluster

TIP: Install scanpy for full functionality: pip install scanpy"""

    def _help_proteomics(self):
        return """PROTEOMICS GUIDE
==================

PROTEIN STRUCTURE ANALYSIS:
- Enter a PDB ID (e.g., 1CRN) or upload a PDB file
- Results show: atoms, residues, chains, resolution
- Ramachandran plot data
- Surface area calculation

STRUCTURE PREDICTION:
- Enter a protein sequence
- Or enter a UniProt accession ID
- Uses ESMFold (pure Python) or AlphaFold DB

MOLECULAR DOCKING:
- Upload receptor PDB file
- Upload ligand PDB file
- Results show binding energy and pose coordinates

TIP: For best structure prediction, register at alphafold.ebi.ac.uk"""

    def _help_crispr(self):
        return """CRISPR GUIDE RNA DESIGN GUIDE
===============================

INPUT: Enter target DNA sequence (50-100bp recommended)

PAM TYPES:
- SpCas9 -- Most common (NGG PAM, 20bp guide)
- SaCas9 -- Smaller PAM (NNGRRT)
- Cas12a -- Different PAM (TTTV, 23bp guide)

RESULTS:
- Guide RNA sequences ranked by quality score
- GC content of each guide
- Position and strand information

SCORING FACTORS:
- GC content (40-70% is optimal)
- No poly-T runs (TTTT = Pol III terminator)
- No excessive GC repeats

TIP: Higher-scoring guides generally have better on-target efficiency."""

    def _help_metagenomics(self):
        return """METAGENOMICS GUIDE
====================

TAXONOMIC CLASSIFICATION:
- Select reads FASTQ file
- Results show taxonomic composition
- Bar chart of relative abundances

DIVERSITY METRICS:
- Shannon entropy -- Species diversity
- Simpson index -- Dominance measure
- Chao1 estimator -- Total species count
- Bray-Curtis distance -- Community dissimilarity

TIP: For real metagenomics, install Kraken2 for accurate classification."""

    def _help_popgen(self):
        return """POPULATION GENETICS GUIDE
==========================

INPUT: Enter genotype matrix (rows = samples, values 0/1/2)
- 0 = homozygous reference
- 1 = heterozygous
- 2 = homozygous alternate

ANALYSES:
- Hardy-Weinberg Equilibrium -- Test genotype frequencies
- FST -- Population differentiation
- Nucleotide Diversity (pi) -- Sequence variation
- Tajima's D -- Selection detection
- Linkage Disequilibrium -- Non-random association
- PCA -- Population structure visualization

EXAMPLE:
0,0,1,2,0,1
0,1,1,2,0,0
1,1,2,2,1,0"""

    def _help_ml(self):
        return """MACHINE LEARNING GUIDE
=======================

INPUT: CSV file with feature columns and a label column

MODELS:
- Random Forest (RF) -- Best for feature importance
- Support Vector Machine (SVM) -- Good for small datasets

PARAMETERS:
- Model type: RF or SVM
- Label column: Name of column with class labels

OUTPUT:
- Accuracy score
- Cross-validation scores (5-fold)
- Feature importances (RF only)
- Confusion matrix

TIP: For best results, normalize features before training."""

    def _help_databases(self):
        return """DATABASE SEARCH GUIDE
=======================

SELECT DATABASE from dropdown:
- NCBI -- Nucleotide/protein sequences
- UniProt -- Protein annotation
- PDB -- 3D protein structures
- KEGG -- Biological pathways
- Ensembl -- Gene annotations
- All -- Search all at once

HOW TO SEARCH:
1. Select database from dropdown
2. Type search query (gene name, protein name, accession)
3. Click "Search"
4. Results appear in the text area

TIP: Add API keys in "API Keys Config" for faster NCBI searches."""

    def _help_fileformats(self):
        return """FILE FORMATS GUIDE
====================

SUPPORTED FORMATS:
- BED -- Genomic intervals (chrom, start, end, ...)
- GFF/GTF -- Genome annotation
- Newick -- Phylogenetic trees
- Stockholm -- Sequence alignments

HOW TO USE:
1. Click "Browse" to select a file
2. Click "Parse File"
3. Results show file summary and contents

BED FILE FORMAT:
chrom start end name score strand
chr1 1000 2000 gene1 0 +

GFF FILE FORMAT:
seqid source feature start end score strand phase attributes

NEWICK FORMAT:
(A:0.1,B:0.2)C:0.3;

TIP: Files must be plain text (no compressed formats)."""

    def _help_apikeys(self):
        return """API KEYS SETUP GUIDE
=======================

API keys are OPTIONAL. Everything works without them.
Keys just make database searches faster.

HOW TO ADD KEYS:
1. Go to "API Keys Config" tab
2. Enter your email and/or API key for each service
3. Click "Save All Keys"
4. Keys are stored locally in biosuite_config.json

WHERE TO GET KEYS:

NCBI (for faster nucleotide/protein search):
- Go to: ncbi.nlm.nih.gov/account
- Create free account
- Go to Settings -> API Key Management
- Generate key
- Enter email + key in BioSuite

KEGG (for pathway database):
- Go to: genome.jp/kegg
- Register for academic access
- Enter your email

AlphaFold (for structure prediction):
- Go to: alphafold.ebi.ac.uk
- Register for API access
- Enter your email

UniProt, PDB, Ensembl:
- No key needed -- just use them directly

TIP: The NCBI key gives 10x faster searches (10/sec vs 3/sec)."""

    def _help_external(self):
        return """EXTERNAL TOOLS GUIDE
======================

BioSuite works WITHOUT external tools.
External tools are optional SPEED BOOSTS.

LIST OF OPTIONAL TOOLS:

Alignment:
- BLAST+ -- Faster database search
- Clustal Omega -- Faster multiple alignment
- MUSCLE -- Alternative MSA tool
- MAFFT -- Large-scale alignment

NGS:
- BWA -- Faster read alignment
- Bowtie2 -- Alternative aligner
- FreeBayes -- Variant calling
- MACS2 -- ChIP-seq peaks
- Salmon -- RNA-seq quantification
- Kallisto -- Alternative quantifier

Assembly:
- SPAdes -- Short read assembly
- MEGAHIT -- Metagenome assembly

Phylogenetics:
- RAxML -- Maximum likelihood trees
- IQ-TREE -- Alternative ML method
- MrBayes -- Bayesian inference

Other:
- Kraken2 -- Taxonomic classification
- AutoDock Vina -- Molecular docking
- OpenMM -- Molecular dynamics
- COBRApy -- Metabolic modeling

HOW TO INSTALL:
Each tool has its own installation instructions.
Most are available as conda packages or pre-compiled binaries.

TIP: Start without external tools. Add them only when you need faster processing on large datasets."""

    def _help_visualization(self):
        return """VISUALIZATION GUIDE
=====================

BioSuite includes 36+ visualization types in the Plots Gallery.

STATISTICAL PLOTS:
- Volcano Plot -- Differential expression (log2FC vs p-value)
- PCA Plot -- Principal component analysis
- Manhattan Plot -- GWAS association signals
- MA Plot -- Mean vs fold change
- QQ-Plot -- Normality check

BIOLOGICAL PLOTS:
- Venn Diagram -- Set intersections (2-3 sets)
- UpSet Plot -- Multi-set intersections (better than Venn for 3+ sets)
- Heatmap -- Correlation or expression matrix
- Clustered Heatmap -- With dendrograms

DISTRIBUTION PLOTS:
- Barplot, Boxplot, Violin, Raincloud, Ridge, Dot Plot

GENOMICS PLOTS:
- Genome Browser -- BAM coverage, VCF, BED track viewer
- Synteny Dotplot -- Whole-genome comparison
- Network Plot -- PPI, regulatory, metabolic networks

SEQUENCE PLOTS:
- Sequence Logo -- Information-content-weighted
- Conservation Bar -- Per-position conservation scores
- Alignment Viewer -- Colored alignment display

INTERACTIVE PLOTS (Plotly):
- Scatter, Bar, Heatmap, Volcano, Line, Pie
- Export as HTML with hover, zoom, and save

HOW TO USE:
1. Select a category from the left panel
2. Click a plot name
3. Click "Generate Plot"
4. Plots open in a new window"""

    def _help_workflow(self):
        return """WORKFLOW & PIPELINE GUIDE
==========================

Pipeline Builder:
Chain multiple analysis steps into automated workflows.

Usage:
1. Enter steps (one per line): name:function_name
2. Click "Run Pipeline"
3. View results in the output panel

Example steps:
  gc:gc_content
  revcomp:reverse_complement
  translate:translate

Batch Processor:
Run the same analysis on hundreds of samples in parallel.

Usage:
1. Enter sample IDs (one per line)
2. Click "Run Batch"
3. View aggregated results

HTML Report Generator:
Create styled HTML reports with stats, plots, and tables.

Usage:
1. Click "Load CSV" or "Demo Data"
2. Click "Generate"
3. Report opens in browser"""

    def _help_go(self):
        return """GENE ONTOLOGY BROWSER
========================

Browse and search Gene Ontology terms.

GO NAMESPACES:
- BP -- Biological Process (e.g., cell adhesion, apoptosis)
- MF -- Molecular Function (e.g., kinase activity, ATP binding)
- CC -- Cellular Component (e.g., nucleus, cytoplasm)

HOW TO USE:
1. Enter a search term (e.g., "kinase", "apoptosis")
2. Click "Search"
3. Click a term to see parents and children
4. Use "Browse BP/MF/CC" to list all terms in a namespace

SEARCHABLE TERMS:
kinase, apoptosis, adhesion, phosphorylation, transcription,
transport, signaling, catalytic, binding, membrane, nucleus...

TIP: The GO browser includes 30 built-in terms for quick reference.
For full GO, install goatools and provide an OBO file."""

    def _help_gwas(self):
        return """GWAS ANALYSIS GUIDE
====================

Genome-Wide Association Studies -- find genetic variants associated with traits.

INPUT:
- Load a CSV file with columns: chrom, pos, snp_id, case_alt, case_ref, ctrl_alt, ctrl_ref
- Or click "Demo Data" to use simulated GWAS data

ANALYSIS:
- Chi-squared test for each SNP
- Benjamini-Hochberg correction for multiple testing
- Lead SNP detection per locus

OUTPUT:
- Total SNPs tested
- Genome-wide significant (p < 5e-8)
- Suggestive associations (p < 1e-5)
- Lead SNPs with odds ratios

INTERPRETATION:
- p < 5e-8 = Genome-wide significant
- p < 1e-5 = Suggestive
- OR > 1 = Risk allele
- OR < 1 = Protective allele"""

    def _help_epitope(self):
        return """EPITOPE PREDICTION GUIDE
==========================

Predict T-cell and B-cell epitopes for vaccine design.

T-CELL EPITOPES:
- MHC Class I binding prediction
- Supports multiple HLA types (A0201, A0101, A0301, etc.)
- Peptide lengths: 8-11 amino acids
- Scoring: anchor residues, hydrophobicity, HLA supertype

B-CELL EPITOPES:
- Surface accessibility (Emini scale)
- Flexibility (Karplus-Schulz)
- B-cell propensity (Parker scale)
- Hydrophilicity

INPUT:
- Paste a protein sequence (one-letter codes)
- Select HLA type from dropdown
- Click "Predict"

OUTPUT:
- Top T-cell epitopes with binding scores
- Top B-cell epitopes with accessibility scores
- Linear epitope predictions
- Proteasomal cleavage sites

TIP: Longer peptides (15-20 aa) are better for B-cell epitopes.
Short peptides (8-11 aa) are better for MHC Class I binding."""

    def _help_16s(self):
        return """16S rRNA CLASSIFICATION GUIDE
===============================

Classify 16S rRNA sequences to identify bacterial species.

INPUT:
- Enter sequences in format: name:sequence (one per line)
- Or click "Demo" for example data

BUILT-IN DATABASE:
- Escherichia coli
- Staphylococcus aureus
- Bacillus subtilis
- Lactobacillus acidophilus
- Clostridium difficile
- Pseudomonas aeruginosa
- Streptococcus pneumoniae
- Helicobacter pylori

METHOD:
- Sliding window identity comparison
- Best match against reference 16S sequences
- Confidence score based on identity

OUTPUT:
- Taxonomic classification for each sequence
- Abundance table (% of each species)
- Confidence scores

TIP: For production use, provide a larger reference database.
The built-in DB has 8 common gut bacteria for testing."""

    def _help_svcnv(self):
        return """SV / CNV DETECTION GUIDE
==========================

Detect structural variants and copy number variations from coverage data.

STRUCTURAL VARIANTS (SV):
- DEL -- Deletion (coverage drops)
- DUP -- Duplication (coverage increases)
- INV -- Inversion
- INS -- Insertion

COPY NUMBER VARIATIONS (CN):
- CN=0,1 -- Deletion
- CN=2 -- Normal diploid
- CN=3,4 -- Duplication/amplification

INPUT:
- Load a CSV file with coverage values (one per line)
- Or click "Demo Data" with artificial deletions and duplications

METHOD:
- Sliding window coverage depth analysis
- Ratio comparison to reference
- Overlapping SV merging

OUTPUT:
- List of detected SVs with positions and types
- CNV bins with copy numbers
- Confidence scores for each variant"""

    def _help_bigwig(self):
        return """BIGWIG READER GUIDE
=====================

Read and visualize BigWig (.bw) continuous track data files.

BigWig files store per-base signal data (e.g., ChIP-seq coverage,
RNA-seq expression, conservation scores).

HOW TO USE:
1. Click "Browse File" to select a .bw or .bigwig file
2. Click "Read File" to load and display info
3. View chromosome names and value counts

SUPPORTED OPERATIONS:
- Read chromosome names and sizes
- Extract values for specific regions
- Summarize into bins (mean, max)

COMMON BIGWIG SOURCES:
- UCSC Genome Browser downloads
- ENCODE project data
- UCSC Conservation scores
- ChIP-seq signal tracks

TIP: BigWig files are binary -- they cannot be read as text.
The reader parses the binary format directly."""

    def _help_shortcuts(self):
        return """KEYBOARD SHORTCUTS
====================

BioSuite supports these keyboard shortcuts:

GLOBAL:
- Ctrl+S -- Save current work
- Ctrl+Q -- Quit application
- F1 -- Open Help & Guides
- F5 -- Refresh current tab
- Escape -- Return to Plots Gallery

TIPS:
- Shortcuts work from any tab
- Ctrl+S is context-aware (saves current tab's data)
- F5 reloads the current tab's interface
- Escape always takes you back to the main plots view"""

    def _help_api(self):
        return """REST API SERVER
=================

BioSuite includes a built-in REST API server that exposes all 48
analysis modules as HTTP endpoints. 100% free, no paid features.

HOW TO START:
1. Go to the "System" section and click "Launch API Server"
2. Or use CLI: option 90
3. Or run: python -m biosuite.api.server

API DOCUMENTATION:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health

AVAILABLE ENDPOINTS:
- /api/v1/sequence/* -- GC content, reverse complement, translation
- /api/v1/alignment/* -- Needleman-Wunsch, Smith-Waterman
- /api/v1/expression/* -- Differential expression, normalization
- /api/v1/crispr/* -- Guide RNA design
- /api/v1/metagenomics/* -- Taxonomic classification
- /api/v1/plotting/* -- Generate plots
- /api/v1/database/* -- NCBI, UniProt, PDB, KEGG search
- /api/v1/provenance/* -- Workflow tracking

EXAMPLE USAGE:
# GC Content
curl -X POST http://localhost:8000/api/v1/sequence/gc-content \\
     -H "Content-Type: application/json" \\
     -d '{"sequence": "ATCGATCG"}'

# Alignment
curl -X POST http://localhost:8000/api/v1/alignment/needleman-wunsch \\
     -H "Content-Type: application/json" \\
     -d '{"seq1": "AGTACGCA", "seq2": "TATGC"}'

# From Python
import requests
r = requests.post("http://localhost:8000/api/v1/sequence/gc-content",
                   json={"sequence": "ATCGATCG"})
print(r.json())  # {"gc_percent": 50.0, "sequence_length": 8}

BENEFITS:
- Web apps can integrate with BioSuite
- Mobile apps can call the API
- Other tools can interoperate
- Works offline (no cloud required)
- 100% free, MIT license"""
