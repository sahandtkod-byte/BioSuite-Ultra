"""
GUI tab modules — each mixin adds tab builder + action methods to BioSuiteApp.
"""
from .visualization import VisualizationTabMixin
from .sequence_analysis import SequenceAnalysisTabMixin
from .transcriptomics import TranscriptomicsTabMixin
from .genomics import GenomicsTabMixin
from .advanced import AdvancedTabMixin
from .databases import DatabasesTabMixin
from .workflow import WorkflowTabMixin
from .help import HelpTabMixin
from .cloning import CloningTabMixin
from .survival import SurvivalTabMixin
from .metabolomics import MetabolomicsTabMixin

__all__ = [
    'VisualizationTabMixin',
    'SequenceAnalysisTabMixin',
    'TranscriptomicsTabMixin',
    'GenomicsTabMixin',
    'AdvancedTabMixin',
    'DatabasesTabMixin',
    'WorkflowTabMixin',
    'HelpTabMixin',
    'CloningTabMixin',
    'SurvivalTabMixin',
    'MetabolomicsTabMixin',
]
