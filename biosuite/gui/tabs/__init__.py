"""
GUI tab modules — each mixin adds tab builder + action methods to BioSuiteApp.
"""
from .advanced import AdvancedTabMixin
from .cloning import CloningTabMixin
from .databases import DatabasesTabMixin
from .genomics import GenomicsTabMixin
from .help import HelpTabMixin
from .metabolomics import MetabolomicsTabMixin
from .sequence_analysis import SequenceAnalysisTabMixin
from .survival import SurvivalTabMixin
from .transcriptomics import TranscriptomicsTabMixin
from .visualization import VisualizationTabMixin
from .workflow import WorkflowTabMixin

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
