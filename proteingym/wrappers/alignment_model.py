"""
Base class for all alignment models
"""
from typing import Union
from .generic_models import ProbabilitySequenceFitnessModel
from ..utils.alignment import Alignment


class AlignmentModel(ProbabilitySequenceFitnessModel):
    """
    Superclass for all alignment models
    """

    def __init__(
        self,
        model_checkpoint: Union[str, None] = None,
        alignment: Union[Alignment, None] = None,
        alignment_file: str = "",
        alignment_format: str = "fasta",
        focus_seq_ind: int = 0,
        perc_max_gap_seq_fragments: float = 1.0,
        perc_max_gap_column_coverage: float = 1.0,
        uppercase: bool = True
    ):
        super().__init__(model_checkpoint=model_checkpoint)
        if alignment is not None:
            self.alignment = alignment
        else:
            self.alignment = Alignment(alignment_file, alignment_format, focus_seq_ind,
                                       perc_max_gap_seq_fragments, perc_max_gap_column_coverage, uppercase)
