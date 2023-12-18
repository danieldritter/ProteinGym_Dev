"""
Base class for all alignment models
"""
from .generic_models import ProbabilitySequenceFitnessModel
from ..utils.alignment_utils import read_fasta, get_focus_alignment, get_focus_cols


class AlignmentModel(ProbabilitySequenceFitnessModel):
    """
    Superclass for all alignment models
    """

    def __init__(
        self,
        alignment_file: str,
        model_name: str = "",
        focus_seq_ind: int = 0,
    ):
        """
        Initializes an AlignmentModel object.

        Parameters:
        alignment_file (str): The path to the alignment file.
        model_name (str): The name of the model (optional).
        model_type (str): The type of the model (optional).
        focus_cols_only (bool): Flag indicating whether to store focus columns (default is True).
        focus_seq_ind (int): The index of the focus sequence (default is 0).
        """
        super().__init__(model_name)

        # Set instance variables
        self.alignment_file = alignment_file
        self.focus_seq_ind = focus_seq_ind

        # Read the alignment file
        if self.focus_seq_ind is not None:
            # Read alignment file and get focus alignment
            self.alignment = get_focus_alignment(
                read_fasta(self.alignment_file),
                self.focus_seq_ind,
            )
        else:
            # Read alignment file
            self.alignment = read_fasta(self.alignment_file)

        # Get focus columns
        self.focus_cols = get_focus_cols(self.alignment, self.focus_seq_ind)
