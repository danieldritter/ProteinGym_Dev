"""
This module contains the SiteIndependentModel class.
"""
from typing import Union, List
import numpy as np
from proteingym.wrappers import alignment_model
from proteingym.constants.data import AA_INTEGER_MAP


@alignment_model.AlignmentModel.register("site_independent_model")
class SiteIndependentModel(alignment_model.AlignmentModel):
    """
    Site independent columnwise probability alignment model
    """

    def __init__(
        self,
        alignment: Union[alignment_model.Alignment, None] = None,
        alignment_file: str = "",
        alignment_format: str = "fasta",
        focus_seq_ind: int = 0,
        perc_max_gap_seq_fragments: float = 1.0,
        perc_max_gap_column_coverage: float = 1.0,
        pseudocount: Union[int, np.ndarray] = 1,
    ) -> None:
        """
        Initializes an instance of the class.

        Args:
            alignment_file (str): The path to the alignment file.
            focus_seq_ind (int, optional): The index of the alignment focus sequence. Defaults to 0.
            pseudocount (int or np.ndarray, optional): pseudocount values to add initially to frequency matrix.
                Can be a matrix of values or one integer, in which case the integer is used in all entries.

        Returns:
            None
        """
        # Uppercasing converts . to - along with letters
        super().__init__(None, alignment, alignment_file, alignment_format, focus_seq_ind,
                         perc_max_gap_seq_fragments, perc_max_gap_column_coverage, uppercase=True)
        self.pseudocount = pseudocount
        if isinstance(pseudocount, np.ndarray):
            assert pseudocount.shape == (len(list(AA_INTEGER_MAP.keys(
            ))), len(self.alignment.alignment[0])), "Pseudocount array must be the same shape as the alignment sequences"
        self.log_frequencies = np.log(  # type: ignore
            self.compute_frequencies())

    def compute_frequencies(self) -> np.ndarray:
        """
        Compute the frequencies of amino acids in the alignment.

        Returns:
            np.ndarray: A 2D array representing the normalized frequencies of amino acids in the alignment.
        """
        if isinstance(self.pseudocount, int):
            frequencies = (np.ones((len(list(AA_INTEGER_MAP.keys())), len(
                self.alignment.alignment[0]))) * self.pseudocount)
        else:
            frequencies = self.pseudocount

        for sequence in self.alignment.alignment:
            for i, aa in enumerate(sequence):
                frequencies[AA_INTEGER_MAP[aa], i] += 1
        frequencies = frequencies / np.sum(frequencies, axis=0)
        return frequencies

    def predict_logprobs(self, sequences: List[str], wt_sequence: Union[str, None] = None) -> List[float]:
        """
        Generate the function comment for the given function body in a markdown code block with the correct language syntax.

        Parameters:
            sequences (list): The input sequences to predict log probabilities for.

        Returns:
            logprobs (np.ndarray): The predicted log probabilities for the given sequences.
        """
        logprobs = []
        if wt_sequence is not None:
            assert len(wt_sequence) == len(
                self.alignment.alignment[0]
            ), "wt sequence must be the same length as the alignment"
            wt_seq_indices = [AA_INTEGER_MAP[aa] for aa in wt_sequence]
            wt_logprob = np.sum(
                self.log_frequencies[wt_seq_indices, range(len(wt_sequence))])
        else:
            wt_logprob = 0.0

        for sequence in sequences:
            assert len(sequence) == len(
                self.alignment.alignment[0]
            ), "Sequences must be the same length as the alignment"
            seq_indices = [AA_INTEGER_MAP[aa] for aa in sequence]
            logprob = np.sum(
                self.log_frequencies[seq_indices, range(len(sequence))])
            logprob = logprob - wt_logprob
            logprobs.append(logprob)
        return logprobs
