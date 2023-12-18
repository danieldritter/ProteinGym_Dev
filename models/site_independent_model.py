"""
This module contains the SiteIndependentModel class.
"""
from typing import Union, List
import numpy as np
from ..wrappers import alignment_model
from ..constants.data import AA_INTEGER_MAP


class SiteIndependentModel(alignment_model.AlignmentModel):
    """
    Site independent columnwise probability alignment model
    """

    def __init__(
        self,
        alignment_file: str,
        model_name: str = "",
        focus_seq_ind=0,
        pseudocount: Union[int, np.ndarray] = 0,
    ) -> None:
        """
        Initializes an instance of the class.

        Args:
            alignment_file (str): The path to the alignment file.
            model_name (str, optional): The name of the model. Defaults to "".
            focus_seq_ind (int, optional): The index of the alignment focus sequence. Defaults to 0.
            pseudocount (int or np.ndarray, optional): pseudocount values to add initially to frequency matrix.
                Can be a matrix of values or one integer, in which case the integer is used in all entries.

        Returns:
            None
        """
        super().__init__(alignment_file, model_name, focus_seq_ind)
        self.pseudocount = pseudocount
        if isinstance(pseudocount, np.ndarray):
            assert pseudocount.shape == (len(self.alignment[0]), len(list(AA_INTEGER_MAP.keys(
            )))), "Pseudocount array must be the same shape as the alignment sequences"
        self.log_frequencies = np.log(  # type: ignore
            self.compute_frequencies())

    def compute_frequencies(self) -> np.ndarray:
        """
        Compute the frequencies of amino acids in the alignment.

        Returns:
            np.ndarray: A 2D array representing the normalized frequencies of amino acids in the alignment.
        """
        if isinstance(self.pseudocount, int):
            frequencies = (
                np.ones((len(self.alignment[0]), len(list(AA_INTEGER_MAP.keys())))) * self.pseudocount)
        else:
            frequencies = self.pseudocount

        for sequence in self.alignment:
            for i, aa in enumerate(sequence):
                frequencies[i, AA_INTEGER_MAP[aa]] += 1

        return frequencies

    def predict_logprobs(self, sequences: List[str]) -> np.ndarray:
        """
        Generate the function comment for the given function body in a markdown code block with the correct language syntax.

        Parameters:
            sequences (list): The input sequences to predict log probabilities for.

        Returns:
            logprobs (np.ndarray): The predicted log probabilities for the given sequences.
        """
        logprobs = []
        for sequence in sequences:
            assert len(sequence) == len(
                self.alignment[0]
            ), "Sequences must be the same length as the alignment"
            seq_indices = [AA_INTEGER_MAP[aa] for aa in sequence]
            logprobs.append(np.sum(self.log_frequencies[:, seq_indices]))
        return np.array(logprobs)
