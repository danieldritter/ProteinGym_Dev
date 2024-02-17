"""
This module contains the SiteIndependentModel class.
"""
from typing import Union, List
import numpy as np
from proteingym.wrappers.generic_models import AlignmentProbabilityModel
from proteingym.constants.data import AA_INTEGER_MAP


@AlignmentProbabilityModel.register("site_independent_model")
class SiteIndependentModel(AlignmentProbabilityModel):
    """
    Site independent columnwise probability alignment model
    """

    def __init__(
        self,
        pseudocount: Union[int, np.ndarray] = 1,
        include_gaps: bool = False,
        **kwargs
    ):
        # Uppercasing converts . to - along with letters
        kwargs_parsed = self.parse_config(kwargs)
        # Defaulting to uppercase alignments for site-independent models
        if "alignment_kwargs" in kwargs:
            kwargs_parsed["alignment_kwargs"]["uppercase"] = True
        else:
            kwargs_parsed["alignment_kwargs"] = {"uppercase": True}
        super().__init__(model_checkpoint=None, **kwargs_parsed)
        self.pseudocount = pseudocount
        self.include_gaps = include_gaps
        if include_gaps:
            self.aa_dict = AA_INTEGER_MAP + {"-": len(AA_INTEGER_MAP)}
        else:
            self.aa_dict = AA_INTEGER_MAP
        self.alignment_length = len(
            list(self.alignment.seq_name_to_sequences.values())[0])
        if isinstance(pseudocount, np.ndarray):
            assert pseudocount.shape == (len(list(self.aa_dict.keys(
            ))), self.alignment_length), "Pseudocount array must be the same shape as the alignment sequences"
        self.log_frequencies = np.log(  # type: ignore
            self.compute_frequencies())

    def compute_frequencies(self) -> np.ndarray:
        """
        Compute the frequencies of amino acids in the alignment.

        Returns:
            np.ndarray: A 2D array representing the normalized frequencies of amino acids in the alignment.
        """
        if isinstance(self.pseudocount, int):
            frequencies = (np.ones(
                (len(list(self.aa_dict.keys())), self.alignment_length)) * self.pseudocount)
        else:
            frequencies = self.pseudocount

        for sequence in self.alignment.seq_name_to_sequences.values():
            for i, aa in enumerate(sequence):
                if not self.include_gaps and aa == "-":
                    continue
                frequencies[self.aa_dict[aa], i] += 1
        frequencies = frequencies / np.sum(frequencies, axis=0)
        return frequencies

    def predict_logprobs(self, sequences: List[str], wt_sequence: Union[str, None] = None) -> List[float]:
        logprobs = []
        if wt_sequence is not None:
            assert len(
                wt_sequence) == self.alignment_length, "wt sequence must be the same length as the alignment"
            wt_seq_indices = [self.aa_dict[aa] for aa in wt_sequence]
            wt_logprob = np.sum(
                self.log_frequencies[wt_seq_indices, range(len(wt_sequence))])
        else:
            wt_logprob = 0.0

        for sequence in sequences:
            assert len(
                sequence) == self.alignment_length, "Sequences must be the same length as the alignment"
            seq_indices = [self.aa_dict[aa] for aa in sequence]
            logprob = np.sum(
                self.log_frequencies[seq_indices, range(len(sequence))])
            logprob = logprob - wt_logprob
            logprobs.append(logprob)
        return logprobs
