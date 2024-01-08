"""
Base class for all protein language model models
"""
from typing import List, Union
from abc import abstractmethod
import numpy as np
from .generic_models import ProbabilitySequenceFitnessModel


class ProteinLanguageModel(ProbabilitySequenceFitnessModel):
    """
    Base class for all protein language models
    """

    def __init__(self, model_checkpoint: Union[str, None] = None, eval_mode: bool = True, nogpu: bool = False):
        super().__init__(model_checkpoint)
        self.eval_mode = eval_mode
        self.nogpu = nogpu

    @abstractmethod
    def predict_position_logprobs(self, sequences: List[str], wt_sequence: Union[str, None] = None) -> List[np.ndarray]:
        """
        Predicts the per-position log probabilities for a given list of sequences.

        Args:
            sequences (List[str]): The list of sequences for which to predict the position log probabilities.
            wt_sequence (Union[str, None], optional): The wild-type sequence to compare against. Defaults to None.
        Returns:
            fitnesses (List[np.ndarray]): A list of numpy arrays representing the position log probabilities for each sequence. 
                Each ndarray is shape (seq_len, vocab_size)
        """

    @abstractmethod
    def get_embeddings(self, sequences: List[str], layer: str = "last") -> List[np.ndarray]:
        """
        Get contextualized embeddings from the model. Defaults to embeddings 
        from last layer, but a layer name can be specified. 

        Args:
            sequences (List[str]): List of sequences to get embeddings for.
            layer (str, optional): Name of layer to get embeddings for. Defaults to "last".

        Returns:
            List[np.ndarray]: List of embeddings for each sequence. Each ndarray is shape (seq_len, embedding_dim)
        """
