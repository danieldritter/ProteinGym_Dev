"""
Generic model classes 
"""

import json
import warnings
from abc import ABC, abstractmethod, abstractproperty
from typing import Callable, List, Optional, Union

import numpy as np
import torch

from ..utils.alignment import Alignment


class SequenceFitnessModel(ABC):
    """
    the abstract base class for all sequence fitness models
    """

    _available_models = {}

    def __init__(
        self,
        model_checkpoint: Optional[str] = None,
        model_type: Optional[str] = None,
        **kwargs,
    ):
        self.model_checkpoint = model_checkpoint
        self.model_type = model_type
        # This is done so that when we multiple inheritance cases or want to pass additional arguments beyond what is necessary (e.g. calling a non-alignment constructor with
        # configs that have alignment info) we don't get an error. All the kwargs that aren't used somewhere prior in the inheritance chain are stored here
        self.unused_kwargs = kwargs
        if len(self.unused_kwargs) > 0:
            warnings.warn(
                "Unused kwargs in model initialization (Certain kwargs may not be needed for a particular model): "
                + str(list(self.unused_kwargs.keys())),
                stacklevel=2,
            )

    @abstractmethod
    def predict_fitnesses(
        self,
        sequences: List[str],
        wt_sequence: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> List[float]:
        """Given a list of sequences, returns the predicted fitness values for each sequence.

        Args:
            sequences (List[str]): List of input sequences to predict fitness values for
            wt_sequence (Union[str, None], optional): The wild-type sequence to compare against. Defaults to None.
            If passed in, fitness values are relative to the wild-type sequence
            batch_size (Union[int, None], optional): The batch size to use when predicting fitness values. Defaults to None, meaning a single batch of all data.

        Returns:
            List[float]: List of fitness values for each sequence
        """


class ProbabilitySequenceFitnessModel(SequenceFitnessModel):
    """
    Class for sequence fitness models that produce log-probabilities as fitness predictions
    """

    def predict_fitnesses(
        self,
        sequences: List[str],
        wt_sequence: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> List[float]:
        return self.predict_logprobs(sequences, wt_sequence, batch_size=batch_size)

    @abstractmethod
    def predict_logprobs(
        self, sequences: List[str], wt_sequence: Optional[str] = None, batch_size: Optional[int] = None
    ) -> List[float]:
        """Given a list of sequences, produce the log-probability estimates for each sequence.
        If the wild type sequence is passed in, returned values are the log-ratio between the mutated and wild type sequence probability

        Args:
            sequences (List[str]): list of sequences to predict probabilities for
            wt_sequence (Union[str, None], optional): wild-type sequence to compute probabilities relative to. Defaults to None.
            batch_size (Union[int, None], optional): batch size to use when predicting probabilities. Defaults to None, meaning a single batch of all data.

        Returns:
            List[float]: list of predicted log-probabilities for each sequence
        """


class ProteinLanguageModel(ProbabilitySequenceFitnessModel, torch.nn.Module):
    """
    Base class for all protein language models
    """

    def __init__(
        self,
        **kwargs,
    ):
        """Initializes ProteinLanguageModel Class
        """
        torch.nn.Module.__init__(self)
        ProbabilitySequenceFitnessModel.__init__(self, **kwargs)
    
    @property
    @abstractmethod
    def base_model(self) -> torch.nn.Module:
        """Returns the underlying model"""
        
    @abstractmethod
    def predict_position_logprobs(
        self, sequences: List[str], wt_sequence: Optional[str] = None, batch_size: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Predicts the per-position log probabilities for a given list of sequences.

        Args:
            sequences (List[str]): The list of sequences for which to predict the position log probabilities.
            wt_sequence (Union[str, None], optional): The wild-type sequence to compare against. Defaults to None.
            batch_size (Union[int, None], optional): The batch size to use when predicting fitness values. Defaults to None, meaning a single batch of all data.
        Returns:
            fitnesses (List[np.ndarray]): A list of numpy arrays representing the position log probabilities for each sequence.
                Each ndarray is shape (seq_len, vocab_size)
        """

    @abstractmethod
    def get_embeddings(
        self, sequences: List[str], layers: List[int] = [-1], batch_size: Optional[int] = None
    ) -> dict[int, torch.Tensor]:
        """Returns embeddings for the given sequences at the given layers.

        Args:
            sequences (Union[List[str], torch.Tensor]): list of sequences to get embeddings for
            layer (List[int], optional): Name of layers to get embeddings for. Defaults to -1.
            batch_size (Union[int, None], optional): The batch size to use when computing embeddings. Defaults to None, meaning a single batch of all data.

        Returns:
            dict[int,torch.Tensor]: dictionary of layer indexes and their corresponding embeddings
        """

    @abstractmethod
    def get_embed_dim(self, layers: List[int] = [-1]) -> List[int]:
        """returns a list of the dimension of the embeddings for the specified layers. Used in finetuning to initialize head layer on top of embeddings

        Args:
            layers (List[int], optional): Name of layer to get embeddings for. Defaults to -1, last layer.

        Returns:
            List[int]: dimension of the embeddings
        """
    
    @abstractmethod 
    def set_trainable_layers(self, layers: List[int] = [-1]):
        """Sets the specified layers to be trainable in the model
        
        Args:
            layers (List[int], optional): index of layers to set as trainable for. Defaults to -1, last layer.    
        """
        pass 



class AlignmentModel(SequenceFitnessModel):
    """
    Superclass for all alignment models
    """

    def __init__(
        self,
        alignment: Union[Alignment, None] = None,
        alignment_kwargs: Union[dict, None] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if alignment is not None:
            self.alignment_file = alignment.alignment_file
            self.alignment = alignment
        else:
            if alignment_kwargs is None:
                raise ValueError("Must pass in alignment or alignment_kwargs")
            if "alignment_file" not in alignment_kwargs:
                raise ValueError("Must provide alignment in alignment_kwargs")
            alignment_file = alignment_kwargs["alignment_file"]
            del alignment_kwargs["alignment_file"]
            self.alignment = Alignment(alignment_file, **alignment_kwargs)
            self.alignment_file = alignment_file


"""
This class is very similar to the above AlignmentModel class, but I've kept it separate since there are some models that use alignments for probabilities (e.g. EVE, MSATransformer)
and some that use them to produce other fitness values (e.g. Provean or GEMME). It just enforces the abstract methods for a probability model using the ProbabilitySequenceFitnessModel class, 
which is a mixin so there's no __init__ conflicts. 
"""


class AlignmentProbabilityModel(AlignmentModel, ProbabilitySequenceFitnessModel):
    """
    Superclass for all alignment probability models
    """

    def __init__(
        self,
        alignment: Union[Alignment, None] = None,
        alignment_kwargs: Union[dict, None] = None,
        **kwargs,
    ):
        AlignmentModel.__init__(self, alignment, alignment_kwargs)
        ProbabilitySequenceFitnessModel.__init__(self, **kwargs)
