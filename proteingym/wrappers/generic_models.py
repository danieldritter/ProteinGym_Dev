"""Generic model classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import numpy as np
import torch

from proteingym.utils.alignment import Alignment, AlignmentConfig


class SequenceFitnessModel(ABC):
    """the abstract base class for all sequence fitness models."""

    _available_models: ClassVar[dict[str, SequenceFitnessModel]] = {}

    def __init__(
        self,
        model_checkpoint: str | None = None,
        model_type: str | None = None,
    ) -> None:
        """Instantiate a sequence fitness model.

        Specific sequence fitness models will subclass this or a child
        of it

        Args:
            model_checkpoint (str | None, optional): filepath to a model to load.
                Defaults to None
            model_type (str | None, optional): type of model to load. Defaults to None.

        """
        self.model_checkpoint = model_checkpoint
        self.model_type = model_type

    @abstractmethod
    def predict_fitnesses(
        self,
        sequences: list[str],
        wt_sequence: str | None = None,
        batch_size: int | None = None,
    ) -> list[float]:
        """Given a list of sequences, returns predicted fitness values for each sequence.

        Args:
            sequences (List[str]): List of input sequences to predict fitness values for
            wt_sequence (Union[str, None], optional): The wild-type sequence to compare
                against. Defaults to None.
            If passed in, fitness values are relative to the wild-type sequence
            batch_size (Union[int, None], optional): The batch size to use when
                predicting fitness values. Defaults to None, meaning a single batch of
                all data.

        Returns:
            list[float]: List of fitness values for each sequence

        """


class ProbabilitySequenceFitnessModel(SequenceFitnessModel):
    """SequenceFitnessModels that produce logprobs as fitness predictions."""

    def predict_fitnesses(  # noqa: D102 docstrings should be inherited when generating documentation
        self,
        sequences: list[str],
        wt_sequence: str | None = None,
        batch_size: int | None = None,
    ) -> list[float]:
        return self.predict_logprobs(sequences, wt_sequence, batch_size=batch_size)

    @abstractmethod
    def predict_logprobs(
        self,
        sequences: list[str],
        wt_sequence: str | None = None,
        batch_size: int | None = None,
    ) -> list[float]:
        """Given a list of sequences, produce the logprob estimates for each sequence.

        If the wild type sequence is passed in, returned values are the log-ratio
            between the mutated and wild type sequence probability

        Args:
            sequences (List[str]): list of sequences to predict probabilities for
            wt_sequence (Union[str, None], optional): wild-type sequence to compute
                probabilities relative to. Defaults to None.
            batch_size (Union[int, None], optional): batch size to use when predicting
                probabilities. Defaults to None, meaning a single batch of all data.

        Returns:
            List[float]: list of predicted log-probabilities for each sequence

        """


class ProteinLanguageModel(torch.nn.Module, ProbabilitySequenceFitnessModel):
    """Base class for all protein language models."""

    def __init__(
        self,
        model_checkpoint: str | None = None,
        model_type: str | None = None,
        **kwargs,  # noqa: ANN003
    ):
        torch.nn.Module.__init__(self, **kwargs)
        ProbabilitySequenceFitnessModel.__init__(self, model_checkpoint, model_type)

    @property
    @abstractmethod
    def base_model(self) -> torch.nn.Module:
        """Return the underlying model."""

    @abstractmethod
    def predict_position_logprobs(
        self,
        sequences: list[str],
        wt_sequence: str | None = None,
        batch_size: str | None = None,
    ) -> list[np.ndarray]:
        """Predicts the per-position log probabilities for a given list of sequences.

        Args:
            sequences (List[str]): The list of sequences for which to predict the
                position log probabilities.
            wt_sequence (Union[str, None], optional): The wild-type sequence to compare
                against. Defaults to None.
            batch_size (Union[int, None], optional): The batch size to use when
                predicting fitness values. Defaults to None, meaning a single batch of
                all data.

        Returns:
            fitnesses (List[np.ndarray]): A list of numpy arrays representing the
                position log probabilities for each sequence.
                Each ndarray is shape (seq_len, vocab_size)

        """

    @abstractmethod
    def get_embeddings(
        self,
        sequences: list[str],
        layers: tuple[int] = (-1,),
        batch_size: int | None = None,
    ) -> dict[int, torch.Tensor]:
        """Return embeddings for the given sequences at the given layers.

        Args:
            sequences (Union[List[str], torch.Tensor]): list of sequences to get
                embeddings for
            layers (tuple[int], optional): Indices of layers to get embeddings for.
                Defaults to -1.
            batch_size (Union[int, None], optional): The batch size to use when
                computing embeddings. Defaults to None, meaning a single batch
                of all data.

        Returns:
            dict[int,torch.Tensor]: dictionary of layer indexes and their
                corresponding embeddings

        """

    @abstractmethod
    def get_embed_dim(self, layers: tuple[int] = (-1,)) -> list[int]:
        """Return a list of the dimension of the embeddings for the specified layers.

        Used in finetuning to initialize head layer on top of embeddings

        Args:
            layers (tuple[int], optional): Name of layer to get embeddings for.
                Defaults to -1, last layer.

        Returns:
            List[int]: dimension of the embeddings

        """

    @abstractmethod
    def set_trainable_layers(self, layers: tuple[int] = (-1,)) -> None:
        """Set the specified layers to be trainable in the model.

        Args:
            layers (List[int], optional): index of layers to set as trainable for.
                Defaults to -1, last layer.

        """


class AlignmentMixIn:
    """MixIn class to add alignments to a model."""

    def __init__(
        self,
        alignment: Alignment | None = None,
        alignment_config: AlignmentConfig | None = None,
    ) -> None:
        """Initialize an AlignmentMixIn.

        Args:
            alignment (Alignment | None, optional): Can pass an alignment object
                directly. Defaults to None.
            alignment_config (AlignmentConfig | None, optional): Can pass an alignment
                config that will be used to load the alignment. Defaults to None.

        Raises:
            ValueError: if both alignment and alignment_config are None

        """
        if alignment is not None:
            self.alignment_file = alignment.alignment_file
            self.alignment = alignment
        else:
            if alignment_config is None:
                msg = "Must pass in alignment or alignment_config"
                raise ValueError(msg)
            alignment_file = alignment_config.alignment_file
            self.alignment = Alignment(alignment_config)
            self.alignment_file = alignment_file


class AlignmentProbabilityModel(AlignmentMixIn, ProbabilitySequenceFitnessModel):
    """Superclass for all alignment probability models."""

    def __init__(
        self,
        alignment: Alignment | None = None,
        alignment_config: AlignmentConfig | None = None,
        model_name: str | None = None,
        model_type: str | None = None,
    ):
        AlignmentMixIn.__init__(self, alignment, alignment_config)
        ProbabilitySequenceFitnessModel.__init__(self, model_name, model_type)
