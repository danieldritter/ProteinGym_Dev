"""
Generic model classes 
"""
from abc import ABC, abstractmethod
from typing import List, Union, Callable
import json
import numpy as np
from ..utils.alignment import Alignment


class SequenceFitnessModel(ABC):
    """
    the abstract base class for all sequence fitness models
    """
    _available_models = {}

    def __init__(self, model_checkpoint: Union[str, None] = None, model_type: Union[str, None] = None, **kwargs):
        self.model_checkpoint = model_checkpoint
        self.model_type = model_type
        # This is done so that when we multiple inheritance cases or want to pass additional arguments beyond what is necessary (e.g. calling a non-alignment constructor with
        # configs that have alignment info) we don't get an error. All the kwargs that aren't used somewhere prior in the inheritance chain are stored here
        self.unused_kwargs = kwargs

    def parse_config(self, kwargs: dict) -> dict:
        """This function checks if a 'config_file' argument is provided in kwargs, and reads that file and returns it if so. Otherwise it just returns the kwargs

        Args:
            kwargs (dict): dictionary of keyword arguments

        Returns:
            kwargs (dict): dictionary of original keywords plus those included in the config file if provided.  
        """
        if "config_file" in kwargs:
            with open(kwargs["config_file"], encoding="utf8") as config_file:
                config = json.load(config_file)
            for key, value in config.items():
                if key in kwargs:
                    continue  # Skip if the key is already in kwargs so we don't override anything
                kwargs[key] = value
        if "alignment_config_file" in kwargs:
            with open(kwargs["alignment_config_file"], encoding="utf8") as config_file:
                config = json.load(config_file)
            if "alignment_kwargs" not in kwargs:
                kwargs["alignment_kwargs"] = {}
            for key, value in config.items():
                if key in kwargs["alignment_kwargs"]:
                    continue  # Skip if the key is already in kwargs so we don't override anything
                kwargs["alignment_kwargs"][key] = value
        return kwargs

    @classmethod
    def register(cls, model_type: str) -> Callable:
        """This function is used as a decorator on subclasses to automatically add them to a global registry 

        Args:
            model_name (str): name to use for the model in the model registry. 

        Returns:
            inner_decorator (function): decorator function that adds a model name to the class registry dict
        """
        def inner_decorator(constructor):
            cls._available_models[model_type] = constructor
            return constructor
        return inner_decorator

    # TODO: the typing package has a Self option as of python 3.11. Leaving it out for now since I've been using 3.9, but may want to update in the future for this and register function
    @classmethod
    def get_model(cls, model_name: str) -> Callable:
        """This function retrieves a model constructor from the model registry given its name.

        Args:
            model_name (str): the name used to register the model (defined with the register decorator on the model class)

        Raises:
            ValueError: if the model is not found in the registry

        Returns:
            SequenceFitnessModel: class constructor for the model with model_name in the registry
        """
        if model_name not in cls._available_models:
            raise ValueError(
                f"Model {model_name} not found. Available models: {list(cls._available_models.keys())}")
        return cls._available_models[model_name]

    @abstractmethod
    def predict_fitnesses(self, sequences: List[str], wt_sequence: Union[str, None] = None) -> List[float]:
        """Given a list of sequences, returns the predicted fitness values for each sequence. 

        Args:
            sequences (List[str]): List of input sequences to predict fitness values for 
            wt_sequence (Union[str, None], optional): The wild-type sequence to compare against. Defaults to None.
            If passed in, fitness values are relative to the wild-type sequence

        Returns:
            List[float]: List of fitness values for each sequence
        """


class ProbabilitySequenceFitnessModel(SequenceFitnessModel):
    """
    Class for sequence fitness models that produce log-probabilities as fitness predictions
    """

    def predict_fitnesses(self, sequences: List[str], wt_sequence: Union[str, None] = None) -> List[float]:
        return self.predict_logprobs(sequences, wt_sequence)

    @abstractmethod
    def predict_logprobs(self, sequences: List[str], wt_sequence: Union[str, None] = None) -> List[float]:
        """Given a list of sequences, produce the log-probability estimates for each sequence. 
        If the wild type sequence is passed in, returned values are the log-ratio between the mutated and wild type sequence probability

        Args:
            sequences (List[str]): list of sequences to predict probabilities for
            wt_sequence (Union[str, None], optional): wild-type sequence to compute probabilities relative to. Defaults to None.

        Returns:
            List[float]: list of predicted log-probabilities for each sequence
        """


class ProteinLanguageModel(ProbabilitySequenceFitnessModel):
    """
    Base class for all protein language models
    """

    def __init__(self, eval_mode: bool = True, nogpu: bool = False, batch_size: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.eval_mode = eval_mode
        self.nogpu = nogpu
        self.batch_size = batch_size

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


class AlignmentModel(SequenceFitnessModel):
    """
    Superclass for all alignment models
    """

    def __init__(
        self,
        alignment: Union[Alignment, None] = None,
        alignment_file: Union[str, None] = None,
        alignment_kwargs: Union[dict, None] = None,
        alignment_config_file: Union[str, None] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if alignment is not None:
            self.alignment_file = alignment.alignment_file
            self.alignment = alignment
        else:
            if alignment_file is not None:
                if alignment_kwargs is None:
                    alignment_kwargs = {"alignment_file": alignment_file}
                else:
                    alignment_kwargs["alignment_file"] = alignment_file
            else:
                if alignment_kwargs is None or "alignment_file" not in alignment_kwargs:
                    raise ValueError(
                        "Alignment file must be provided directly or in alignment_kwargs")
            self.alignment = Alignment(**alignment_kwargs)
            self.alignment_file = self.alignment.alignment_file
            self.alignment_config_file = alignment_config_file


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
        alignment_file: Union[str, None] = None,
        alignment_kwargs: Union[dict, None] = None,
        alignment_config_file: Union[str, None] = None,
        **kwargs
    ):
        AlignmentModel.__init__(
            self, alignment, alignment_file, alignment_kwargs, alignment_config_file)
        ProbabilitySequenceFitnessModel.__init__(self, **kwargs)
