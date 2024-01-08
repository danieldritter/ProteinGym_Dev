"""
Generic model classes 
"""
from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class SequenceFitnessModel(ABC):
    """
    the abstract base class for all sequence fitness models
    """
    _available_models = {}

    def __init__(self, model_checkpoint: Union[str, None] = None):
        self.model_checkpoint = model_checkpoint

    @classmethod
    def register(cls, model_name: str):
        """
        Registers a new model with the given name.

        :param model_name: The name of the model.
        """
        def inner_decorator(constructor):
            cls._available_models[model_name] = constructor
            return constructor
        return inner_decorator

    @classmethod
    def get_model(cls, model_name: str):
        """
        Returns the constructor for the given model name.

        :param model_name: The name of the model.
        """
        if model_name not in cls._available_models:
            raise ValueError(
                f"Model {model_name} not found. Available models: {list(cls._available_models.keys())}")
        return cls._available_models[model_name]

    @abstractmethod
    def predict_fitnesses(self, sequences: List[str], wt_sequence: Union[str, None] = None):
        """
        Predicts the fitnesses of given sequences.

        Parameters:
        - sequences: a list of sequences to predict the fitnesses for.
        - wt_sequence: the wild-type sequence to compare against (default: None). If given fitnesses 
            are returned as the ratio between the sequence fitness and wild type fitness.

        Returns:
        - fitnesses: a list of fitnesses for the given sequences.
        """


class ProbabilitySequenceFitnessModel(SequenceFitnessModel):
    """
    Class for sequence fitness models that produce log-probabilities as fitness predictions
    """

    def predict_fitnesses(self, sequences: List[str], wt_sequence: Union[str, None] = None) -> List[float]:
        return self.predict_logprobs(sequences, wt_sequence)

    @abstractmethod
    def predict_logprobs(self, sequences, wt_sequence: Union[str, None] = None) -> List[float]:
        """_summary_

        :param sequences: _description_
        :type sequences: _type_
        :param wt_sequence: _description_, defaults to None
        :type wt_sequence: Union[str, None], optional
        :return: _description_
        :rtype: List[float]
        """
        pass
