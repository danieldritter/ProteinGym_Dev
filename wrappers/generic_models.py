"""
Generic model classes 
"""
from abc import ABC, abstractmethod


class SequenceFitnessModel(ABC):
    """
    the abstract base class for all fitness models
    """

    def __init__(self, model_name=""):
        """
        Initializes a new FitnessModel.

        :param model_name: The name of the model (default is an empty string).
        """
        self.model_name = model_name

    @abstractmethod
    def predict_fitnesses(self, sequences, wt_sequence=None):
        """
        Predicts the fitnesses of given sequences.

        Parameters:
        - sequences: a list of sequences to predict the fitnesses for.
        - wt_sequence: the wild-type sequence to compare against (default: None). If given fitnesses 
            are returned as the ratio between the sequence fitness and wild type fitness.

        Returns:
        - fitnesses: a list of fitnesses for the given sequences.
        """
        pass


class ProbabilitySequenceFitnessModel(SequenceFitnessModel):

    def __init__(self, model_name=""):
        super().__init__(model_name)

    def predict_fitnesses(self, sequences, wt_sequence=None):
        return self.predict_logprobs(sequences)

    @abstractmethod
    def predict_logprobs(self, sequences):
        pass
