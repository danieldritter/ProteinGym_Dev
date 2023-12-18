"""
Base class for all autoregressive models
"""
from .generic_models import ProbabilitySequenceFitnessModel


class AutoregressiveProteinLanguageModel(ProbabilitySequenceFitnessModel):
    """
    Base class for all autoregressive models
    """

    def __init__(self, model_name=""):
        """
        Initializes a new instance of the class.

        Args:
            model_name (str): The name of the model. Defaults to an empty string.
        """
        super().__init__(model_name)

    def predict_logprobs(self, sequences):
        pass

    def predict_fitnesses(self, sequences, wt_sequence=None):
        pass

    def predict_position_logprobs(self, sequences):
        pass
