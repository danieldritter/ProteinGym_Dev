import numpy as np
from typing import Union, List
from .generic_models import SequenceFitnessModel
from .alignment_model import AlignmentModel
from ..utils.alignment import Alignment


class SequenceFitnessModelEnsemble:

    def __init__(self, model_types: List[str], model_configs: List[dict], alignment_file: Union[str, None] = None,
                 model_names: Union[List[str], None] = None):
        if alignment_file is not None:
            self.alignment = Alignment(alignment_file)
        self.models = []
        for i, model in enumerate(model_types):
            model_constructor = SequenceFitnessModel.get_model(model)
            if issubclass(model_constructor, AlignmentModel):
                if alignment_file is None:
                    raise ValueError(
                        "Must supply alignment file if using alignment model")
                self.models.append(model_constructor(
                    alignment=self.alignment, **model_configs[i]))
            else:
                self.models.append(model_constructor(**model_configs[i]))
        if model_names is not None:
            assert len(model_names) == len(
                self.models), "Must supply same number of model names as models in ensemble"
            self.model_names = model_names

    def predict_fitnesses(self, sequences: List[str], wt_sequence: Union[str, None] = None, standardize: bool = False) -> List[float]:
        """ predicts fitnesses of sequences for an ensemble of sequence fitness models, averaging the scores together

        :param sequences: list of sequences to score 
        :type sequences: List[str]
        :param wt_sequence: wild type sequence to compare against, defaults to None
        :type wt_sequence: Union[str, None], optional
        :param standardize: whether to standardize the fitnesses before averaging them together (subtracts mean and divides by standard deviation), defaults to False
        :type standardize: bool, optional
        :return: list of fitnesses, averaged across all models in ensemble 
        :rtype: List[float]
        """
        all_fitnesses = []
        for model in self.models:
            fitnesses = model.predict_fitnesses(sequences, wt_sequence)
            if standardize:
                fitnesses = np.array(fitnesses)
                fitnesses = (fitnesses - np.mean(fitnesses)) / \
                    np.std(fitnesses)
            all_fitnesses.append(fitnesses)
        return np.mean(all_fitnesses, axis=0).tolist()
