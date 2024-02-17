import numpy as np
from typing import Union, List
from .generic_models import SequenceFitnessModel
from .generic_models import AlignmentModel
from ..utils.alignment import Alignment


class SequenceFitnessModelEnsemble:
    """
    This class represents an naive ensemble of sequence fitness models. It scores a list of sequences on each model
    and then averages the scores together to produce the final fitness value. 

    TODOs: would be nice to have an option for specifying separate alignments for each model. Could also adjust alignment logic to allow
    for alignment parameters to be passed into the model and combined into an Alignment object there, rather than requiring it be specified up front. 
    """
    def __init__(self, model_configs: List[dict], alignment_file: Union[str, None] = None, model_names: Union[List[str], None] = None):
        if alignment_file is not None:
            self.alignment = Alignment(alignment_file)
        self.model_configs = model_configs
        self.models = []
        for i, config in enumerate(self.model_configs):
            model_type = config["model_type"]
            model_constructor = SequenceFitnessModel.get_model(model_type)
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
        """Predicts the fitness for a list of sequences. If wild type sequence is passed in, then fitnesses are relative to the wild type sequence.

        Args:
            sequences (List[str]): List of sequences to predict fitnesses for
            wt_sequence (Union[str, None], optional): wild type sequence to compute fitnesses relative to. Defaults to None.
            standardize (bool, optional): Whether to standardize fitnesses by subtracting mean and dividing by standard deviation prior to averaging. Defaults to False.

        Returns:
            List[float]: List of predicted fitnesses
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
