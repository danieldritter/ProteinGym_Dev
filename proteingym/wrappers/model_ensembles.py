"""Ensemble models combining multiple fitness models."""

from __future__ import annotations

import numpy as np

from .generic_models import SequenceFitnessModel


class SequenceFitnessModelEnsemble:
    """Represents an naive ensemble of sequence fitness models.

    It scores a list of sequences on each model
    and then averages the scores together to produce the final fitness value.

    TODOs: Add option to run each model iteratively, store scores, and then combine,
    for larger models where holding all of them in memory is costly/inefficient
    """

    def __init__(
        self,
        model_configs: list[dict],
        model_names: list[str] | None = None,
    ):
        self.model_configs = model_configs
        self.models = []
        for config in self.model_configs:
            model_constructor = SequenceFitnessModel.get_model(config["model_type"])
            self.models.append(model_constructor(**config))
        if model_names is not None:
            assert len(model_names) == len(
                self.models,
            ), "Must supply same number of model names as models in ensemble"
            self.model_names = model_names
            print(f"Naive Ensemble of {len(self.models)} models: {self.model_names}")

    def predict_fitnesses(
        self,
        sequences: List[str],
        wt_sequence: Union[str, None] = None,
        standardize: bool = False,
    ) -> List[float]:
        """Predicts the fitness for a list of sequences. If wild type sequence is passed in, then fitnesses are relative to the wild type sequence.

        Args:
            sequences (List[str]): List of sequences to predict fitnesses for
            wt_sequence (Union[str, None], optional): wild type sequence to compute fitnesses relative to. Defaults to None.
            standardize (bool, optional): Whether to standardize fitnesses by subtracting mean and dividing by standard deviation prior to averaging. Defaults to False.

        Returns:
            List[float]: List of predicted fitnesses

        """
        all_fitnesses = []
        for i, model in enumerate(self.models):
            print(f"Scoring with {self.model_names[i]}")
            fitnesses = model.predict_fitnesses(sequences, wt_sequence)
            if standardize:
                fitnesses = np.array(fitnesses)
                fitnesses = (fitnesses - np.mean(fitnesses)) / np.std(fitnesses)
            all_fitnesses.append(fitnesses)
        return np.mean(all_fitnesses, axis=0).tolist()
