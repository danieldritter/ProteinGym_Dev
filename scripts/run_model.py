"""Basic script to run a fitness model on a set of mutations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import pandas as pd
import torch
from scipy.stats import spearmanr

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from proteingym.utils.alignment import AlignmentConfig
    from proteingym.wrappers.generic_models import SequenceFitnessModel

from proteingym.utils.scoring_utils import get_mutations, set_random_seeds

parent_name = str(Path(__file__).resolve().parent.parent)


def prepare_alignment_config(
    config: DictConfig,
    ref_df: pd.DataFrame,
    experiment_index: int,
) -> AlignmentConfig | None:
    """Prepare alignment configuration if needed."""
    alignment_config = None

    if "alignment" not in config:
        return alignment_config

    alignment_config = hydra.utils.instantiate(config.alignment)
    print(alignment_config)
    print(type(alignment_config))
    # Handle MSA filename
    if "MSA_filename" in ref_df.columns:
        if "alignment_folder" not in config:
            msg = "Must provide alignment_folder in config if MSA_filename is provided"
            raise ValueError(msg)
        alignment_config.alignment_file = (
            config.alignment_folder + os.sep + ref_df["MSA_filename"][experiment_index]
        )

    # Handle weight filename
    if "weight_file_name" in ref_df.columns:
        if "weights_folder" not in config:
            msg = (
                "Must provide weights_folder in config if weight_file_name is provided"
            )
            raise ValueError(msg)
        alignment_config.weights_file = (
            config.weights_folder
            + os.sep
            + ref_df["weight_file_name"][experiment_index]
        )

    return alignment_config


def instantiate_model(
    config: DictConfig,
    alignment_config: AlignmentConfig | None = None,
) -> SequenceFitnessModel:
    """Instantiate model with alignment if appropriate."""
    # First, get the model class without instantiating
    model_class = hydra.utils.get_class(config.model._target_)

    # Check if model is a subclass of AlignmentMixIn
    if hasattr(model_class, "__mro__"):
        # Look for AlignmentMixIn in the method resolution order
        is_alignment_model = any(
            base.__name__ == "AlignmentMixIn" for base in model_class.__mro__
        )

        # Only pass alignment_kwargs if it's an alignment model
        if is_alignment_model:
            if alignment_config is None:
                msg = "Must provide alignment_config if model subclasses AlignmentMixIn"
                raise ValueError(msg)
            return hydra.utils.instantiate(
                config.model,
                alignment_config=alignment_config,
            )

    # Default instantiation without alignment
    return hydra.utils.instantiate(config.model)


@hydra.main(
    version_base=None,
    config_path=f"{parent_name}/configs",
    config_name="default_zero_shot_config",
)
def main(config: DictConfig) -> None:
    """Sample script to run a model and compute spearman."""
    set_random_seeds(config.random_seed)
    ref_df = pd.read_csv(config.reference_file)
    mut_file = (
        config.data_folder + os.sep + ref_df["DMS_filename"][config.experiment_index]
    )
    mutations = get_mutations(
        mut_file,
        str(ref_df["target_seq"][config.experiment_index]),
    )
    # Prepare alignment configuration if needed
    alignment_config = prepare_alignment_config(config, ref_df, config.experiment_index)

    # Instantiate model with conditional alignment
    model = instantiate_model(config, alignment_config)
    if isinstance(model, torch.nn.Module) and "device" in config:
        model = model.to(config.device)
    logprobs = model.predict_fitnesses(
        mutations["mutated_sequence"].to_numpy().tolist(),
        str(ref_df["target_seq"][config.experiment_index]),
        batch_size=config.batch_size,
    )
    print(spearmanr(mutations["DMS_score"], logprobs))


if __name__ == "__main__":
    main()
