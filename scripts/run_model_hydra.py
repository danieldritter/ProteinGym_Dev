"""
Basic script to run a fitness model on a set of mutations 
"""
import argparse
import os
import pandas as pd
from scipy.stats import spearmanr
import warnings
from omegaconf import DictConfig, OmegaConf
import hydra
from proteingym.utils.scoring_utils import get_mutations
from proteingym.wrappers.generic_models import SequenceFitnessModel, AlignmentModel


@hydra.main(version_base=None, config_path=f"{os.path.dirname(os.path.dirname(__file__))}/configs", config_name="default_config")
def main(config: DictConfig):
    ref_df = pd.read_csv(config.reference_file)
    mut_file = config.data_folder + os.sep + ref_df["DMS_filename"][config.experiment_index]
    mutations = get_mutations(mut_file, str(ref_df["target_seq"][config.experiment_index]))
    config.alignment["alignment_file"] = config.alignment_folder + os.sep + ref_df["MSA_filename"][config.experiment_index]
    config.alignment["weights_file"] = config.weights_folder + os.sep + ref_df["weight_file_name"][config.experiment_index]
    model = hydra.utils.instantiate(config.model, alignment_kwargs=config.alignment)
    logprobs = model.predict_fitnesses(
        mutations["mutated_sequence"].values.tolist(),
        ref_df["target_seq"][config.experiment_index]
    )
    print(len(logprobs))

if __name__ == "__main__":
    main()