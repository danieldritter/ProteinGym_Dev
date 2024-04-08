"""
Basic script to run a fitness model on a set of mutations 
"""
import argparse
import os
import pandas as pd
from scipy.stats import spearmanr
import warnings
from omegaconf import DictConfig, OmegaConf,open_dict
import hydra
from proteingym.utils.scoring_utils import get_mutations
from proteingym.wrappers.generic_models import SequenceFitnessModel, AlignmentModel

# TODO: Fix config scheme so it can fit in a folder, but still be overridden and not require removing "zero_shot_configs" key
@hydra.main(version_base=None, config_path=f"{os.path.dirname(os.path.dirname(__file__))}/configs", config_name="zero_shot_configs/default_config")
def main(config: DictConfig):
    config = config["zero_shot_configs"]
    ref_df = pd.read_csv(config.reference_file)
    mut_file = config.data_folder + os.sep + ref_df["DMS_filename"][config.experiment_index]
    mutations = get_mutations(mut_file, str(ref_df["target_seq"][config.experiment_index]))
    if "alignment" in config:
        if "MSA_filename" in ref_df.columns:
            assert "alignment_folder" in config, "Must provide alignment_folder in config if MSA_filename is provided"
            config.alignment["alignment_file"] = config.alignment_folder + os.sep + ref_df["MSA_filename"][config.experiment_index]
        if "weights_file_name" in ref_df.columns:
            assert "weights_folder" in config, "Must provide weights_folder in config if weight_file_name is provided"
            with open_dict(config.alignment):
                config.alignment["weights_file"] = config.weights_folder + os.sep + ref_df["weight_file_name"][config.experiment_index]
        model = hydra.utils.instantiate(config.model, alignment_kwargs=config.alignment)
    else:
        model = hydra.utils.instantiate(config.model)
    logprobs = model.predict_fitnesses(
        mutations["mutated_sequence"].values.tolist(),
        ref_df["target_seq"][config.experiment_index]
    )
    print(spearmanr(mutations["DMS_score"], logprobs))

if __name__ == "__main__":
    main()