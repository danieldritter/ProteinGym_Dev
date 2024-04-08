"""
Basic script to run a fitness model ensemble on a set of mutations 
"""

import argparse
import json
import os
from scipy.stats import spearmanr 
import pandas as pd
from scripts.run_model_old import add_alignment_config, parse_config

from proteingym.utils.scoring_utils import get_mutations
from proteingym.wrappers.generic_models import (AlignmentModel,
                                                SequenceFitnessModel)
from proteingym.wrappers.model_ensembles import SequenceFitnessModelEnsemble

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a fitness model on a set of mutations"
    )
    parser.add_argument("--data_folder", type=str, default="")
    parser.add_argument("--alignment_folder", type=str, default="")
    parser.add_argument("--weights_folder", type=str, default="")
    parser.add_argument("--reference_file", type=str, default="")
    parser.add_argument("--experiment_index", type=int)
    parser.add_argument(
        "--model_configs", type=str, nargs="+", help="list of model configs"
    )
    args = parser.parse_args()

    ref_df = pd.read_csv(args.reference_file)
    mut_file = args.data_folder + os.sep + ref_df["DMS_filename"][args.experiment_index]
    print(f"Scoring {ref_df['DMS_id'][args.experiment_index]}")

    # uses the filenames of the config files as unique model names
    configs = []
    model_names = []
    for config in args.model_configs:
        config_dict = json.load(open(config))
        config_dict = parse_config(config_dict)
        # model names are just identifiers for the individual models, defaults to name of the config file
        model_name = os.path.splitext(os.path.basename(config))[0]
        model_constructor = SequenceFitnessModel.get_model(config_dict["model_type"])
        if issubclass(model_constructor, AlignmentModel):
            config_dict = add_alignment_config(config_dict, args, ref_df)
        configs.append(config_dict)
        model_names.append(model_name)
    model = SequenceFitnessModelEnsemble(model_configs=configs, model_names=model_names)
    mutations = get_mutations(
        mut_file, str(ref_df["target_seq"][args.experiment_index])
    )
    logprobs = model.predict_fitnesses(
        mutations["mutated_sequence"].values.tolist(),
        str(ref_df["target_seq"][args.experiment_index]),
        standardize=True,
    )
    mutations["predictions"] = logprobs
    print(spearmanr(mutations["DMS_score"], mutations["predictions"]))
