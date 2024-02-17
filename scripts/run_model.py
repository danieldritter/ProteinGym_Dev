"""
Basic script to run a fitness model on a set of mutations 
"""
from proteingym.wrappers.generic_models import SequenceFitnessModel
from proteingym.utils.scoring_utils import get_mutations
import os
import pandas as pd
import json
from scipy.stats import spearmanr
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a fitness model on a set of mutations")
    parser.add_argument("--data_folder", type=str, default="")
    parser.add_argument("--alignment_folder", type=str, default="")
    parser.add_argument("--weights_folder", type=str, default="")
    parser.add_argument("--reference_file", type=str, default="")
    parser.add_argument("--experiment_index", type=int)
    parser.add_argument("--model_config", type=str, help="model config file")
    args = parser.parse_args()

    ref_df = pd.read_csv(args.reference_file)
    mut_file = args.data_folder + os.sep + \
        ref_df["DMS_filename"][args.experiment_index]
    model_config = json.load(open(args.model_config))
    model_type = model_config["model_type"]
    # Ideally wouldn't have to special case this, but the alignment models generally
    # need the alignment up front, but the alignments also differ for each dataset, so that information shouldn't be in the model config json file
    # I think this is also fixed by passing **kwargs to top level class
    alignment_file = args.alignment_folder + os.sep + \
        ref_df["MSA_filename"][args.experiment_index]
    if args.weights_folder != "":
        weights_file = args.weights_folder + os.sep + \
            ref_df["weight_file_name"][args.experiment_index]
        if "alignment_kwargs" in model_config:
            model_config["alignment_kwargs"]["weights_file"] = weights_file
        else:
            model_config["alignment_kwargs"] = {
                "weights_file": weights_file}
    model_config["alignment_file"] = alignment_file
    model_config["weights_file"] = weights_file
    model = SequenceFitnessModel.get_model(model_type)(**model_config)
    mutations = get_mutations(mut_file, str(
        ref_df["target_seq"][args.experiment_index]))
    logprobs = model.predict_fitnesses(
        mutations["mutated_sequence"].values.tolist(), ref_df["target_seq"][args.experiment_index])
    print(logprobs)
    mutations["predictions"] = logprobs
    print(mutations)
    print(spearmanr(mutations["DMS_score"], mutations["predictions"]))
