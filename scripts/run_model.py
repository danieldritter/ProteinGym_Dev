"""
Basic script to run a fitness model on a set of mutations 
"""
import argparse
import json
import pandas as pd
import os
from proteingym.utils.scoring_utils import get_mutations
from proteingym.wrappers.generic_models import SequenceFitnessModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a fitness model on a set of mutations")
    parser.add_argument("--data_folder", type=str, default="")
    parser.add_argument("--alignment_folder", type=str, default="")
    parser.add_argument("--reference_file", type=str, default="")
    parser.add_argument("--experiment_index", type=int)
    parser.add_argument("--model_config", type=str, help="model config file")
    parser.add_argument("--model_name", type=str,
                        default="", help="Name of model to use")
    args = parser.parse_args()

    ref_df = pd.read_csv(args.reference_file)
    mut_file = args.data_folder + os.sep + \
        ref_df["DMS_filename"][args.experiment_index]
    model_config = json.load(open(args.model_config))
    if args.model_name == "":
        model_name = os.path.splitext(os.path.basename(args.model_config))[0]
    else:
        model_name = args.model_name
    # Ideally wouldn't have to special case this, but the alignment models generally
    # need the alignment up front, but it differs per dataset, so shouldn't be in the model config json file
    if args.alignment_folder != "":
        alignment_file = args.alignment_folder + os.sep + \
            ref_df["MSA_filename"][args.experiment_index]
        model = SequenceFitnessModel.get_model(model_name)(
            alignment_file=alignment_file, **model_config)
    else:
        model = SequenceFitnessModel.get_model(model_name)(**model_config)
    mutations = get_mutations(mut_file, str(
        ref_df["target_seq"][args.experiment_index]))
    logprobs = model.predict_fitnesses(
        mutations["mutated_sequence"].values.tolist(), ref_df["target_seq"][args.experiment_index])
    print(logprobs)
    mutations["predictions"] = logprobs
    print(mutations)
