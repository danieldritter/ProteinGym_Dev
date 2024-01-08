"""
Basic script to run a fitness model ensemble on a set of mutations 
"""
import argparse
import json
import pandas as pd
import os
from proteingym.utils.scoring_utils import get_mutations
from proteingym.wrappers.model_ensembles import SequenceFitnessModelEnsemble

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a fitness model on a set of mutations")
    parser.add_argument("--data_folder", type=str, default="")
    parser.add_argument("--alignment_folder", type=str, default=None)
    parser.add_argument("--reference_file", type=str, default="")
    parser.add_argument("--experiment_index", type=int)
    parser.add_argument("--model_configs", type=str,
                        nargs="+", help="list of model configs")
    parser.add_argument("--model_types", type=str,
                        nargs="+", help="types for each model to use")
    args = parser.parse_args()

    ref_df = pd.read_csv(args.reference_file)
    mut_file = args.data_folder + os.sep + \
        ref_df["DMS_filename"][args.experiment_index]
    print(f"Scoring {ref_df['DMS_id'][args.experiment_index]}")

    # assumes that models are in the same order as model_configs
    # also uses the filenames of the config files as unique model names
    configs = []
    model_names = []
    for i, name in enumerate(args.model_types):
        configs.append(json.load(open(args.model_configs[i])))
        model_names = os.path.splitext(
            os.path.basename(args.model_configs[i]))[0]
    if args.alignment_folder is None:
        alignment_file = None
    else:
        alignment_file = args.alignment_folder + os.sep + \
            ref_df["MSA_filename"][args.experiment_index]
    model = SequenceFitnessModelEnsemble(
        model_types=args.model_types, model_configs=configs, alignment_file=alignment_file)

    mutations = get_mutations(mut_file, str(
        ref_df["target_seq"][args.experiment_index]))
    logprobs = model.predict_fitnesses(
        mutations["mutated_sequence"].values.tolist(), str(ref_df["target_seq"][args.experiment_index]), standardize=True)
    mutations["predictions"] = logprobs
    print(mutations)
