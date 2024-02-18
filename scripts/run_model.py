"""
Basic script to run a fitness model on a set of mutations 
"""

import argparse
import json
import os

import pandas as pd
from scipy.stats import spearmanr
import warnings
from proteingym.utils.scoring_utils import get_mutations
from proteingym.wrappers.generic_models import SequenceFitnessModel, AlignmentModel

def parse_config(top_config: dict, config_file_key:str="config_file", alignment_config_key:str="alignment_config_file"):
    """parses additional configuration parameters from files specified with the top-level config file. 
    This allows nesting one config file inside another. e.g. if you want to run all ESM2 models and 
    just change the checkpoints, you can make one default ESM2 config, and then add it in the "config_file"
    field of each model-specific config you make. That way all the models will use the same parameters aside 
    from the model checkpoint, without rewriting them many times. 

    Args:
        top_config (dict): Dictionary from configuration file passed in to script 
        config_file_key (str, optional): the key to look for a nested config file. Defaults to "config_file".
        alignment_config_key (str, optional): the key to look for a nested config file for alignment paratmers. Defaults to "alignment_config_file".

    Returns:
        top_config: the top-level configuration with updated kwargs from the nested config files
    """
    if config_file_key in top_config:
        print("Parsing additional config parameters from file: " + top_config[config_file_key])
        with open(top_config[config_file_key], encoding="utf8") as config_file:
            config = json.load(config_file)
        # TODO: Might be nice to allow recursively chaining config files together, but that gets messy/may be worth using some library to manage configs instead
        # if config_key in config:
        #     config = self.parse_config(config)
        if config_file_key in config:
            warnings.warn(
                "Nesting multiple config files is not supported, using only the top level config file"
            )
        for key, value in config.items():
            if key in top_config:
                continue  # Skip if the key is already in kwargs so we don't override anything
            top_config[key] = value
        del top_config[config_file_key]

    if alignment_config_key in top_config:
        print("Parsing additional alignment config parameters from file: " + top_config[alignment_config_key])
        with open(top_config[alignment_config_key], encoding="utf8") as config_file:
            config = json.load(config_file)
        if "alignment_kwargs" not in top_config:
            top_config["alignment_kwargs"] = {}
        for key, value in config.items():
            if key in top_config["alignment_kwargs"]:
                continue  # Skip if the key is already in kwargs so we don't override anything
            top_config["alignment_kwargs"][key] = value
        del top_config[alignment_config_key]
    return top_config

def add_alignment_config(
    config: dict, args: argparse.Namespace, ref_df: pd.DataFrame
) -> dict:
    """Finalizes model config by adding alignment and weights files if necessary

    Args:
        config (dict): original model config
        args (argparse.Namespace): parsed arguments from command line
        ref_df (pd.DataFrame): reference dataframe for ProteinGym datasets
    Returns:
        config (dict): finalized model config
    """
    if args.alignment_folder != "":
        alignment_file = (
            args.alignment_folder
            + os.sep
            + ref_df["MSA_filename"][args.experiment_index]
        )
        if "alignment_kwargs" in config:
            config["alignment_kwargs"]["alignment_file"] = alignment_file
        else:
            config["alignment_kwargs"] = {"alignment_file": alignment_file}
    if args.weights_folder != "":
        weights_file = (
            args.weights_folder
            + os.sep
            + ref_df["weight_file_name"][args.experiment_index]
        )
        if "alignment_kwargs" in config:
            config["alignment_kwargs"]["weights_file"] = weights_file
        else:
            config["alignment_kwargs"] = {"weights_file": weights_file}
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a fitness model on a set of mutations"
    )
    parser.add_argument("--data_folder", type=str, default="")
    parser.add_argument("--alignment_folder", type=str, default="")
    parser.add_argument("--weights_folder", type=str, default="")
    parser.add_argument("--reference_file", type=str, default="")
    parser.add_argument("--experiment_index", type=int)
    parser.add_argument("--model_config", type=str, help="model config file")
    args = parser.parse_args()

    ref_df = pd.read_csv(args.reference_file)
    mut_file = args.data_folder + os.sep + ref_df["DMS_filename"][args.experiment_index]
    model_config = json.load(open(args.model_config))
    model_config = parse_config(model_config)
    model_type = model_config["model_type"]
    print("Config file: " + args.model_config)
    model_constructor = SequenceFitnessModel.get_model(model_type)
    # TODO: look into linting error where model_constructor is an incorrect parameter type. Probably due to underspecified type hints
    if issubclass(model_constructor, AlignmentModel):
        model_config = add_alignment_config(model_config, args, ref_df)
    model = model_constructor(**model_config)
    mutations = get_mutations(
        mut_file, str(ref_df["target_seq"][args.experiment_index])
    )
    logprobs = model.predict_fitnesses(
        mutations["mutated_sequence"].values.tolist(),
        ref_df["target_seq"][args.experiment_index],
    )
    # print(logprobs)
    mutations["predictions"] = logprobs
    # print(mutations)
    print(spearmanr(mutations["DMS_score"], mutations["predictions"]))
