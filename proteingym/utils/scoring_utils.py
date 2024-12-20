"""Contains functions for scoring  mutations on fitness models."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from Bio import SeqUtils

from proteingym.constants.data import AA_INTEGER_MAP


def set_random_seeds(
    seed: int = 0,
    torch_seed: int | None = None,
    np_seed: int | None = None,
    random_seed: int | None = None,
) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed (int, default 0): The seed for the random number generator
        torch_seed (int, optional): The seed for the torch random number generator
        np_seed (int, optional): The seed for the numpy random number generator
        random_seed (int, optional): The seed for the python random number generator

    Returns:
        None

    """
    torch.manual_seed(torch_seed if torch_seed is not None else seed)
    np.random.seed(np_seed if np_seed is not None else seed)  # noqa: NPY002, keeping in case people use older numpy.random calls
    random.seed(random_seed if random_seed is not None else seed)


def get_mutations(
    mutation_file: str,
    target_seq: str = "",
    mutant_delim: str = ":",
) -> pd.DataFrame:
    """Read in mutations from a csv.

    Args:
        mutation_file (str): The path to the mutation file.
        target_seq (str): Target sequence, used to generate mutations from mutant column
            if mutated sequence is not present
        mutant_delim (str): Delimiter for mutant strings, e.g. A26G:G37T for default ":"
    Returns:
        mutations (pd.DataFrame): A dataframe containing the mutations, with a mutant
            column and mutated sequence column

    """
    mutations: pd.DataFrame = pd.read_csv(mutation_file)
    if "mutated_sequence" not in mutations.columns:
        if "mutant" in mutations.columns:
            if target_seq == "":
                msg = "target_seq must be provided if mutated_sequence is not present"
                raise ValueError(msg)
            mutations["mutated_sequence"] = get_mutated_sequences(
                mutations,
                target_seq,
                mutant_delim=mutant_delim,
            )
        else:
            msg = "mutant column must be present if mutated_sequence is not present"
            raise ValueError(msg)
    elif "mutant" not in mutations.columns:
        mutations["mutant"] = get_mutants(
            mutations,
            target_seq,
            mutant_delim=mutant_delim,
        )
    return mutations


def get_mutated_sequences(
    mutations: pd.DataFrame,
    target_seq: str,
    mutant_delim: str = ":",
) -> list[str]:
    """Get mutated sequences from mutations dataframe.

    Args:
        mutations (pd.DataFrame): The dataframe containing the mutations.
        target_seq (str): The target sequence to mutate
        mutant_delim (str): Delimiter for mutant strings, e.g. A26G:G37T for default ":"

    Returns:
        mutated_sequences (List[str]): A series containing the mutated sequences.

    """
    mutants = mutations["mutant"]
    mutated_sequences = []
    for mutant in mutants:
        if is_indel(mutant):
            # assumes indels are in hgvsp format
            mutated_sequences.append(get_indel_sequence(mutant, target_seq))
        else:
            indiv_mutants = mutant.split(mutant_delim)
            mut_seq = target_seq
            for indiv_mutant in indiv_mutants:
                orig_aa, pos, mut_aa = (
                    indiv_mutant[0],
                    indiv_mutant[1:-1],
                    indiv_mutant[-1],
                )
                if not pos.isnumeric():
                    msg = "mutants must be in single letter triplet form, e.g. A23M"
                    raise ValueError(msg)
                if orig_aa not in AA_INTEGER_MAP:
                    msg = f"{orig_aa} is not a valid amino acid"
                    raise ValueError(msg)
                if mut_aa not in AA_INTEGER_MAP:
                    msg = f"{mut_aa} is not a valid amino acid"
                    raise ValueError(msg)
                if orig_aa != target_seq[int(pos) - 1]:
                    msg = (
                        f"original amino acid {orig_aa} at position {pos} does not"
                        "match target sequence amino acid {target_seq[int(pos)-1]}"
                    )
                    raise ValueError(msg)
                pos = int(pos)
                mut_seq = mut_seq[: int(pos) - 1] + mut_aa + mut_seq[int(pos) :]
            mutated_sequences.append(mut_seq)
    return mutated_sequences


def get_mutants(
    mutations: pd.DataFrame,
    target_seq: str,
    mutant_delim: str = ":",
) -> list[str]:
    """Get mutants from mutations dataframe.

    Args:
        mutations (pd.DataFrame): The dataframe containing the mutations.
        target_seq (str): The target sequence.
        mutant_delim (str, default ':'): The delimiter used to separate individual
            mutations.

    Returns:
        mutants (List[str]): A series containing the mutants.

    """
    mutants = []
    for mutated_sequence in mutations["mutated_sequence"]:
        indiv_mutants = []
        if len(mutated_sequence) != len(target_seq):
            # currently using a hack to get indel mutants
            # just treating the whole mutated sequence as a delins that replaces
            # the wild type
            # it's gross, but valid HGVSp, and saves the trouble of finding the
            # most parsimonious representation
            # TODO (@DanielR): Find a better way to get indel mutants if possible
            mutants.append(
                "p."
                + target_seq[0]
                + "1"
                + "_"
                + target_seq[-1]
                + str(len(target_seq))
                + "delins"
                + mutated_sequence,
            )
        else:
            for i, char in enumerate(mutated_sequence):
                if char != target_seq[i]:
                    indiv_mutants.append(target_seq[i] + str(i + 1) + char)
            # if mutated sequence is same as target sequence, add synonomous mutation
            # at first position, just to have something
            if len(indiv_mutants) == 0:
                indiv_mutants = [target_seq[0] + str(1) + target_seq[0]]
            mutants.append(mutant_delim.join(indiv_mutants))
    return mutants


def is_indel(mutant: str) -> bool:
    """Check if a mutation is an indel.

    Args:
        mutant (str): The HGVSp string describing the mutation.

    Returns:
        bool: True if the mutation is an indel, False otherwise.

    """
    indel_strs = ["del", "ins", "dup"]
    return any(indel_str in mutant for indel_str in indel_strs)


def get_indel_sequence(mutant: str, target_seq: str) -> str:
    """Get the mutated sequence for an indel.

    Args:
        mutant (str): The HGVSp string describing the mutation.
        target_seq (str): The target sequence.

    Raises:
        ValueError: If the mutation is not an indel.

    Returns:
        str: The mutated sequence

    """
    parsed_mutant = parse_hgvsp(mutant)
    if parsed_mutant["mutation_type"] == "del":
        return (
            target_seq[: int(parsed_mutant["orig_seq_start"]) - 1]
            + target_seq[int(parsed_mutant["orig_seq_end"]) :]
        )
    if parsed_mutant["mutation_type"] == "ins":
        return (
            target_seq[: int(parsed_mutant["orig_seq_start"])]
            + parsed_mutant["ins_seq"]
            + target_seq[int(parsed_mutant["orig_seq_start"]) :]
        )
    if parsed_mutant["mutation_type"] == "dup":
        dup_seq = target_seq[
            int(parsed_mutant["orig_seq_start"]) - 1 : int(
                parsed_mutant["orig_seq_end"],
            )
        ]
        return (
            target_seq[: int(parsed_mutant["orig_seq_start"])]
            + dup_seq
            + target_seq[int(parsed_mutant["orig_seq_start"]) :]
        )
    if parsed_mutant["mutation_type"] == "delins":
        return (
            target_seq[: int(parsed_mutant["orig_seq_start"]) - 1]
            + parsed_mutant["ins_seq"]
            + target_seq[int(parsed_mutant["orig_seq_end"]) :]
        )
    msg = (
        f"Mutation type {parsed_mutant['mutation_type']} is not supported."
        "Should never get here"
    )
    raise ValueError(msg)


def parse_hgvsp(hgvsp: str) -> dict[str, str]:
    """Parse HGVSp string into dictionary of AAs and positions.

    Args:
        hgvsp (str): The HGVSp string

    Raises:
        ValueError: If the HGVSp string is not valid

    Returns:
        dict[str, str]: Dictionary of start and end positions and AAs for mutation

    """
    component_dict = {}
    if hgvsp.startswith("p."):
        hgvsp = hgvsp[2:]
    elif hgvsp.startswith("p.("):
        hgvsp = hgvsp[3:-1]
    hgvsp = replace_three_letter_aas(hgvsp)
    if "delins" in hgvsp:
        orig_seq, ins_seq = hgvsp.split("delins")
        component_dict["mutation_type"] = "delins"
        orig_seq_start, orig_seq_end = orig_seq.split("_")
    elif "del" in hgvsp:
        del_region = hgvsp[:-3]
        orig_seq_start, orig_seq_end = del_region.split("_")
        component_dict["mutation_type"] = "del"
        ins_seq = ""
    elif "ins" in hgvsp:
        orig_seq, ins_seq = hgvsp.split("ins")
        component_dict["mutation_type"] = "ins"
        orig_seq_start, orig_seq_end = orig_seq.split("_")
    elif "dup" in hgvsp:
        dup_region = hgvsp[:-3]
        orig_seq_start, orig_seq_end = dup_region.split("_")
        component_dict["mutation_type"] = "dup"
        ins_seq = ""
    else:
        msg = f"Invalid HGVSp string: {hgvsp}"
        raise ValueError(msg)
    component_dict["orig_seq_start_aa"] = orig_seq_start[0]
    component_dict["orig_seq_start_pos"] = orig_seq_start[1:]
    component_dict["orig_seq_end_aa"] = orig_seq_end[0]
    component_dict["orig_seq_end_pos"] = orig_seq_end[1:]
    component_dict["ins_seq"] = ins_seq
    return component_dict


def replace_three_letter_aas(mutant: str) -> str:
    """Replace three letter AAs with single letter AAs.

    Args:
        mutant (str): The mutant string or HGVSp

    Returns:
        str: The mutant string with single letter AAs replacing three letter AAs

    """
    for key, value in SeqUtils.IUPACData.protein_letters_3to1.items():
        mutant = mutant.replace(key, value)
    return mutant


# TODO (@DanielR): Confirm that the below function is getting used correctly. e.g. in
# ESMModel the seq_len_wo_special parameter
# is passed as seq_len + 2, which seems like it includes the special tokens.


def get_optimal_window(
    mutation_position_relative: int,
    seq_len_wo_special: int,
    model_window: int,
) -> list[int]:
    """Get optimal window for scoring.

    Determine the section of the sequence to score based on the mutation position,
    sequence length,and model context length.

    Args:
        mutation_position_relative (int): The position of the mutation in the sequence
        seq_len_wo_special (int): The length of the sequence without special tokens
        model_window (int): The length of the model context window

    Returns:
        list[int]: The start and end indices of the section of the sequence to score

    """
    half_model_window = model_window // 2
    if seq_len_wo_special <= model_window:
        return [0, seq_len_wo_special]
    if mutation_position_relative < half_model_window:
        return [0, model_window]
    if mutation_position_relative >= seq_len_wo_special - half_model_window:
        return [seq_len_wo_special - model_window, seq_len_wo_special]
    return [
        max(0, mutation_position_relative - half_model_window),
        min(
            seq_len_wo_special,
            mutation_position_relative + half_model_window,
        ),
    ]
