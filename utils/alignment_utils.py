"""
General utility functions for working with multiple sequence alignments
"""
from typing import List
from Bio import SeqIO


def read_fasta(fasta_file: str) -> List[str]:
    """
    Read a fasta file and yield a list of sequences
    """
    return list([seq.seq for seq in SeqIO.parse(fasta_file, "fasta")])


def get_focus_cols(alignment: List[str], focus_seq_ind: int = 0) -> List[int]:
    """
    Get the columns of the alignment that are in the focus sequence
    """
    focus_cols = []
    for i, aa in enumerate(alignment[focus_seq_ind]):
        if aa != "-":
            focus_cols.append(i)
    return focus_cols


def get_focus_alignment(alignment: List[str], focus_seq_ind: int = 0) -> List[str]:
    """
    Get the alignment with only the columns in the focus sequence
    """
    focus_cols = get_focus_cols(alignment, focus_seq_ind)
    focus_alignment = []
    for seq in alignment:
        focus_alignment.append("".join([seq[i] for i in focus_cols]))
    return focus_alignment
