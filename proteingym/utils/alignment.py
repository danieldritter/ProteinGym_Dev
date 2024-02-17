import os
from typing import List, Tuple, Union

import numpy as np
from Bio import SeqIO
from proteingym.constants.data import (AA_ALPHABET_ALL, AA_ALPHABET_UPPER, AA_INTEGER_MAP,
                                       AA_INTEGER_MAP_ALL)


class Alignment:
    """Class to handle operations on alignments. Defaults to taking first sequence as focus sequence, keeping only columns where 
    the focus sequence is not a gap, and uppercasing all amino acids. 

    TODOs: Generally efficiency of I/O reading stuff. Adding iterators since currently this will OOM on large alignments. Adding numerical 
    representations of alignments (e.g. assigning indices and converting or one-hot encoding).
    TODOS: Make sure filtering and gap characters apply across alignment formats (I'm guessing they probably don't), and adjust to handle multiple formats if needed
    TODOs: Apply memory efficiency changes we made in proteingym to this class (e.g. lazy loading of one-hot encodings to avoid OOM)
    """

    def __init__(self, alignment_file: str, weights_file: Union[str, None] = None, alignment_format: str = "fasta",
                 focus_seq_ind: int = 0, perc_max_gap_seq_fragments: float = 1.0, perc_max_gap_column_coverage: float = 1.0,
                 uppercase: bool = True, remove_sequences_with_indeterminate_aas: bool = False, alphabet: Union[str, None] = None,
                 one_hot_encode: bool = False, aa_dict: Union[dict, None] = None,
                 weight_theta: float = 0.2):
        self.alignment_file = alignment_file
        self.alignment_format = alignment_format
        self.focus_seq_ind = focus_seq_ind
        self.focus_seq_id, self.focus_seq, self.seq_name_to_sequences = self.read_alignment(
            alignment_file, alignment_format, uppercase)
        self.weights_file = weights_file
        self.perc_max_gap_seq_fragments = perc_max_gap_seq_fragments
        self.perc_max_gap_column_coverage = perc_max_gap_column_coverage
        self.uppercase = uppercase
        self.remove_sequences_with_indeterminate_aas = remove_sequences_with_indeterminate_aas
        self.one_hot_encode = one_hot_encode
        self.weight_theta = weight_theta
        if alphabet is None:
            if self.uppercase:
                self.alphabet = AA_ALPHABET_UPPER
            else:
                self.alphabet = AA_ALPHABET_ALL
        else:
            self.alphabet = alphabet
        if aa_dict is None:
            if self.uppercase:
                self.aa_dict = AA_INTEGER_MAP
            else:
                self.aa_dict = AA_INTEGER_MAP_ALL
        else:
            self.aa_dict = aa_dict
        # Read the alignment file
        if self.focus_seq_ind is not None:
            if self.focus_seq is None or self.focus_seq_id is None:
                raise TypeError(
                    "focus_seq_ind is not None but either focus_seq or focus_seq_id are. This likely means the alignment file does not contain a sequence at index focus_seq_ind")
            # Read alignment file and get focus alignment
            self.focus_cols = self.get_focus_cols(self.focus_seq)
            self.seq_name_to_sequences = self.get_focus_alignment(
                self.seq_name_to_sequences, self.focus_cols)
            self.focus_seq = self.focus_seq.replace("-", "").replace(".", "")
        else:
            self.focus_cols = None
        # Filter sequences
        if perc_max_gap_seq_fragments != 1.0:
            self.seq_name_to_sequences = self.filter_sequence_fragments(
                self.seq_name_to_sequences, perc_max_gap_seq_fragments)
        if perc_max_gap_column_coverage != 1.0:
            self.covered_columns = self.get_covered_columns(
                self.seq_name_to_sequences, perc_max_gap_column_coverage)
        else:
            self.covered_columns = list(
                range(len(list(self.seq_name_to_sequences.values())[0])))
        if remove_sequences_with_indeterminate_aas:
            orig_len = len(self.seq_name_to_sequences)
            self.seq_name_to_sequences = {seq_name: seq for seq_name, seq in self.seq_name_to_sequences.items() if set(
                seq).issubset(set(list(self.alphabet) + ["-", "."]))}
            print(
                f"Removed {orig_len - len(self.seq_name_to_sequences)} sequences with indeterminate amino acids")
        if self.one_hot_encode:
            self.one_hot_alignment = self.one_hot_encode_alignment(
                self.seq_name_to_sequences)
        if self.weights_file is not None:
            if os.path.exists(self.weights_file):
                self.weights = np.load(file=self.weights_file)
            else:
                print(
                    f"Computing sequence weights and saving to {self.weights_file}")
                if not self.one_hot_encode:
                    print(
                        "Cannot compute weights without one-hot encoding, setting one_hot_encode to true and generatign one-hot encoding")
                    self.one_hot_encode = True
                    self.one_hot_alignment = self.one_hot_encode_alignment(
                        self.seq_name_to_sequences)
                self.weights = self.compute_weights(self.one_hot_alignment)
                np.save(file=self.weights_file, arr=self.weights)
            # Iterating over dictionary keys and values returns consistent ordering as long as dictionary isn't altered, so this is fine
            self.seq_name_to_weights = {seq_name: self.weights[i] for i, seq_name in enumerate(
                self.seq_name_to_sequences.keys())}

    def read_alignment(self, alignment_file: str, alignment_format: str, uppercase: bool = True) -> Tuple[Union[str, None], Union[str, None], dict[str, str]]:
        """Reads an alignment file and returns a list of the sequences in the alignment. Defaults to uppercasing. Any format 
        suppporting by BioPython can be used. 

        Args:
            alignment_file (str): filepath of alignment to read
            alignment_format (str): format of alignment, e.g. "a2m", "stockholm", etc. 
            uppercase (bool, optional): Whether to uppercase all amino acids in alignment. Defaults to True.

        Returns:
            Tuple[str, str, dict]: id of focus sequence, focus sequence, and dictionary mapping from sequence id to sequence
        """
        seq_name_to_sequence = {}
        focus_seq = None
        focus_seq_id = None
        for i, record in enumerate(SeqIO.parse(alignment_file, alignment_format)):
            if focus_seq is None:
                if i == self.focus_seq_ind:
                    focus_seq = str(record.seq)
                    focus_seq_id = record.id
            seq_name_to_sequence[record.id] = str(record.seq)
        if uppercase:
            return focus_seq_id, focus_seq, {seq_id: seq.upper().replace(".", "-") for seq_id, seq in seq_name_to_sequence.items()}
        return focus_seq_id, focus_seq, seq_name_to_sequence

    def get_focus_cols(self, focus_seq: str) -> List[int]:
        """Get columns of alignment that are not gaps in the focus sequence. Focus sequence by default is first sequence in alignment.

        Args:
            focus_seq (str): Focus sequence to get columns for
        Returns:
            focus_cols (List[int]): List of column indices that are not gaps in the focus sequence
        """
        focus_cols = []
        for i, aa in enumerate(focus_seq):
            if aa != "-":
                focus_cols.append(i)
        return focus_cols

    def get_focus_alignment(self, alignment: dict[str, str], focus_cols: List[int]) -> dict[str, str]:
        """Removes columns in alignment that are gaps in the focus sequence and returns the alignment. 

        Args:
            alignment (dict[str,str]): dictionary mapping from sequence id to sequence
            focus_cols (List[int]): List of indices of non-gap columns in the focus sequence

        Returns:
            focus_alignment (dict[str,str]): Dictionary mapping from sequence id to sequence, with only focus sequence columns included
        """
        focus_alignment = {}
        for seq_id in alignment:
            focus_alignment[seq_id] = "".join(
                [alignment[seq_id][i] for i in focus_cols])
        return focus_alignment

    def filter_sequence_fragments(self, alignment: dict[str, str], perc_max_gap: float = 1.0) -> dict[str, str]:
        """Filters out sequences in an alignment with a percentage of gaps greater than perc_max_gap

        Args:
            alignment (dict[str,str]): dictionary mapping from sequence id to sequence
            perc_max_gap (float, optional): The maximum percentage of gaps allowed in a sequence. Defaults to 1.0, so no filtering. 

        Returns:
            filtered_alignment (dict[str,str]): Dictionary mapping from sequence id to sequence, with sequence fragments below the threshold excluded
        """
        orig_len = len(alignment)
        filtered_alignment = {seq_id: seq for seq_id, seq in alignment.items() if not ((seq.count(
            "-") + seq.count(".")) / len(seq)) > perc_max_gap}
        print(f"Filtered {orig_len - len(filtered_alignment)} sequences")
        return filtered_alignment

    def get_covered_columns(self, alignment: dict[str, str], perc_max_gap: float = 1.0) -> List[int]:
        """Returns column indices that are sufficiently covered (i.e. have a percentage of gaps less than perc_max_gap)

        Args:
            alignment (dict[str,str]): dictionary mapping from sequence id to sequence
            perc_max_gap (float, optional): The maximum percentage of gaps allowed in a column. Defaults to 1.0.

        Returns:
            List[int]: List of column indices that are sufficiently covered
        """
        covered_cols = []
        all_seqs = list(alignment.values())
        orig_len = len(all_seqs[0])
        # TODO: Definitely a more efficient way to do this than looping with a list comprehension over the whole alignment
        for i in range(orig_len):
            column = "".join([seq[i] for seq in all_seqs])
            if not (column.count("-") + column.count(".")) / len(column) > perc_max_gap:
                covered_cols.append(i)
        print(f"{orig_len - len(covered_cols)} columns are uncovered")
        return covered_cols

    def one_hot_encode_alignment(self, alignment: dict[str, str]) -> np.ndarray:
        """One-hot encodes the alignment

        Args:
            alignment (dict[str,str]): List of sequences in the alignment

        Returns:
            np.ndarray: One-hot encoded alignment
        """
        one_hot_encoding = np.zeros(
            (len(alignment), len(alignment[list(alignment.keys())[0]]), len(self.alphabet)))
        for i, sequence in enumerate(alignment):
            for j, letter in enumerate(sequence):
                if letter in self.aa_dict:
                    k = self.aa_dict[letter]
                    one_hot_encoding[i, j, k] = 1.0
        return one_hot_encoding

    def compute_weights(self, one_hot_encoding: np.ndarray) -> np.ndarray:
        """Computes MSA sequence weights for an alignment

        Args:
            one_hot_encoding (np.ndarray): One-hot encoded alignment. A numpy array of size (num_seqs, seq_len, aa_vocab_size)

        Returns:
            weights (np.ndarray): A numpy array of size (num_seqs) containing the weight for each sequence
        """
        list_seq = one_hot_encoding
        list_seq = list_seq.reshape(
            (list_seq.shape[0], list_seq.shape[1] * list_seq.shape[2]))

        def compute_weight(seq):
            number_non_empty_positions = np.dot(seq, seq)
            if number_non_empty_positions > 0:
                denom = np.dot(list_seq, seq) / np.dot(seq, seq)
                denom = np.sum(denom > 1 - self.weight_theta)
                return 1/denom
            else:
                return 0.0  # return 0 weight if sequence is fully empty
        weights = np.array(list(map(compute_weight, list_seq)))
        return weights
