from typing import List
from Bio import SeqIO


class Alignment:

    def __init__(self, alignment_file: str, alignment_format: str = "fasta", focus_seq_ind: int = 0,
                 perc_max_gap_seq_fragments: float = 1.0, perc_max_gap_column_coverage: float = 1.0,
                 uppercase: bool = True) -> None:
        self.alignment_file = alignment_file
        self.alignment_format = alignment_format
        self.alignment = self.read_alignment(
            alignment_file, alignment_format, uppercase)
        self.focus_seq_ind = focus_seq_ind
        self.perc_max_gap_seq_fragments = perc_max_gap_seq_fragments
        self.perc_max_gap_column_coverage = perc_max_gap_column_coverage
        self.uppercase = uppercase
        # Read the alignment file
        if self.focus_seq_ind is not None:
            # Read alignment file and get focus alignment
            self.focus_cols = self.get_focus_cols(
                self.alignment, self.focus_seq_ind)
            self.alignment = self.get_focus_alignment(
                self.alignment, self.focus_cols)
        else:
            # Read alignment file
            self.alignment = self.read_alignment(
                self.alignment_file, self.alignment_format)
            self.focus_cols = None
        # Filter sequences
        if perc_max_gap_seq_fragments != 1.0:
            self.alignment = self.filter_sequence_fragments(
                self.alignment, perc_max_gap_seq_fragments)
        if perc_max_gap_column_coverage != 1.0:
            self.covered_columns = self.get_covered_columns(
                self.alignment, perc_max_gap_column_coverage)
        else:
            self.covered_columns = list(range(len(self.alignment[0])))

    def read_alignment(self, alignment_file: str, alignment_format: str, uppercase: bool = True) -> List[str]:
        """
        Read an alignment file and yield a list of sequences,
        """
        if uppercase:
            return [str(seq.seq).upper().replace(".", "-") for seq in SeqIO.parse(alignment_file, alignment_format)]
        return [str(seq.seq) for seq in SeqIO.parse(alignment_file, alignment_format)]

    def get_focus_cols(self, alignment: List[str], focus_seq_ind: int = 0) -> List[int]:
        """
        Get the columns of the alignment that are in the focus sequence
        """
        focus_cols = []
        for i, aa in enumerate(alignment[focus_seq_ind]):
            if aa != "-":
                focus_cols.append(i)
        return focus_cols

    def get_focus_alignment(self, alignment: List[str], focus_cols: List[int]) -> List[str]:
        """
        Get the alignment with only the columns in the focus sequence
        """
        focus_alignment = []
        for seq in alignment:
            focus_alignment.append("".join([seq[i] for i in focus_cols]))
        return focus_alignment

    def filter_sequence_fragments(self, alignment: List[str], perc_max_gap: float = 1.0) -> List[str]:
        """
        Filter sequences in an alignment based on the percentage of gaps

        Parameters:
        - alignment: a list of sequences
        - perc_max_gap: the maximum percentage of gaps allowed in a sequence. Default is 1.0, so no filtering

        Returns:
        - a list of sequences
        """
        orig_len = len(alignment)
        filtered_alignment = [seq for seq in alignment if not seq.count(
            "-") / len(seq) > perc_max_gap]
        print(f"Filtered {orig_len - len(filtered_alignment)} sequences")
        return filtered_alignment

    def get_covered_columns(self, alignment: List[str], perc_max_gap: float = 1.0) -> List[int]:
        """
        Filter columns in an alignment based on the percentage of gaps

        Parameters:
        - alignment: a list of sequences
        - perc_max_gap: the maximum percentage of gaps allowed in a column. Default is 1.0, so no columns are removed

        Returns:
        - covered_cols: a list of columns in the alignment that are sufficiently covered
        """
        orig_len = len(alignment[0])
        covered_cols = []
        # TODO: Definitely a more efficient way to do this than looping with a list comprehension over the whole alignment
        for i in range(orig_len):
            column = "".join([seq[i] for seq in alignment])
            if not column.count("-") / len(column) > perc_max_gap:
                covered_cols.append(i)
        print(f"{orig_len - len(covered_cols)} columns are uncovered")
        return covered_cols
