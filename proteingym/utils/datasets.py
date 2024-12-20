"""Contains classes for ProteinGym datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
import torch


class MutationDataset[LabelType](torch.utils.data.Dataset):
    """Represents a dataset of mutations."""

    def __init__(
        self,
        mutations: list[str],
        labels: list[LabelType],
        mutants: list[str] | None = None,
        target_seq: str | None = None,
    ) -> None:
        """Initialize a MutationDataset.

        Args:
            mutations (list[str]): list of mutations
            labels (list): list of labels for each sequence
            mutants (list[str] | None, optional): optionally can pass
                in mutant triplets. Defaults to None.
            target_seq (list[str] | None, optional): Can pass in target seq along with
                mutants triplets to derive mutants. Defaults to None.

        """
        self.mutations = np.array(mutations)
        self.labels = np.array(labels)
        if mutants is not None:
            self.mutants = np.array(mutants)
        else:
            self.mutants = None
        self.target_seq = target_seq

    @classmethod
    def from_df(
        cls: type[MutationDataset],
        df: pd.DataFrame,
        mutated_sequence_column: str = "mutated_sequence",
        label_column: str = "DMS_score",
        mutant_column: str | None = "mutant",
        target_seq: str | None = None,
    ) -> MutationDataset:
        """Create a MutationDataset from a Pandas dataframe.

        Args:
            cls (type[MutationDataset]): class name of MutationDataset
            df (pd.DataFrame): Pandas dataframe to convert
            mutated_sequence_column (str, optional): column containing mutant sequences.
                Defaults to "mutated_sequence".
            label_column (str, optional): column containing sequence labels.
                Defaults to "DMS_score".
            mutant_column (str | None, optional): column containing triplet mutants.
                Can be passed in with target seq to derive mutant sequences.
                Defaults to "mutant".
            target_seq (str | None, optional): base sequence to mutate.
                Can be passed in with mutant column to derive mutant sequences.
                Defaults to None.

        Returns:
            MutationDataset: torch dataset representing mutations in dataframe

        """
        if mutant_column in df:
            return cls(
                df[mutated_sequence_column].to_numpy().tolist(),
                df[label_column].to_numpy().tolist(),
                df[mutant_column].to_numpy().tolist(),
                target_seq,
            )
        return cls(
            df[mutated_sequence_column].to_numpy().tolist(),
            df[label_column].to_numpy().tolist(),
            target_seq=target_seq,
        )

    def __len__(self) -> int:
        """Return length of the dataset.

        Returns:
            int: number of mutations in dataset

        """
        return len(self.mutations)

    def __getitem__(self, idx: int) -> dict[str, str | LabelType]:
        """Return an element of the dataset for a given index.

        Args:
            idx (int): index of element to return

        Returns:
            dict[str, str | LabelType]: dict containing input sequence and label

        """
        return {"inputs": self.mutations[idx], "labels": self.labels[idx]}

    def train_val_test_split(
        self,
        split_type: str = "random",
        train_ratio: float = 0.8,
        val_ratio: float = 0.10,
    ) -> tuple[MutationDataset, MutationDataset, MutationDataset]:
        """Split dataset into train, val, and test sets.

        Note: data not used in train and val is used in test
        Args:
            split_type (str, optional): type of split to use. Must be one of
                ["random", "contiguous", "modulo"]. Defaults to "random".
            train_ratio (float, optional): percentage of data to use for training.
                Defaults to 0.8.
            val_ratio (float, optional): percentage of data to use for validation.
                Defaults to 0.10.

        Raises:
            ValueError: if train_ratio + val ratio is greater than 1
            ValueError: self.mutants is not defined but "contiguous" is passed
            ValueError: self.mutants is not defined but "modulo" is passed
            ValueError: split_type is not one of ["random", "contiguous", "modulo"]

        Returns:
            tuple[MutationDataset, MutationDataset, MutationDataset]: tuple of
            train, validation and test dataset

        """
        if not train_ratio + val_ratio <= 1:
            msg = "train_ratio + val_ratio should be less than or equal to 1"
            raise ValueError(msg)
        if split_type == "random":
            indices = torch.randperm(len(self.mutations))
            train_size = int(train_ratio * len(self.mutations))  # int rounds down
            val_size = int(val_ratio * len(self.mutations))
            train_idx, val_idx, test_idx = (
                indices[:train_size],
                indices[train_size : train_size + val_size],
                indices[train_size + val_size :],
            )  # In the event of a remainder, test set will have an additional datapoint
        # TODO (@DanielR): implement contiguous and modulo splits
        elif split_type == "contiguous":
            if self.mutants is None:
                msg = "triplet mutants must be provided when split_type is 'contiguous'"
                raise ValueError(msg)
            raise NotImplementedError
        elif split_type == "modulo":
            if self.mutants is None:
                msg = "triplet mutants must be provided when split_type is 'modulo'"
                raise ValueError(msg)
            raise NotImplementedError
        else:
            msg = "split_type must be one of ['random', 'contiguous', 'modulo']"
            raise ValueError(msg)
        if self.mutants is not None:
            train_mutants = self.mutants[train_idx]
            val_mutants = self.mutants[val_idx]
            test_mutants = self.mutants[test_idx]
        else:
            train_mutants = None
            val_mutants = None
            test_mutants = None
        return (
            MutationDataset(
                self.mutations[train_idx],
                self.labels[train_idx],
                mutants=train_mutants,
                target_seq=self.target_seq,
            ),
            MutationDataset(
                self.mutations[val_idx],
                self.labels[val_idx],
                mutants=val_mutants,
                target_seq=self.target_seq,
            ),
            MutationDataset(
                self.mutations[test_idx],
                self.labels[test_idx],
                mutants=test_mutants,
                target_seq=self.target_seq,
            ),
        )
