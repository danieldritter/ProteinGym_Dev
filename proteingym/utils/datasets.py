import torch 
from typing import List, Optional
import pandas as pd 
import numpy as np
class MutationDataset(torch.utils.data.Dataset):

    def __init__(self, mutations: List[str], labels: List, mutants:Optional[List[str]]=None, target_seq:Optional[str]=None):
        self.mutations = np.array(mutations)
        self.labels = np.array(labels)
        if mutants is not None:
            self.mutants = np.array(mutants)
        else:
            self.mutants = None
        self.target_seq = target_seq
    
    @classmethod 
    def from_df(cls, df: pd.DataFrame, mutated_sequence_column:str = "mutated_sequence", label_column:str = "DMS_score", mutant_column: Optional[str] = "mutant",
                target_seq:Optional[str]=None):
        if mutant_column in df:
            return cls(df[mutated_sequence_column].values.tolist(), df[label_column].values.tolist(), df[mutant_column].values.tolist(), target_seq)
        else:
            return cls(df[mutated_sequence_column].values.tolist(), df[label_column].values.tolist(), target_seq=target_seq)
    
    def __len__(self):
        return len(self.mutations)
    
    def __getitem__(self, idx):
        return {"inputs":self.mutations[idx], "labels":self.labels[idx]}
    
    def train_val_test_split(self, split_type="random", train_ratio=0.8, val_ratio=0.10, split_k=Optional[int]):
        if not train_ratio + val_ratio <= 1:
            raise ValueError("train_ratio + val_ratio should be equal to 1")
        if split_type == "random":
            indices = torch.randperm(len(self.mutations))
            train_size = int(train_ratio * len(self.mutations)) # int rounds down 
            val_size = int(val_ratio * len(self.mutations))
            train_idx, val_idx, test_idx = indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:] # In the event of a remainder, test set will have an additional datapoint
        elif split_type == "contiguous":
            if self.mutants is None:
                raise ValueError("triplet mutants must be provided when split_type is 'contiguous'")
        elif split_type == "modulo":
            if self.mutants is None:
                raise ValueError("triplet mutants must be provided when split_type is 'modulo'")
        else:
            raise ValueError("split_type must be one of ['random', 'contiguous', 'modulo']")
        return MutationDataset(self.mutations[train_idx], self.labels[train_idx], self.target_seq), MutationDataset(self.mutations[val_idx], self.labels[val_idx], self.target_seq), MutationDataset(self.mutations[test_idx], self.labels[test_idx], self.target_seq)


