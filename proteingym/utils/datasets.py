import torch 
from typing import List, Optional
import pandas as pd 
class MutationDataset(torch.utils.data.Dataset):

    def __init__(self, mutations: List[str], labels: List, target_seq:Optional[str]=None):
        self.mutations = mutations
        self.labels = labels
        self.target_seq = target_seq
    
    @classmethod 
    def from_df(cls, df: pd.DataFrame, mutated_sequence_column:str = "mutated_sequence", label_column:str = "DMS_score", target_seq:Optional[str]=None):
        return cls(df[mutated_sequence_column].values.tolist(), df[label_column].values.tolist(), target_seq)
    
    def __len__(self):
        return len(self.mutations)
    
    def __getitem__(self, idx):
        return self.mutations[idx], self.labels[idx]

