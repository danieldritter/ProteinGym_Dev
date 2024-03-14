import json
from typing import List, Union

import pandas as pd
import torch
from numpy.core import ndarray
from transformers import PreTrainedTokenizerFast
from proteingym.models.model_repos import tranception
from proteingym.wrappers.generic_models import ProteinLanguageModel


@ProteinLanguageModel.register("tranception")
class TranceptionModel(ProteinLanguageModel):

    def __init__(
        self,
        tokenizer_file: str,
        tranception_config: str,
        score_both_directions: bool = True,
        num_workers: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.score_both_directions = score_both_directions
        self.num_workers = num_workers
        self.tranception_config = json.load(open(tranception_config))
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
        )
        self.tranception_config["tokenizer"] = self.tokenizer
        config = tranception.config.TranceptionConfig(**self.tranception_config)
        self.model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(
            pretrained_model_name_or_path=self.model_checkpoint, config=config
        )
        if not self.nogpu and torch.cuda.is_available():
            self.model = self.model.cuda()

    def predict_logprobs(
        self, sequences: List[str], wt_sequence: Union[str, None] = None
    ) -> List[float]:
        # TODO: using score_mutants function within model for now, but this is designed to handle a lot of retrieval cases
        # could write a pared down version that doesn't require putting things in dataframes and just scores the sequences directly.
        seq_df = pd.DataFrame({"mutated_sequence": sequences})
        scores = self.model.score_mutants(
            DMS_data=seq_df,
            target_seq=wt_sequence,
            scoring_mirror=self.score_both_directions,
            batch_size_inference=self.batch_size,
            num_workers=self.num_workers,
            indel_mode=True,
        )
        return scores["avg_score"].tolist()

    def predict_position_logprobs(
        self, sequences: List[str], wt_sequence: Union[str, None] = None
    ) -> List[ndarray]:
        pass

    def get_embeddings(
        self, sequences: List[str], layer: str = "last"
    ) -> List[ndarray]:
        pass
