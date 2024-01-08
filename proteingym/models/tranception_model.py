from proteingym.wrappers.protein_language_model import ProteinLanguageModel
from proteingym.models.model_repos import tranception
from typing import Union


class TranceptionModel(ProteinLanguageModel):

    def __init__(self, model_checkpoint: Union[str, None] = None, eval_mode: bool = True, nogpu: bool = False, attention_mode="tranception",
                 position_embedding="group_alibi", tokenizer=None, scoring_window="optimal"):
        super().__init__(model_checkpoint=model_checkpoint, eval_mode=eval_mode, nogpu=nogpu)
        self.attention_mode = attention_mode
        self.position_embedding = position_embedding
        self.tokenizer = tokenizer
        self.scoring_window = scoring_window
        config = tranception.config.TranceptionConfig(
            attention_mode=self.attention_mode, position_embedding=self.position_embedding,
            tokenizer=self.tokenizer, scoring_window=self.scoring_window)
        self.model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(
            pretrained_model_name_or_path=self.model_checkpoint, config=config)
