import torch
from proteingym.wrappers.protein_language_model import ProteinLanguageModel
from proteingym.models.model_repos import esm


@ProteinLanguageModel.register("msa_transformer_model")
class MSATransformerModel(ProteinLanguageModel):

    def __init__(self, model_checkpoint, random_seeds, eval_mode=True, nogpu=False):
        super().__init__(model_checkpoint=model_checkpoint, eval_mode=eval_mode, nogpu=nogpu)
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(
            model_checkpoint)
        self.batch_converter = self.alphabet.get_batch_converter()
        if self.eval_mode:
            self.model.eval()
        if not self.nogpu and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.random_seeds = random_seeds
