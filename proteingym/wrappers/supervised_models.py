from typing import Callable, List, Union, Dict, Any, Optional
import functools
import lightning as L
from proteingym.constants.data import AA_INTEGER_MAP
import torch
from proteingym.wrappers.generic_models import ProteinLanguageModel, SequenceFitnessModel


class SupervisedModel(L.LightningModule):

    def __init__(self, loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
                 optimizer_base: functools.partial = functools.partial(torch.optim.Adam, lr=1e-3),
                 scheduler_base: Optional[functools.partial] = None,
                 scheduler_settings: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.loss_func = loss_func
        self.optimizer_base = optimizer_base
        self.scheduler_base = scheduler_base
        self.scheduler_settings = scheduler_settings

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        params = list(self.parameters())
        optimizer = self.optimizer_base(params=params)
        if self.scheduler_base is not None:
            scheduler = self.scheduler_base(optimizer=optimizer)
            if self.scheduler_settings is not None:
                lr_config = dict(self.scheduler_settings)
                lr_config.update({"scheduler": scheduler})
            else:
                lr_config = {"scheduler": scheduler}
            return {"optimizer": optimizer, "lr_scheduler": lr_config}
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        x = batch["inputs"]
        y = batch["labels"]
        # For one dimensional case, adding output_dim to labels since they are just batch_size scalars
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        y_hat = self(x)
        loss = self.loss_func(y_hat, y.type(torch.float32))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["inputs"]
        y = batch["labels"]
        y_hat = self(x)
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        loss = self.loss_func(y_hat, y.type(torch.float32))
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch["inputs"]
        y = batch["labels"]
        y_hat = self(x)
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        loss = self.loss_func(y_hat, y.type(torch.float32))
        self.log("test_loss", loss)
        return loss


class PLMFinetunedModel(SupervisedModel):
    """Class to finetune a protein language model with a linear head on top of the PLM embeddings"""

    # TODO: Add option to cache embeddings if underlying model is not being finetuned (separate EmbeddingCache class would be useful
    # to store all the logic about caching on disk vs. storing in memory)
    def __init__(
        self,
        model: ProteinLanguageModel,
        output_dim: int,
        model_trainable_layers: Optional[List[int]] = None,
        pool_op: str = "mean",
        embedding_layers: List[int] = [-1],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model = model
        self.model_trainable_layers = model_trainable_layers  # layers of embedding model that are not frozen, default of None is all frozen
        if self.model_trainable_layers is None:
            self.model.base_model.eval()
            self.model.requires_grad_(False)
        else:
            self.model.base_model.train()
            self.model.set_trainable_layers(self.model_trainable_layers)
        self.embedding_layers = embedding_layers
        self.pool_op = pool_op
        if self.pool_op == "mean":
            self.pool = lambda x: torch.mean(x, dim=1)
        elif self.pool == "max":
            self.pool = lambda x: torch.max(x, dim=1).values
        elif self.pool_op == "sum":
            self.pool = lambda x: torch.sum(x, dim=1)
        else:
            raise ValueError("pool_op must be one of ['mean', 'max', 'sum']")
        # default is to just concat multiple layer embeddings together
        self.embed_dim = sum(model.get_embed_dim(layers=embedding_layers))
        self.linear_head = torch.nn.Linear(self.embed_dim, output_dim)
        # TODO: fix save_hyperparameters here (fails because of pickling issue in pool_op and loss_func, I think)
        self.save_hyperparameters(
            ignore=["loss_func", "model", "scheduler_base", "optimizer_base"]
        )

    def forward(self, x):
        embeddings = self.model.get_embeddings(x,layers=self.embedding_layers)
        concat_embeddings = torch.cat(
            [embeddings[layer] for layer in self.embedding_layers], dim=2
        )  # resulting shape is (num_seqs, L, self.embed_dim)
        pooled_embeddings = self.pool(
            concat_embeddings
        )  # resulting shape is (num_seqs, self.embed_dim)
        return self.linear_head(pooled_embeddings)

class OneHotEncodingRegression(SupervisedModel):
    # TODO: Add option to add inputs from arbitrary additional features (their output fitness predictions, not embeddings)

    def __init__(
        self,
        output_dim: int, 
        vocab: Dict[str,int] = AA_INTEGER_MAP,
        auxillary_models: Optional[List[SequenceFitnessModel]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab = vocab
        self.pad_idx = len(self.vocab)
        self.output_dim = output_dim
        self.auxillary_models = auxillary_models

    def one_hot_encode_and_pad_sequences(self, x: List[str]) -> torch.Tensor:
        one_hot_enc = []
        for seq in x:
            seq_one_hot = []
            for aa in seq:
                seq_one_hot.append(self.vocab[aa])
            one_hot_enc.append(torch.tensor(seq_one_hot))
        return torch.nn.utils.rnn.pad_sequence(one_hot_enc, batch_first=True, padding_value=self.pad_idx)

    def forward(self, x: List[str]) -> torch.Tensor:
        one_hot_enc = self.one_hot_encode_and_pad_sequences(x) # batch_size x seq_len x vocab_size
        aux_scores = torch.Tensor([aux_model.predict_fitnesses(x) for aux_model in self.auxillary_models]) # batch_size scalars 
        full_enc = torch.cat([one_hot_enc] + aux_scores, dim=1)