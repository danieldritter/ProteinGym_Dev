from typing import Callable, List

import lightning as L
import torch
from proteingym.wrappers.generic_models import ProteinLanguageModel


class PLMFinetunedModel(L.LightningModule):
    """Class to finetune a protein language model with a linear head on top of the PLM embeddings"""

    def __init__(
        self,
        model: ProteinLanguageModel,
        output_dim: int,
        pool_op: Callable[[torch.Tensor], torch.Tensor] = lambda x: torch.mean(x, dim=1),
        loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.nn.MSELoss,
        embedding_layers: List[int] = [-1],
        **kwargs
    ):
        super().__init__()
        self.model = model
        self.embedding_layer = embedding_layers
        self.pool_op = pool_op
        self.loss_func = loss_func
        # default is to just concat multiple layer embeddings together 
        self.embed_dim = sum(model.get_embed_dim(layers=embedding_layers))
        self.linear_head = torch.nn.Linear(self.embed_dim, output_dim)

    def forward(self, x):
        embeddings = self.model.get_embeddings(x)
        concat_embeddings = torch.cat([embeddings[layer] for layer in self.embedding_layer], dim=2) # resulting shape is (num_seqs, L, self.embed_dim)
        pooled_embeddings = self.pool_op(concat_embeddings) # resulting shape is (num_seqs, self.embed_dim)
        return self.linear_head(pooled_embeddings)

    def training_step(self, batch, batch_idx):
        x = batch["inputs"]
        y = batch["labels"]
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        return loss 

    def validation_step(self, batch, batch_idx):
        x = batch["inputs"]
        y = batch["labels"]
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch["inputs"]
        y = batch["labels"]
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        # TODO: Make more general 
        return torch.optim.Adam(self.parameters(), lr=1e-3)

