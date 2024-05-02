from typing import Callable, List, Union, Dict, Any, Optional
import functools
import lightning as L
import torch
from proteingym.wrappers.generic_models import ProteinLanguageModel


class PLMFinetunedModel(L.LightningModule):
    """Class to finetune a protein language model with a linear head on top of the PLM embeddings"""
    # TODO: Add option to cache embeddings if underlying model is not being finetuned (separate EmbeddingCache class would be useful 
    # to store all the logic about caching on disk vs. storing in memory)
    def __init__(
        self,
        model: ProteinLanguageModel,
        output_dim: int,
        pool_op: str = "mean",
        loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.nn.MSELoss(),
        optimizer_base: functools.partial = functools.partial(torch.optim.Adam, lr=1e-3),
        scheduler_base: Optional[functools.partial] = None,
        scheduler_settings: Optional[Dict[str, Any]] = None,   
        embedding_layers: List[int] = [-1],
    ):
        super().__init__()
        # TODO: Parameters of embedding model not being included in training for some reason 
        self.model = model
        self.embedding_layers = embedding_layers
        self.pool_op = pool_op
        self.optimizer_base = optimizer_base 
        self.scheduler_base = scheduler_base
        self.scheduler_settings = scheduler_settings
        if self.pool_op == "mean":
            self.pool = lambda x: torch.mean(x, dim=1)
        elif self.pool == "max":
            self.pool = lambda x: torch.max(x, dim=1).values
        elif self.pool_op == "sum":
            self.pool = lambda x: torch.sum(x, dim=1)
        else:
            raise ValueError("pool_op must be one of ['mean', 'max', 'sum']")
        self.loss_func = loss_func
        # default is to just concat multiple layer embeddings together 
        self.embed_dim = sum(model.get_embed_dim(layers=embedding_layers))
        self.linear_head = torch.nn.Linear(self.embed_dim, output_dim)
        # TODO: fix save_hyperparameters here (fails because of pickling issue in pool_op and loss_func, I think)
        self.save_hyperparameters(ignore=["loss_func","model","scheduler_base","optimizer_base"]) 

    def forward(self, x):
        embeddings = self.model.get_embeddings(x)
        concat_embeddings = torch.cat([embeddings[layer] for layer in self.embedding_layers], dim=2) # resulting shape is (num_seqs, L, self.embed_dim)
        pooled_embeddings = self.pool(concat_embeddings) # resulting shape is (num_seqs, self.embed_dim)
        return self.linear_head(pooled_embeddings)

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
        loss = self.loss_func(y_hat, y)
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch["inputs"]
        y = batch["labels"]
        y_hat = self(x)
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        loss = self.loss_func(y_hat, y)
        self.log("test_loss", loss)
        return loss
    
    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        # Fitting this in positional argument portion of hydra config
        optimizer = self.optimizer_base(params=self.parameters())
        if self.scheduler_base is not None:
            scheduler = self.scheduler_base(optimizer=optimizer)
            if self.scheduler_settings is not None:
                lr_config = dict(self.scheduler_settings)
                lr_config.update({"scheduler":scheduler})
            else:
                lr_config = {"scheduler": scheduler}
            return {"optimizer":optimizer,"lr_scheduler":lr_config}
        else:
            return optimizer
