import lightning as L 
import torch
from proteingym.wrappers.generic_models import ProteinLanguageModel 

class PLMFinetunedModel(L.LightningModule):
    """ Class to finetune a protein language model with a linear head on top of the PLM embeddings
    """

    def __init__(self, model:ProteinLanguageModel, output_dim: int, pool_op:str = "avg", embedding_layer:str ="last", **kwargs):
        super().__init__()
        self.model = model
        self.embedding_layer = embedding_layer
        if pool_op == "avg":
            self.pool_func = lambda x: torch.mean(x, dim=1) # embeddings are num_sequences x seq_len x embed_dim, so pool to get num_sequences x embed_dim
        elif pool_op == "max":
            self.pool_func = lambda x: torch.max(x, dim=1)
        self.linear_head = torch.nn.Linear(model.get_embed_dim(layer=embedding_layer), output_dim)

    def forward(self, x):
        embeddings = self.model.get_embeddings(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)