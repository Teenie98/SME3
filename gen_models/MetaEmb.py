import torch
import torch.nn as nn
from gen_models.base import BaseGenModel


class MetaEmb(BaseGenModel):
    def __init__(self, args):
        super().__init__(args)
        self.generated_emb_layer = nn.Sequential(
            nn.Linear(3 * args.embedding_size, 16),
            nn.LeakyReLU(),
            nn.Linear(16, args.embedding_size),
            nn.Tanh()
        )

    def forward(self, rec_model, x):
        embs = rec_model.get_embs(x)
        year_emb, genres_emb, title_emb = embs[5], embs[6], embs[7]
        attr_emb = torch.cat([year_emb, genres_emb, title_emb], dim=1)
        output = self.generated_emb_layer(attr_emb)
        return output, 0
