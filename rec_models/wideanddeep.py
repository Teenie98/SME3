import torch
import torch.nn as nn
from rec_models.base import BaseRecModel, col_num


class WideAndDeep(BaseRecModel):
    def __init__(self, args):
        super().__init__(args)

        # dnn
        sum_emb_size = self.args.embedding_size * len(col_num)
        self.dnn_layers = nn.Sequential(
            nn.Linear(sum_emb_size, self.args.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.args.hidden_layer_size, self.args.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.args.hidden_layer_size, self.args.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.args.hidden_layer_size, 1)
        )

        # wide
        self.wide_layer = nn.Linear(sum_emb_size, 1)

        self.act_func = nn.Sigmoid()

    def forward_with_embs(self, embs):
        # wide
        wide_emb = torch.cat(embs, dim=1)
        wide_output = self.wide_layer(wide_emb)

        # deep
        dnn_emb = torch.cat(embs, dim=1)

        dnn_output = self.dnn_layers(dnn_emb)

        output = wide_output + dnn_output
        output = self.act_func(output)
        return output
