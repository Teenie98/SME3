import torch
import torch.nn as nn
from rec_models.base import BaseRecModel, col_num


class DeepFM(BaseRecModel):
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

        # fm
        self.fm_layer = nn.Linear(sum_emb_size, 1)
        self.act_func = nn.Sigmoid()

    def forward_with_embs(self, embs):
        # fm
        fm_emb_1 = torch.cat(embs, dim=1)
        fm_1st = self.fm_layer(fm_emb_1)
        fm_emb_2 = torch.stack(embs, dim=1)
        square_of_sum = fm_emb_2.sum(dim=1).pow(2)
        sum_of_square = fm_emb_2.pow(2).sum(dim=1)
        cross_term = square_of_sum - sum_of_square
        fm_2nd = 0.5 * torch.sum(cross_term, dim=1, keepdim=True)

        # dnn
        dnn_emb = torch.cat(embs, dim=1)
        dnn_output = self.dnn_layers(dnn_emb)

        output = fm_1st + fm_2nd + dnn_output
        output = self.act_func(output)
        return output
