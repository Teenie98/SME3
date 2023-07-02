import torch
import torch.nn as nn
from rec_models.base import BaseRecModel, col_num


class IPNN(BaseRecModel):
    def __init__(self, args):
        super().__init__(args)

        cross_attr_num = len(col_num) * (len(col_num) - 1) // 2
        sum_emb_size = self.args.embedding_size * len(col_num) + cross_attr_num

        self.kernel_shape = (self.args.embedding_size, len(col_num))
        self.kernel = nn.Parameter(torch.randn(self.kernel_shape))
        nn.init.xavier_uniform_(self.kernel.data)

        self.dnn_layers = nn.Sequential(
            nn.Linear(sum_emb_size, self.args.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.args.hidden_layer_size, self.args.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.args.hidden_layer_size, self.args.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.args.hidden_layer_size, 1)
        )

        self.act_func = nn.Sigmoid()

    def forward_with_embs(self, embs):
        pnn_emb = torch.stack(embs, dim=1)
        pnn_emb = torch.sum(pnn_emb.unsqueeze(1) * self.kernel.reshape(1, self.kernel.shape[0], -1, 1), dim=1)
        row, col = list(), list()
        for i in range(len(col_num) - 1):
            for j in range(i + 1, len(col_num)):
                row.append(i), col.append(j)

        pnn_emb = torch.sum(pnn_emb[:, row] * pnn_emb[:, col], dim=2)

        dnn_emb = torch.cat(embs, dim=1)

        dnn_output = self.dnn_layers(torch.cat([pnn_emb, dnn_emb], dim=1))

        output = self.act_func(dnn_output)
        return output
