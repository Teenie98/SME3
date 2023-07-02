import torch
import torch.nn as nn
from rec_models.base import BaseRecModel, col_num


class AFM(BaseRecModel):
    def __init__(self, args):
        super().__init__(args)

        sum_emb_size = len(col_num) * self.args.embedding_size
        self.linear_layer = nn.Linear(sum_emb_size, 1)

        self.attention_layer = nn.Linear(self.args.embedding_size, 16)
        self.attention_h = nn.Linear(16, 1, bias=False)
        self.fc_layer = nn.Linear(self.args.embedding_size, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward_with_embs(self, embs):
        linear_emb = torch.cat(embs, dim=1)
        linear_output = self.linear_layer(linear_emb)
        afm_emb = torch.stack(embs, dim=1)

        row, col = list(), list()
        for i in range(len(col_num) - 1):
            for j in range(i + 1, len(col_num)):
                row.append(i), col.append(j)

        afm_emb1, afm_emb2 = afm_emb[:, row], afm_emb[:, col]
        inner_product = afm_emb1 * afm_emb2
        attention_score = self.attention_layer(inner_product)
        attention_score = self.relu(attention_score)
        attention_score = self.attention_h(attention_score)
        attention_score = self.softmax(attention_score)
        attention_output = torch.sum(attention_score * inner_product, dim=1)
        attention_output = self.fc_layer(attention_output)

        output = self.sigmoid(linear_output + attention_output)

        return output
