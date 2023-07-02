import os
import torch
import torch.nn as nn
import numpy as np
import pickle
from data import Dataset, col_num
from rec_models import device
from gen_models.base import BaseGenModel


def get_items(name):
    data = Dataset('../data/' + name).x
    data_x, data_y, data_g, data_t = data[:, 0: 1], data[:, 5: 6], data[:, 6: 10], data[:, 10:]
    data = np.hstack([data_x, data_y, data_g, data_t])
    data = np.unique(data, axis=0)

    items = []
    for u in data:
        idx, y, g, t = u[0], u[1], set(u[2: 6]), set(u[6:])
        g.discard(0)
        t.discard(0)
        items.append((idx, y, g, t))
    return items


class SME(BaseGenModel):
    def __init__(self, args):
        super().__init__(args)
        self.generated_emb_layer = nn.Sequential(
            nn.Linear(3 * args.embedding_size, 16),
            nn.LeakyReLU(),
            nn.Linear(16, args.embedding_size),
            nn.Tanh()
        )
        self.generated_emb_layer2 = nn.Sequential(
            nn.Linear(4 * args.embedding_size, 16),
            nn.LeakyReLU(),
            nn.Linear(16, args.embedding_size),
            nn.Tanh()
        )
        self.sim_mat = torch.load('./data/SME/sim_mat.pkl').to(device)

    def forward(self, rec_model, x):
        embs = rec_model.get_embs(x)
        year_emb, genres_emb, title_emb = embs[5], embs[6], embs[7]
        embs = torch.cat([year_emb, genres_emb, title_emb], dim=1)
        embs1 = self.generated_emb_layer(embs)

        if self.training:
            sim_temp = self.sim_mat[x[:, 0]]
            mask = (torch.rand(sim_temp.shape) > 0.5).float().to(device)
            sim_temp = sim_temp * mask
            sim = torch.zeros(sim_temp.shape[0], 20).to(device)
            for idx in range(sim_temp.shape[0]):
                cnt = 0
                for value in sim_temp[idx]:
                    if value != 0:
                        sim[idx][cnt] = value
                        cnt += 1
                    if cnt == 20:
                        break
            sim_emb = rec_model.embeddings['MovieID'](sim.long())
        else:
            sim_emb = rec_model.embeddings['MovieID'](self.sim_mat[x[:, 0]][:, :20])
        # sim_emb = rec_model.embeddings['MovieID'](self.sim_mat[x[:, 0]][:, :20])
        scores = torch.sum(embs1.unsqueeze(1) * sim_emb, dim=-1).softmax(dim=-1)
        sim_embs = (sim_emb * scores.unsqueeze(-1)).sum(dim=1)

        embs2 = self.generated_emb_layer2(torch.cat([embs, sim_embs], dim=1))

        output = sim_embs + embs2
        return output, 0


if __name__ == '__main__':
    # 找属性相似的旧项目
    old_items = get_items('big_train_main')
    new_items = get_items('test_test')
    items = old_items + new_items

    sim_mat = torch.zeros((col_num[0][1], 64)).long()
    for idx, y, g, t in items:
        scores = []
        for o_idx, o_y, o_g, o_t in old_items:
            if o_idx == idx:
                continue
            s_y = 1 if o_y == y else 0
            a = len(g.intersection(o_g)) + len(t.intersection(o_t)) + (1 if o_y == y else 0)
            b = len(g.union(o_g)) + len(t.union(o_t)) + (1 if o_y == y else 2)
            s_g = len(g.intersection(o_g)) / len(g.union(o_g))
            s_t = len(t.intersection(o_t)) / len(t.union(o_t))
            scores.append((o_idx, a / b))

        scores.sort(key=lambda x: -x[1])
        for i in range(sim_mat.size(1)):
            sim_mat[idx, i] = scores[i][0]

    filename = "./data/SME/sim_mat.pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(sim_mat, filename)
