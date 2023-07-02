import os
import torch
import torch.nn as nn
import numpy as np
from data import Dataset, col_num
from rec_models import device
from gen_models.base import BaseGenModel


def get_items_2(name):
    data = Dataset('../data/' + name).x
    data_like = Dataset('../data/' + name).y
    data_m, data_u = data[:, 0: 1], data[:, 1: 2]
    data = np.hstack([data_m, data_u, data_like])
    data = np.unique(data[np.where(data[:, 2] == 1)], axis=0)

    movielist = np.unique(data_m)
    dic = dict([(k, []) for k in movielist])
    for u in data:
        dic[u[0]].append(u[1])
    return dic


def dic_tolist(dic):
    items = []
    for k, v in dic.items():
        items.append((k, set(v)))
    return items


def dic_combine(d1, d2):
    combined_keys = d1.keys() | d2.keys()
    d_comb = {key: d1.get(key, []) + d2.get(key, []) for key in combined_keys}
    return d_comb


class SME2(BaseGenModel):
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
        self.sim_mat = torch.load('./data/SME/sim_mat_2.pkl').to(device)

    def forward(self, rec_model, x):
        embs = rec_model.get_embs(x)
        year_emb, genres_emb, title_emb = embs[5], embs[6], embs[7]
        embs = torch.cat([year_emb, genres_emb, title_emb], dim=1)
        embs1 = self.generated_emb_layer(embs)

        if self.training:
            sim_temp = self.sim_mat[x[:, 0]]
            mask = (torch.rand(sim_temp.shape) > 0.4).float().to(device)
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
        # output = sim_embs
        # output = embs2
        return output, 0


if __name__ == '__main__':
    # 找用户交互相似的旧项目
    old_items = dic_tolist(get_items_2('big_train_main'))
    new_item_a = get_items_2('test_oneshot_a')
    new_item_b = get_items_2('test_oneshot_b')
    new_item_c = get_items_2('test_oneshot_c')
    new_items = dic_combine(new_item_a, new_item_b)
    new_items = dic_combine(new_items, new_item_c)
    items = old_items + dic_tolist(new_items)

    sim_mat = torch.zeros((col_num[0][1], 64)).long()
    for idx, userlist in items:
        scores = []
        for o_idx, o_userlist in old_items:
            if o_idx == idx:
                continue
            a = len(userlist.intersection(o_userlist))
            b = len(userlist.union(o_userlist))
            scores.append((o_idx, a / b))

        scores.sort(key=lambda x: -x[1])
        for i in range(sim_mat.size(1)):
            sim_mat[idx, i] = scores[i][0]

    filename = "./data/SME/sim_mat_2.pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(sim_mat, filename)
