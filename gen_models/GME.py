import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from data import col_num, read_pkl
from rec_models import device
from gen_models.base import BaseGenModel


def get_items(dir, file_name):
    data = read_pkl(dir + file_name + '.pkl')
    data.Title = data.Title.apply(
        lambda x: np.array((8 - len(x)) * [0] + x[:8], dtype=np.int32)).to_numpy()
    data.Genres = data.Genres.apply(
        lambda x: np.array((4 - len(x)) * [0] + x[:4], dtype=np.int32)).to_numpy()

    data_items = []
    data_features = {}
    for item, records in data.groupby('MovieID'):
        data_items.append(records.iloc[0:1, [1, 6, 7, 8]])
        data_features[item] = records.iloc[0:1, [1, 6, 7, 8]]
    data_items = pd.concat(data_items, axis=0)

    item_features = {}
    for k, v in data_features.items():
        item_features[k] = []
        item_features[k].append(*list(v.Year.to_numpy()))
        item_features[k].extend(*v.Genres)
        item_features[k].extend(*v.Title)

    return data, data_items, item_features


class GME(BaseGenModel):
    def __init__(self, args):
        super().__init__(args)
        self.W_meta = nn.Parameter(torch.randn(args.embedding_size * 3, 16))
        cur_range = 0.01
        # cur_range = np.sqrt(6.0 / (args.embedding_size * 3 + 16))
        nn.init.uniform_(self.W_meta, -cur_range, cur_range)
        # nn.init.normal_(self.W_meta, std=1e-2, mean=0)

        self.W_gat = nn.Parameter(torch.randn(args.embedding_size * 3, args.embedding_size * 3))
        # cur_range = np.sqrt(6.0 / (args.embedding_size * 3 + args.embedding_size * 3))
        nn.init.uniform_(self.W_gat, -cur_range, cur_range)
        # nn.init.normal_(self.W_gat, std=1e-2, mean=0)

        self.a_gat = nn.Parameter(torch.randn(args.embedding_size * 3 * 2, 1))
        # cur_range = np.sqrt(6.0 / (args.embedding_size * 3 * 2 + 1))
        nn.init.uniform_(self.a_gat, -cur_range, cur_range)
        # nn.init.normal_(self.a_gat, std=1e-2, mean=0)

        self.leaky_relu = nn.LeakyReLU()
        self.softmax1 = nn.Softmax(dim=1)
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

        self.items_ngb = torch.load('./data/GME/items_ngb.pkl').to(device)

    def forward(self, rec_model, x):
        embs = rec_model.get_embs(x)
        year_emb, genres_emb, title_emb = embs[5], embs[6], embs[7]
        embs = torch.cat([year_emb, genres_emb, title_emb], dim=1)

        ngb_x_emb = rec_model.embeddings['Year'](self.items_ngb[x[:, 0], :, 0])
        ngb_g_emb = self.get_masked_mul_hot_emb_ngb(rec_model.embeddings, self.items_ngb[x[:, 0], :, 1:5], 'Genres')
        ngb_t_emb = self.get_masked_mul_hot_emb_ngb(rec_model.embeddings, self.items_ngb[x[:, 0], :, 5:], 'Title')
        ngb_emb_ori = torch.cat([ngb_x_emb, ngb_g_emb, ngb_t_emb], dim=-1)

        emb_exp = embs.unsqueeze(1)
        emb_tile = torch.tile(emb_exp, (1, 10 + 1, 1))
        ngb_emb = torch.cat([ngb_emb_ori, emb_exp], dim=1)

        emb_2d = emb_tile.reshape(-1, 3 * self.args.embedding_size)
        ngb_emb_2d = ngb_emb.reshape(-1, 3 * self.args.embedding_size)

        temp_self = torch.matmul(emb_2d, self.W_gat)
        temp_ngb = torch.matmul(ngb_emb_2d, self.W_gat)

        wgt = self.leaky_relu(torch.matmul(torch.cat((temp_self, temp_ngb), dim=1), self.a_gat))

        wgt = wgt.reshape(-1, 10 + 1, 1)

        nlz_wgt = self.softmax1(wgt)
        temp_ngb_re = temp_ngb.reshape(-1, 10 + 1, self.args.embedding_size * 3)
        # print(temp_ngb_re.shape, nlz_wgt.shape)
        up_emb = self.elu(torch.mul(temp_ngb_re, nlz_wgt).sum(1))

        pred_emb = 1.0 * self.tanh(torch.matmul(up_emb, self.W_meta))

        return pred_emb, 0

    def get_masked_mul_hot_emb_ngb(self, embeddings, mul_hot_index, cate_type):
        data_mask = torch.greater(mul_hot_index, 0).float()
        data_mask2 = torch.greater(mul_hot_index, 0).float()
        data_mask = data_mask.unsqueeze(-1)
        data_mask = torch.tile(data_mask, (1, 1, 1, self.args.embedding_size))
        mul_hot_emb = embeddings[cate_type](mul_hot_index)
        mul_hot_emb = torch.mul(mul_hot_emb, data_mask)
        mul_hot_emb = torch.sum(mul_hot_emb, dim=2) / torch.sum(data_mask2, dim=2, keepdim=True)

        return mul_hot_emb



if __name__ == '__main__':
    data_dir = '../data/'
    old_data, old_items, old_features = get_items(data_dir, 'train_oneshot_a')
    new_data, new_items, _ = get_items(data_dir, 'test_oneshot_a')
    items = pd.concat([new_items, old_items], axis=0)

    year_dic = {}
    genre_dic = {}
    title_dic = {}
    for idx, rows in items.iterrows():
        if rows.Year not in year_dic:
            year_dic[rows.Year] = set()
        year_dic[rows.Year].add(rows.MovieID)

        for genre in rows.Genres:
            if genre == 0:
                continue
            if genre not in genre_dic:
                genre_dic[genre] = set()
            genre_dic[genre].add(rows.MovieID)

        for title in rows.Title:
            if title == 0:
                continue
            if title not in title_dic:
                title_dic[title] = set()
            title_dic[title].add(rows.MovieID)

    is_new = np.zeros(4000)
    is_new[new_items.MovieID] = 1

    res_dic = torch.zeros((col_num[0][1], 10, 13)).long()
    for idx, rows in items.iterrows():
        temp_d = {}
        for mid in year_dic[rows.Year]:
            if mid not in temp_d:
                temp_d[mid] = 0
            temp_d[mid] += 1

        for title in rows.Title:
            if title == 0:
                continue
            for mid in title_dic[title]:
                if mid not in temp_d:
                    temp_d[mid] = 0
                temp_d[mid] += 1

        for genre in rows.Genres:
            if genre == 0:
                continue
            for mid in genre_dic[genre]:
                if mid not in temp_d:
                    temp_d[mid] = 0
                temp_d[mid] += 1

        temp_d = sorted(temp_d.items(), key=lambda x: -x[1])
        kcnt = 0
        for mid, sim in temp_d:
            if is_new[mid] == 0 and mid != rows.MovieID:
                res_dic[rows.MovieID, kcnt] = torch.tensor(old_features[mid]).long()
                kcnt += 1
            if kcnt >= 10:
                break

    filename = "./data/GME/items_ngb.pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(res_dic, filename)

