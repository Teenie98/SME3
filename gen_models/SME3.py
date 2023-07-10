import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data import Dataset, col_num, read_pkl
from rec_models import device
from gen_models.base import BaseGenModel
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


class SME3(BaseGenModel):
    def __init__(self, args):
        super().__init__(args)
        self.generated_emb_layer = nn.Sequential(
            nn.Linear(4 * args.embedding_size, 16),
            nn.LeakyReLU(),
            nn.Linear(16, args.embedding_size),
            nn.Tanh()
        )
        self.generated_emb_layer2 = nn.Sequential(
            nn.Linear(5 * args.embedding_size, 16),
            nn.LeakyReLU(),
            nn.Linear(16, args.embedding_size),
            nn.Tanh()
        )
        self.cross_domain_p = torch.load('../rec_models/cross_domain_p/{}_parameter.pkl'.format(args.base_model))
        self.cross_domain_user_emb = self.cross_domain_p['embeddings.UserID.weight']
        self.sim_mat = pickle.load(open('./data/SME/similar_user.pkl', 'rb')).to(device)

    def forward(self, rec_model, x):
        embs = rec_model.get_embs(x)
        age_emb, gender_emb, occupation_emb = embs[2], embs[3], embs[4]
        cross_domain_emb = self.cross_domain_user_emb[x[:, 1].long()]
        embs = torch.cat([age_emb, gender_emb, occupation_emb, cross_domain_emb], dim=1)
        embs1 = self.generated_emb_layer(embs)
        #
        # sim_emb = rec_model.embeddings['UserID'](self.sim_mat[x[:, 0]][:, :20])
        #
        # scores = torch.sum(embs1.unsqueeze(1) * sim_emb, dim=-1).softmax(dim=-1)
        # sim_embs = (sim_emb * scores.unsqueeze(-1)).sum(dim=1)
        #
        # embs2 = self.generated_emb_layer2(torch.cat([embs, sim_embs], dim=1))
        #
        # output = sim_embs + embs2
        return embs1, 0


if __name__ == '__main__':
    # 找属性相似的旧项目
    old_users = read_pkl('../data/big_train_main.pkl')
    old_users = old_users[['UserID', 'Age', 'Gender', 'Occupation']].drop_duplicates().reset_index()
    new_users = read_pkl('../data/test_test.pkl')
    new_users = new_users[['UserID', 'Age', 'Gender', 'Occupation']].drop_duplicates().reset_index()


    # 计算相似度
    similarities = cosine_similarity(new_users[['Age', 'Gender', 'Occupation']], old_users[['Age', 'Gender', 'Occupation']])

    # 找到每个new_user对应的前20个old_user
    top_20_similar_users = {}
    for i, similarity_scores in enumerate(similarities):
        top_20_indices = similarity_scores.argsort()[-20:][::-1]
        top_20_users = old_users.loc[top_20_indices, 'UserID'].tolist()
        test_user_id = new_users.loc[i, 'UserID']
        top_20_similar_users[test_user_id] = top_20_users

    tensor = torch.zeros((6051, 20))
    for key, value in top_20_similar_users.items():
        tensor[key] = torch.tensor(value)

    # 存数据
    with open('./data/SME/similar_user.pkl', 'wb') as f:
        pickle.dump(tensor.long(), f)
