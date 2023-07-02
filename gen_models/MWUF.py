import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from data import Dataset, col_num, read_pkl
from rec_models import device
from gen_models.base import BaseGenModel
from torch.utils.data import DataLoader


class MetaScaling(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.layers = nn.Sequential(
            nn.Linear(args.embedding_size * 3, 16),
            nn.ReLU(),
            nn.Linear(16, args.embedding_size)
        )

    def forward(self, rec_model, x):
        embs = rec_model.get_embs(x)
        year_emb, genres_emb, title_emb = embs[5], embs[6], embs[7]
        embs = torch.cat([year_emb, genres_emb, title_emb], dim=1)
        scaling_emb = self.layers(embs)
        return scaling_emb


class MetaShifting(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.layers = nn.Sequential(
            nn.Linear(args.embedding_size, 16),
            nn.ReLU(),
            nn.Linear(16, args.embedding_size)
        )
        self.act_users = read_pkl('data/MWUF/item_actuser_list.pkl')

    def forward(self, rec_model, x):
        users = torch.tensor(np.array([self.act_users[key] for key in x[:, 0].tolist()])).to(device)
        avg_users_embedding = rec_model.embeddings['UserID'](users).mean(1)
        shifting_emb = self.layers(avg_users_embedding)
        return shifting_emb


def get_unique_idlst(name):
    data = read_pkl('../data/'+name+'.pkl')
    list = data.MovieID.unique()
    return list


class MWUF(BaseGenModel):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.loss_func = nn.BCELoss(reduction='mean')

    def meta_network_train(self, scaling_net, shifting_net, rec_model):
        old_items = torch.tensor(get_unique_idlst('train_oneshot_a')).to(device)
        old_items_emb = rec_model.embeddings['MovieID'](old_items).mean(dim=0)

        # 所有物品初始化为老物品表示的均值
        new_item_embeddings = nn.Embedding(col_num[0][1], self.args.embedding_size).to(device)
        new_item_embeddings.weight.data[:] = old_items_emb
        new_items = get_unique_idlst('test_oneshot_a')
        rec_model.embeddings['MovieID'].weight.data[new_items] = old_items_emb

        meta_model_optimizer = torch.optim.Adam([{'params': scaling_net.parameters()},
                                                 {'params': shifting_net.parameters()}], lr=self.args.meta_learning_rate)
        embedding_optimizer = torch.optim.Adam(new_item_embeddings.parameters(), lr=self.args.meta_learning_rate)

        for D in ['a', 'b', 'c', 'd']:

            train_dataloader_a = DataLoader(Dataset('../data/train_oneshot_' + D),
                                            batch_size=self.args.generator_train_batch_size, shuffle=True)

            rec_model.train()
            scaling_net.train()
            shifting_net.train()
            for (x, y) in train_dataloader_a:
                x, y = x.to(device), y.float().to(device)

                cold_emb = new_item_embeddings(x[:, 0])
                embs = rec_model.get_embs(x)
                embs[0] = cold_emb
                pred_cold = rec_model.forward_with_embs(embs)
                loss_cold = self.loss_func(pred_cold, y)

                embedding_optimizer.zero_grad()
                loss_cold.backward()
                embedding_optimizer.step()

                cold_emb = new_item_embeddings(x[:, 0])
                scaling_emb = scaling_net(rec_model, x)
                shifting_emb = shifting_net(rec_model, x)

                warm_emb = torch.mul(scaling_emb, cold_emb) + shifting_emb
                embs = rec_model.get_embs(x)
                embs[0] = warm_emb
                pred_warm = rec_model.forward_with_embs(embs)
                loss_warm = self.loss_func(pred_warm, y)

                meta_model_optimizer.zero_grad()
                loss_warm.backward()
                meta_model_optimizer.step()

    def warm_up_train(self, rec_model, scaling_net, shifting_net, batch_size, lr, learnable_col):
        rec_model.train()
        optimizer = torch.optim.Adam(rec_model.embeddings[learnable_col].parameters(), lr)
        loss_func = nn.BCELoss(reduction='mean')
        tot_loss = 0.0
        tot_epoch = 0

        for idx in ['a', 'b', 'c']:
            train_loader = DataLoader(Dataset('../data/test_oneshot_' + idx), batch_size=batch_size, shuffle=True)
            for x, y in train_loader:
                x, y = x.to(device), y.float().to(device)
                cold_emb = rec_model.embeddings['MovieID'](x[:, 0])
                scaling_emb = scaling_net(rec_model, x)
                shifting_emb = shifting_net(rec_model, x)

                warm_emb = torch.mul(scaling_emb, cold_emb) + shifting_emb
                embs = rec_model.get_embs(x)
                embs[0] = warm_emb
                pred_y = rec_model.forward_with_embs(embs)
                loss = loss_func(pred_y, y.float().to(device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tot_loss += loss.item()
                tot_epoch += 1
            print('warm-up {} train loss:{:.4f}'.format(idx, tot_loss / tot_epoch))
            test_auc, test_logloss = rec_model.predict()
            print('test auc: {:.4f}, logloss: {:.4f}'.format(test_auc, test_logloss))

if __name__ == '__main__':
    testa_data = read_pkl('../data/test_oneshot_a.pkl')
    testb_data = read_pkl('../data/test_oneshot_b.pkl')
    testc_data = read_pkl('../data/test_oneshot_c.pkl')
    traina_data = read_pkl('../data/train_oneshot_a.pkl')
    trainb_data = read_pkl('../data/train_oneshot_b.pkl')
    trainc_data = read_pkl('../data/train_oneshot_c.pkl')
    traind_data = read_pkl('../data/train_oneshot_d.pkl')

    concat_data = pd.concat((traina_data, trainb_data, trainc_data, traind_data, testa_data, testb_data, testc_data))
    item_actuser_list = {}
    for item, records in concat_data.groupby('MovieID'):
        item_actuser_list[item] = records.UserID[:].to_numpy()

    filedir = "./data/MWUF/"
    os.makedirs(filedir, exist_ok=True)

    with open('data/MWUF/item_actuser_list.pkl', 'wb') as f:
        pickle.dump(item_actuser_list, f)
