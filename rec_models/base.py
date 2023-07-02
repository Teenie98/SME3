import time
import torch
import random
import numpy as np
import torch.nn as nn
from data import col_num, Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_seed(seed, cudnn=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # note: the below slows down the code but makes it reproducible
    if (seed is not None) and cudnn:
        torch.backends.cudnn.deterministic = True


# 某个属性可能同时有多个取值，获取multi-hot属性的emb
def get_mul_hot_emb(embeddings, mul_hot_index):
    # mul_hot_index.shape == (batch_size, feat_size)
    data_mask = (mul_hot_index > 0).float()

    mul_hot_emb = embeddings(mul_hot_index)
    mul_hot_emb = mul_hot_emb * data_mask.unsqueeze(-1)
    mul_hot_emb = torch.sum(mul_hot_emb, dim=1)
    mul_hot_emb = mul_hot_emb / torch.sum(data_mask, dim=1, keepdim=True)
    return mul_hot_emb


class BaseRecModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(num, self.args.embedding_size) for col, num in col_num
        })

        for key, val in self.embeddings.items():
            nn.init.normal_(val.weight, mean=0, std=1.0/self.args.embedding_size)

    def get_embs(self, x):
        embs = [self.embeddings[col_num[i][0]](x[:, i]) for i in range(6)]
        genres_emb = get_mul_hot_emb(self.embeddings['Genres'], x[:, 6:10])
        title_emb = get_mul_hot_emb(self.embeddings['Title'], x[:, 10:])
        return embs + [genres_emb, title_emb]

    def forward_with_embs(self, embs):
        raise NotImplementedError

    def forward(self, x):
        return self.forward_with_embs(self.get_embs(x))

    def predict(self, batch_size=1024):
        test_loader = DataLoader(Dataset('../data/test_test'), batch_size=batch_size)
        self.eval()
        with torch.no_grad():
            pred, target = [], []
            for x, y in test_loader:
                pred_y = self(x.to(device))
                pred.extend(pred_y.cpu().numpy().reshape(-1).tolist())
                target.extend(y.float().numpy().reshape(-1).tolist())
        auc = roc_auc_score(target, pred)
        logloss = log_loss(target, pred)
        return auc, logloss

    def pre_train(self, batch_size, lr, filepath='../data/big_train_main', epochs=1):
        train_loader = DataLoader(Dataset(filepath), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr)
        loss_func = nn.BCELoss(reduction='mean')
        tot_loss = 0.0
        tot_epoch = 0

        print('start pre-train...')
        self.train()
        start_time = time.time()

        for i in range(epochs):
            for x, y in train_loader:
                pred_y = self(x.to(device))
                loss = loss_func(pred_y, y.float().to(device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tot_loss += loss.item()
                tot_epoch += 1
            end_time = time.time()
            print('epoch {:2d}/{:2d} pre-train loss:{:.4f}, cost {:.2f}s'.format(
                i + 1, epochs, tot_loss / tot_epoch, end_time - start_time))
            start_time = end_time

    def warm_up_train(self, batch_size, lr, learnable_col):
        self.train()
        optimizer = torch.optim.Adam(self.embeddings[learnable_col].parameters(), lr)
        loss_func = nn.BCELoss(reduction='mean')
        tot_loss = 0.0
        tot_epoch = 0

        for idx in ['a', 'b', 'c']:
            train_loader = DataLoader(Dataset('../data/test_oneshot_' + idx), batch_size=batch_size, shuffle=True)
            for x, y in train_loader:
                pred_y = self(x.to(device))
                loss = loss_func(pred_y, y.float().to(device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tot_loss += loss.item()
                tot_epoch += 1
            print('warm-up {} train loss:{:.4f}'.format(idx, tot_loss / tot_epoch))
            test_auc, test_logloss = self.predict()
            print('test auc: {:.4f}, logloss: {:.4f}'.format(test_auc, test_logloss))
