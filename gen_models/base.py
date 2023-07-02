import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import Dataset, col_num
from rec_models import device


class BaseGenModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, rec_model, x):
        raise NotImplementedError

    def generate_train(self, rec_model, batch_size, lr, cold_lr, alpha):
        self.train()
        rec_model.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_func = nn.BCELoss(reduction='mean')

        for Da, Db in [['a', 'b'], ['c', 'd']]:
            train_loader_a = DataLoader(Dataset('../data/train_oneshot_' + Da), batch_size=batch_size, shuffle=False)
            train_loader_b = DataLoader(Dataset('../data/train_oneshot_' + Db), batch_size=batch_size, shuffle=False)

            tot_loss = 0.0
            tot_epoch = 0
            for (x_a, y_a), (x_b, y_b) in zip(train_loader_a, train_loader_b):
                x_a, y_a = x_a.to(device), y_a.float().to(device)
                x_b, y_b = x_b.to(device), y_b.float().to(device)

                embs = rec_model.get_embs(x_a)
                generate_emb, generate_idx = self(rec_model, x_a)
                embs[generate_idx] = generate_emb
                pred_a = rec_model.forward_with_embs(embs)
                loss_a = loss_func(pred_a, y_a)

                grad_a = torch.autograd.grad(loss_a, generate_emb, retain_graph=True)
                generate_emb = generate_emb - cold_lr * grad_a[0]
                embs[generate_idx] = generate_emb
                pred_b = rec_model.forward_with_embs(embs)
                loss_b = loss_func(pred_b, y_b)

                loss = loss_a * alpha + loss_b * (1 - alpha)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tot_loss += loss.item()
                tot_epoch += 1
            print('generator train loss:{:.4f}'.format(tot_loss / tot_epoch))

    def init_id_embedding(self, rec_model):
        # 每20个数据里包含的item是相同的
        test_loader = DataLoader(Dataset('../data/test_oneshot_a'), batch_size=20, shuffle=False)

        self.eval()
        rec_model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.float()
                generate_emb, generate_idx = self(rec_model, x[:1])

                col = col_num[generate_idx][0]
                idx = x[0, generate_idx]
                rec_model.embeddings[col].weight.data[idx].copy_(generate_emb.squeeze())
