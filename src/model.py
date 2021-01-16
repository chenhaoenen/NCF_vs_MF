# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2020-12-05 13:34
# Description:  
#--------------------------------------------
import torch
from torch import nn


def weight_init(layers):
    for layer in layers:
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
        if isinstance(layer, nn.Embedding):
            layer.weight.data.normal_(0, 0.01)

class MF(nn.Module):
    def __init__(self, num_users, num_items, dim):
        super(MF, self).__init__()
        self.user_embedding = nn.Embedding(num_users+1, dim)
        self.item_embedding = nn.Embedding(num_items+1, dim)
        self.user_bias = nn.Embedding(num_users+1, 1)
        self.item_bias = nn.Embedding(num_items+1, 1)
        self.bias = nn.Parameter(torch.Tensor(1))
        self.activation = nn.Sigmoid()

        self.init_weight()

    def init_weight(self):
        weight_init([self.user_embedding, self.item_embedding])


    def forward(self, user, item):
        u = user.long(); i = item.long()

        u_emb = self.user_embedding(u)
        i_emb = self.item_embedding(i)
        u_b = self.user_bias(u).view(-1)
        i_b = self.item_bias(i).view(-1)

        out = torch.sum(torch.mul(u_emb, i_emb), dim=-1)
        out = out + u_b + i_b + self.bias
        logits = self.activation(out)

        return logits

class MLP(nn.Module):
    def __init__(self, num_users, num_items, dim, layer_hiddens, dropout=0.2):
        super(MLP, self).__init__()
        self.user_embedding = nn.Embedding(num_users+1, dim)
        self.item_embedding = nn.Embedding(num_items+1, dim)

        mlps = [nn.Linear(2*dim, layer_hiddens[0]), nn.ReLU(inplace=True)]
        for i in range(1, len(layer_hiddens)):
            mlps.append(nn.Linear(layer_hiddens[i-1], layer_hiddens[i]))
            mlps.append(nn.ReLU(inplace=True))
            mlps.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*mlps)

        self.dense = nn.Linear(layer_hiddens[-1], 1)
        self.activate = nn.Sigmoid()

        self.init_weight()

    def init_weight(self):
        weight_init([self.user_embedding, self.item_embedding])
        weight_init(self.mlp)

        torch.nn.init.kaiming_uniform_(self.dense.weight)

    def forward(self, user, item):
        u = user.long(); i = item.long()

        u_emb = self.user_embedding(u)
        i_emb = self.item_embedding(i)
        out = torch.cat([u_emb, i_emb], dim=-1)
        out = self.mlp(out)
        out = self.dense(out)
        logits = self.activate(out).view(-1)

        return logits

class GMF(nn.Module):
    def __init__(self, num_users, num_items, dim):
        super(GMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users+1, dim)
        self.item_embedding = nn.Embedding(num_items+1, dim)

        self.dense = nn.Linear(dim, 1)
        self.activation = nn.Sigmoid()

        self.init_weight()

    def init_weight(self):
        weight_init([self.user_embedding, self.item_embedding])
        torch.nn.init.kaiming_uniform_(self.dense.weight)

    def forward(self, user, item):
        u = user.long(); i = item.long()

        u_emb = self.user_embedding(u)
        i_emb = self.item_embedding(i)

        out = torch.mul(u_emb, i_emb)
        out = self.dense(out).view(-1)
        logits = self.activation(out)

        return logits

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, dim, layer_hiddens, dropout=0.2):
        super(NeuMF, self).__init__()

        #gmf
        self.user_embedding_gmf = nn.Embedding(num_users+1, dim)
        self.item_embedding_gmf = nn.Embedding(num_items+1, dim)

        #mlps
        self.user_embedding_mlp = nn.Embedding(num_users+1, dim)
        self.item_embedding_mlp = nn.Embedding(num_items+1, dim)

        mlps = [nn.Linear(2*dim, layer_hiddens[0]), nn.ReLU(inplace=True)]
        for i in range(1, len(layer_hiddens)):
            mlps.append(nn.Linear(layer_hiddens[i-1], layer_hiddens[i]))
            mlps.append(nn.ReLU(inplace=True))
            mlps.append(nn.Dropout(p=dropout))
        self.mlp = nn.Sequential(*mlps)

        #concate
        self.dense = nn.Linear(dim+layer_hiddens[-1], 1)
        self.activation = nn.Sigmoid()

        self.init_weight()

    def init_weight(self):
        weight_init([self.user_embedding_gmf, self.item_embedding_gmf, self.user_embedding_mlp, self.item_embedding_mlp.weight])
        weight_init(self.mlp)

        torch.nn.init.kaiming_uniform_(self.dense.weight)


    def forward(self, user, item):
        u = user.long(); i = item.long()

        #gmf
        u_emb_gmf = self.user_embedding_gmf(u)
        i_emb_gmf = self.item_embedding_gmf(i)
        out_gmf = torch.mul(u_emb_gmf, i_emb_gmf)

        #mlp
        u_emb_mlp = self.user_embedding_mlp(u)
        i_emb_mlp = self.item_embedding_mlp(i)
        x = torch.cat([u_emb_mlp, i_emb_mlp], dim=-1)
        out_mlp = self.mlp(x)

        #concat
        out = torch.cat([out_gmf, out_mlp], dim=-1)
        out = self.dense(out).view(-1)
        logits = self.activation(out)

        return logits
