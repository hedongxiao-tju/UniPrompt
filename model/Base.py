import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv
from load_data import *

class GCNConv_dense(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(GCNConv_dense, self).__init__()
        self.layer = nn.Linear(in_dim, hid_dim)

    def init_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, adj):
        z = self.layer(x)
        z = torch.matmul(adj, z)
        return z

class GCNConv_sparse(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNConv_sparse, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        return torch.sparse.mm(adj, self.linear(x))

class SparseDropout(nn.Module):
    def __init__(self, dprob = 0.5):
        super(SparseDropout, self).__init__()
        self.kprob = 1 - dprob 
        
    def forward(self, adj):
        mask = ((torch.rand(adj._values().size()) + self.kprob).floor()).type(torch.bool)
        rc = adj._indices()[:,mask]
        val = adj._values()[mask]*(1.0 / self.kprob)
        return torch.sparse_coo_tensor(rc, val, adj.shape)

class GCN(nn.Module):
    def __init__(self, layers, in_dim, hid_dim, activation, dropout = 0.1, sparse = True):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.sparse = sparse
        self.act = nn.PReLU() if activation == 'prelu' else nn.ReLU()
        self.encoder_layers = nn.ModuleList()
        if self.sparse:
            self.encoder_layers.append(GCNConv_sparse(in_dim, hid_dim))
            for _ in range(layers - 1):
                self.encoder_layers.append(GCNConv_sparse(hid_dim, hid_dim))
        else:
            self.encoder_layers.append(GCNConv_dense(in_dim, hid_dim))
            for _ in range(layers - 1):
                self.encoder_layers.append(GCNConv_dense(hid_dim, hid_dim))
        if self.sparse:
            self.drop_adj = SparseDropout(dprob=self.dropout)
        else:
            self.drop_adj = nn.Dropout(p=self.dropout)
       

    def forward(self, x, adj):                    
        for conv in self.encoder_layers:
            x = conv(x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x