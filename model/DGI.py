import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj
from load_data import *

class GCN(nn.Module):
    def __init__(self, 
                 in_dim, 
                 hid_dim, 
                 num_layers=2,
                 activation='relu',
                 dropout=0.5,
                 jk_mode='last'):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout
        self.jk_mode = jk_mode
        
        self.layers.append(GCNConv(in_dim, hid_dim))
        for _ in range(num_layers-1):
            self.layers.append(GCNConv(hid_dim, hid_dim))
            
        self.act = nn.ReLU() if activation == 'relu' else nn.PReLU()
        
    def forward(self, x, edge_index, edge_weight=None):
        xs = []
        
        if self.training and self.dropout > 0:
            edge_index, edge_weight = dropout_adj(
                edge_index, 
                edge_weight,
                p=self.dropout
            )
            
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index, edge_weight)
            if i != len(self.layers)-1:  
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
            
        if self.jk_mode == 'last':
            return x
        elif self.jk_mode == 'sum':
            return torch.stack(xs, dim=0).sum(dim=0)
        elif self.jk_mode == 'max':
            return torch.stack(xs, dim=0).max(dim=0)[0]
        elif self.jk_mode == 'cat':
            return torch.cat(xs, dim=-1)
        else:
            raise ValueError(f"Invalid jk_mode: {self.jk_mode}")

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.unsqueeze(1) 
        c_x = c_x.expand(-1, h_pl.size(1))  

        sc_1 = self.f_k(h_pl, c_x).squeeze(1)
        sc_2 = self.f_k(h_mi, c_x).squeeze(1)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 0)

        return logits

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)

class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, 2, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, weight, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, weight)

        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj, weight)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq, adj, weight, msk):
        h_1 = self.gcn(seq, adj, weight)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

def DGI_process(nb_nodes, x, batch_size = 1):
    idx = np.random.permutation(nb_nodes)
    shuf_x = x[idx, :]

    lbl_1 = torch.ones(batch_size, nb_nodes)
    lbl_2 = torch.zeros(batch_size, nb_nodes)
    lbl = torch.cat((lbl_1, lbl_2), 1).squeeze(0)
    return shuf_x, lbl