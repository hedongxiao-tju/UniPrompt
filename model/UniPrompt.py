import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree, coalesce
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
from .Base import *

class UniPrompt(nn.Module):
    def __init__(self, x, k, metric, alpha, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.alpha = alpha
        
        knn_adj = kneighbors_graph(x.cpu().numpy(), k, metric=metric)
        knn_adj = knn_adj.tocoo()
        
        edge_index = torch.tensor(np.vstack([knn_adj.row, knn_adj.col]), dtype=torch.long)
        edge_attr = torch.tensor(knn_adj.data, dtype=torch.float32)
        
        self.base_edge_index = edge_index.to(x.device)
        self.edge_weight = nn.Parameter(edge_attr.to(x.device))

    def forward(self):
        weights = F.elu(self.edge_weight * self.alpha - self.alpha) + 1
        return self.base_edge_index, weights

    def edge_fuse(self, index_ori, weight_ori, index_pt, weight_pt, tau):
        weight_ori = weight_ori * tau 
        weight_pt = weight_pt * (1 - tau)      
        
        comb_index = torch.cat([index_ori, index_pt], dim=1)
        comb_weight = torch.cat([weight_ori.detach(), weight_pt])
        return coalesce(comb_index, comb_weight, reduce='add')
    