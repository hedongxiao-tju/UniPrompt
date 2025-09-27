import sys
import os
import time
import torch
import torchmetrics
import argparse
import random
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch import optim
from tqdm import tqdm
from load_data import load_node
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import add_self_loops, degree, coalesce

def normalize_edge(edge_index, edge_weight, num_nodes):
    row, col = edge_index
    deg = degree(row, num_nodes, dtype=torch.float32)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return norm

def edge_combine(edge_index1, edge_weight1, edge_index2, edge_weight2, tau, device):
    combined_index = torch.cat([edge_index1, edge_index2], dim=1)
    combined_weight = torch.cat([edge_weight1 * tau, edge_weight2 * (1 - tau)])
    
    from torch_geometric.utils import coalesce
    return coalesce(combined_index, combined_weight, reduce='add')


def NodeEva(out, test_idx, data, num_class, device):
    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
    macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
    auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_class).to(device)
    auprc = torchmetrics.classification.AveragePrecision(task="multiclass", num_classes=num_class).to(device)

    accuracy.reset()
    macro_f1.reset()
    auroc.reset()
    auprc.reset()

    pred = out.argmax(dim=1)
    acc = accuracy(pred[test_idx], data.y[test_idx])
    ma_f1 = macro_f1(pred[test_idx], data.y[test_idx])
    roc = auroc(out[test_idx], data.y[test_idx])
    prc = auprc(out[test_idx], data.y[test_idx])
       
    return acc.item(), ma_f1.item(), roc.item(), prc.item()

def create_few_data_folder_FUG(args, data, output_dim):
    k = args.shot  # shot_num 
    task_num = args.trails  # task_num 
    for task_index in range(1, task_num + 1):
        k_shot_folder = './Experiment/sample_data/'+ args.test_dataset +'/' + str(k) +'_shot'
        os.makedirs(k_shot_folder, exist_ok=True)

        folder = os.path.join(k_shot_folder, str(task_index))
        if not os.path.exists(folder):
            os.makedirs(folder)
            node_sample_and_save(data, k, folder, output_dim)
            print(str(k) + ' shot ' + str(task_index) + ' th is saved!!')

def create_few_data_folder(args, data, output_dim):
    k = args.shot  # shot_num 
    task_num = args.trails  # task_num 
    for task_index in range(1, task_num + 1):
        k_shot_folder = './Experiment/sample_data/'+ args.dataset +'/' + str(k) +'_shot'
        os.makedirs(k_shot_folder, exist_ok=True)

        folder = os.path.join(k_shot_folder, str(task_index))
        if not os.path.exists(folder):
            os.makedirs(folder)
            node_sample_and_save(data, k, folder, output_dim)
            print(str(k) + ' shot ' + str(task_index) + ' th is saved!!')

def node_sample_and_save(data, k, folder, num_classes):
    labels = data.y.to('cpu')

    num_test = int(0.9 * data.num_nodes)
    if num_test < 1000:
        num_test = int(0.7 * data.num_nodes)
    test_idx = torch.randperm(data.num_nodes)[:num_test]
    test_labels = labels[test_idx]
    
    remaining_idx = torch.randperm(data.num_nodes)[num_test:]
    remaining_labels = labels[remaining_idx]
    
    train_idx = torch.cat([remaining_idx[remaining_labels == i][:k] for i in range(num_classes)])
    shuffled_indices = torch.randperm(train_idx.size(0))
    train_idx = train_idx[shuffled_indices]
    train_labels = labels[train_idx]

    torch.save(train_idx, os.path.join(folder, 'train_idx.pt'))
    torch.save(train_labels, os.path.join(folder, 'train_labels.pt'))
    torch.save(test_idx, os.path.join(folder, 'test_idx.pt'))
    torch.save(test_labels, os.path.join(folder, 'test_labels.pt'))




    