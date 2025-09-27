import sys
import os
import time
import torch
import torchmetrics
import argparse
import random
import pickle
import warnings
import scipy.sparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch import optim
from tqdm import tqdm
from load_data import load_node
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, degree, coalesce, dropout_adj
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
from model import *
from utils import *
warnings.filterwarnings("ignore", category=UserWarning, message="Sparse CSR tensor support is in beta state")

def run(args, device):
    ###################################################################### 1. Preprocessing Stage
    data, input_dim, output_dim = load_node(args.dataset, args.dataset_dir)
    edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float32)
    
    edge_index, edge_weight = add_self_loops(data.edge_index, edge_weight)
    edge_weight = normalize_edge(edge_index, edge_weight, data.num_nodes).to(device)
    edge_index = edge_index.to(device)
    data = data.to(device)
    
    ###################################################################### 2. Pretraining Stage
    model = None
    loss_func = None
    if args.model == 'DGI':
        model = DGI(input_dim, args.hid_dim, 'prelu').to(device)
        loss_func = nn.BCEWithLogitsLoss()
    elif args.model == 'GraphMAE':
        model = build_model(num_hidden = args.hid_dim, num_features = input_dim).to(device)
    elif args.model == 'GRACE':
        encoder = Encoder(input_dim, args.hid_dim, nn.PReLU(),
                      base_model=GCNConv, k=2).to(device)
        model = Model(encoder, args.hid_dim, args.hid_dim, 0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    loss = None
    cnt_wait = 0
    best = 1e9
    best_t = 0
    with tqdm(total=args.epochs, desc='(T)') as pbar:
        for epoch in range(args.epochs + 1):
            model.train()
            optimizer.zero_grad()
    
            if args.model == 'DGI':
                shuf_x, lbl = DGI_process(data.num_nodes, data.x)
                shuf_x = shuf_x.to(device)
                lbl = lbl.to(device)
                logits = model(data.x, shuf_x, edge_index, edge_weight, None, None, None)
                loss = loss_func(logits, lbl)    
            elif args.model == 'GraphMAE':
                loss, _ = model(data.x, edge_index, edge_weight)
            elif args.model == 'GRACE':
                edge_index_1, edge_weight_1 = dropout_adj(edge_index, edge_weight, p=0.2)
                edge_index_2, edge_weight_2 = dropout_adj(edge_index, edge_weight, p=0.2)
                x_1 = drop_feature(data.x, 0.2)
                x_2 = drop_feature(data.x, 0.2)
                z1 = model(x_1, edge_index_1, edge_weight_1)
                z2 = model(x_2, edge_index_2, edge_weight_2)
                loss = model.loss(z1, z2, batch_size=0)
            pbar.set_postfix({'loss': loss})
            pbar.update()
                
            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                model_save_path = f'./pretrain_model/{args.model}/{args.dataset}/lr_{args.lr}_weightdecay_{args.wd}_hid_dim_{args.hid_dim}.pkl'
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  
                torch.save(model.state_dict(), model_save_path)
            else:
                cnt_wait += 1
    
            if cnt_wait == args.patience:
                print('Early stopping!')
                break
    
            loss.backward()
            optimizer.step()

    ###################################################################### 3. Prompt Stage
    print('Loading {}th epoch'.format(best_t))
    print('{} is upload successfully!'.format(args.model))

    create_few_data_folder(args, data, output_dim) # Create k-shot file
    
    embeds = None
    classifier = None
    prompt = None
    down_optim = None
    down_loss = nn.CrossEntropyLoss()

    model.eval()
    test_accs = []
    f1s = []
    rocs = []
    prcs = []
    loss = 0
    for trail in range(1, args.trails + 1):
        torch.cuda.empty_cache()

        if args.model == 'DGI':
            model.load_state_dict(torch.load(model_save_path, weights_only=True))
            classifier = LogReg(args.hid_dim, output_dim).to(device)
            embeds, _ = model.embed(data.x, edge_index, edge_weight, None)
        elif args.model == 'GraphMAE':
            model.load_state_dict(torch.load(model_save_path, weights_only=True))
            classifier = LogReg(args.hid_dim, output_dim).to(device)
            embeds = model.embed(data.x, edge_index, edge_weight)
        elif args.model == 'GRACE':
            model.load_state_dict(torch.load(model_save_path, weights_only=True))
            classifier = LogReg(args.hid_dim, output_dim).to(device)
            embeds = model.embed(data.x, edge_index, edge_weight)

        prompt = UniPrompt(data.x, args.k, 'cosine', 6, data.num_nodes).to(device)
        model_param_group = []
        model_param_group.append({"params": prompt.parameters()})
        model_param_group.append({"params": classifier.parameters()})
        down_optim = torch.optim.Adam(model_param_group, lr = args.down_lr, weight_decay = args.down_wd)
        train_idx = torch.load("./Experiment/sample_data/{}/{}_shot/{}/train_idx.pt".format(args.dataset, 
                               args.shot, trail), weights_only=False).type(torch.long).to(device)
        train_lbls = torch.load("./Experiment/sample_data/{}/{}_shot/{}/train_labels.pt".format(args.dataset, 
                               args.shot, trail), weights_only=False).type(torch.long).squeeze().to(device)
        test_idx = torch.load("./Experiment/sample_data/{}/{}_shot/{}/test_idx.pt".format(args.dataset, 
                               args.shot, trail), weights_only=False).type(torch.long).to(device)
        test_lbls = torch.load("./Experiment/sample_data/{}/{}_shot/{}/test_labels.pt".format(args.dataset, 
                               args.shot, trail), weights_only=False).type(torch.long).squeeze().to(device)

    ###################################################################### 4. Few-shot Adapting
        best_loss = 1e9
        cnt_wait = 0
        best_t = 0
        best_prompt_state = None
        best_log_state = None
        comb_index = edge_index
        comb_weight = edge_weight
        with tqdm(total=args.down_epochs, desc='(T)') as pbar:
            for epoch in range(args.down_epochs):
                prompt.train()
                classifier.train()
                down_optim.zero_grad()
                prompt_index, prompt_weight = prompt()
                comb_index, comb_weight = prompt.edge_fuse(comb_index, comb_weight, prompt_index, prompt_weight, args.tau)
                
                embeds = None
                if args.model == 'DGI':
                    embeds, _ = model.embed(data.x, comb_index, comb_weight, None)
                elif args.model == 'GraphMAE':
                    embeds = model.embed(data.x, comb_index, comb_weight)
                elif args.model == 'GRACE':
                    embeds = model.embed(data.x, comb_index, comb_weight)
                    
                logits = classifier(embeds)
                loss = down_loss(logits[train_idx], train_lbls)
                pbar.set_postfix({'loss': loss})
                pbar.update()
    
                if loss < best_loss:
                    best_loss = loss
                    best_t = epoch
                    cnt_wait = 0
                    best_prompt_state = prompt.state_dict()
                    best_classifier_state = classifier.state_dict()
                else:
                    cnt_wait += 1
                if cnt_wait == args.patience:
                    print(f'Early stopping at epoch {epoch}!')
                    break
                loss.backward()
                down_optim.step()

    ###################################################################### 5. Evaluation
        print('Loading {}th epoch'.format(best_t))
        prompt.load_state_dict(best_prompt_state)
        classifier.load_state_dict(best_classifier_state)
        prompt.eval()
        classifier.eval()
        with torch.no_grad():
            x_t = data.x
            embeds = None
            if args.model == 'DGI':
                embeds, _ = model.embed(x_t, comb_index, comb_weight, None)
            elif args.model == 'GraphMAE':
                embeds = model.embed(x_t, comb_index, comb_weight)
            elif args.model == 'GRACE':
                embeds = model.embed(x_t, comb_index, comb_weight)
            
            logits = classifier(embeds)
            test_acc, ma_f1, roc, prc = NodeEva(logits, test_idx, data, output_dim, device)
            print(f"Final True Accuracy: {test_acc:.4f} | Macro F1 Score: {ma_f1:.4f} | AUROC: {roc:.4f} | AUPRC: {prc:.4f}" )     
            test_accs.append(test_acc)
            f1s.append(ma_f1)
            rocs.append(roc)
            prcs.append(prc)
    mean_test_acc = np.mean(test_accs)
    std_test_acc = np.std(test_accs)    
    mean_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)   
    mean_roc = np.mean(rocs)
    std_roc = np.std(rocs)   
    mean_prc = np.mean(prcs)
    std_prc = np.std(prcs)
    print('Acc List', test_accs) 
    print(" Final best | test Accuracy {:.4f} ± {:.4f}(std)".format(mean_test_acc, std_test_acc))   
    print(" Final best | test F1 {:.4f} ± {:.4f}(std)".format(mean_f1, std_f1))   
    print(" Final best | AUROC {:.4f} ± {:.4f}(std)".format(mean_roc, std_roc))   
    print(f"Task completed: {args.model} using {args.prompt} on {args.dataset} at {args.shot}-shot")
    with open(args.log_dir + args.dataset + ".txt", 'a') as f:
        f.write(" Final best | test Accuracy {:.4f} ± {:.4f}(std)".format(mean_test_acc, std_test_acc))
        f.write(" Final best | test F1 {:.4f} ± {:.4f}(std)".format(mean_f1, std_f1))
        f.write(" Final best | AUROC {:.4f} ± {:.4f}(std)".format(mean_roc, std_roc))
        f.write(f"Task completed: {args.model} using {args.prompt} on {args.dataset} at {args.shot}-shot \n\n")
    print('----------------- \n')   
    return None  

parser = argparse.ArgumentParser(description='UniPrompt Pipeline') 
parser.add_argument('--dataset', type=str, default='Cora', help='dataset used in downstream task') 
parser.add_argument('--dataset_dir', type=str, default='./datasets/', help='downstream dataset repo position') 
parser.add_argument('--log_dir', type=str, default='./log/', help='result position') 
parser.add_argument('--seed', type=int, default=42, help='seed') 
###################################################################### Pretraining 
parser.add_argument('--lr', type=float, default=0.001, help='learning rate') 
parser.add_argument('--wd', type=float, default=0.00001, help='weight decay') 
parser.add_argument('--epochs', type=int, default=500, help='maximum number of epochs') 
parser.add_argument('--patience', type=int, default=20, help='early stopping') 
parser.add_argument('--model', type=str, default='GRACE', help='pretrain method') 
parser.add_argument('--hid_dim', type=int, default=128, help='pretrained_model output size')
###################################################################### Prompting 
parser.add_argument('--prompt', type=str, default='UniPrompt', help='prompt method') 
parser.add_argument('--shot', type=int, default=1, help='number of labeled data in each class')
parser.add_argument('--trails', type=int, default=10, help='maximum number of traning') 
parser.add_argument('--down_lr', type=float, default=0.001, help='learning rate in dowm-stream tasks') 
parser.add_argument('--down_wd', type=float, default=0.00001, help='weight decay in down-stream tasks') 
parser.add_argument('--down_epochs', type=int, default=500, help='maximum number of epochs in down-stream tasks') 
parser.add_argument('--k', type=int, default=50, help='number of neighbors in new edge_index') 
parser.add_argument('--tau', type=float, default=0.99, help='') 
args = parser.parse_args()
args.log_dir = args.log_dir + '/' + args.model + '/' + args.prompt + '/'
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
with open(args.log_dir + args.dataset + ".txt", 'a') as f:
    f.write('**'*40+'\n')
    f.write(str(args) + '\n')
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
run(args, device)   
