from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Amazon, Coauthor, WebKB, Actor, WikipediaNetwork, ppi
from torch_geometric.datasets import HeterophilousGraphDataset
import torch_geometric.transforms as T

def load_node(dataname, dataset_dir):
    print(f'Dataloader: Loading {dataname}')
    dataset_mapping = {
        'Planetoid': ['PubMed', 'CiteSeer', 'Cora'],
        'Amazon': ['Computers', 'Photo'],
        'Reddit': ['Reddit'],
        'WikiCS': ['WikiCS'],
        'Flickr': ['Flickr'],
        'WebKB': ['Cornell', 'Wisconsin', 'Texas'],
        'WikipediaNetwork': ['Chameleon', 'Crocodile', 'Squirrel'],
        'Actor': ['Actor'],
        'HeterophilousGraphDataset': ['roman_empire', 'amazon_ratings', 'minesweeper', 'tolokers', 'questions'],
        'ogbn-arxiv': ['ogbn-arxiv']
    }
    
    # Load dataset
    if dataname in dataset_mapping['Planetoid']:
        dataset = Planetoid(root = dataset_dir, name = dataname, transform=T.NormalizeFeatures())
    elif dataname in dataset_mapping['Amazon']:
        dataset = Amazon(root = dataset_dir, name = dataname)
    elif dataname in dataset_mapping['Reddit']:
        dataset = Reddit(root = dataset_dir)
    elif dataname in dataset_mapping['WikiCS']:
        dataset = WikiCS(root = dataset_dir)
    elif dataname in dataset_mapping['Flickr']:
        dataset = Flickr(root = dataset_dir)
    elif dataname in dataset_mapping['WebKB']:
        dataset = WebKB(root = dataset_dir, name=dataname)
    elif dataname in dataset_mapping['Actor']:
        dataset = Actor(root = dataset_dir + '/Actor', transform=T.NormalizeFeatures())
    elif dataname in dataset_mapping['WikipediaNetwork']:
        dataset = WikipediaNetwork(root = dataset_dir, name=dataname, transform=T.NormalizeFeatures())
    elif dataname in dataset_mapping['HeterophilousGraphDataset']:
        dataset = HeterophilousGraphDataset(root=dataset_dir, name=dataname)
    elif dataname in dataset_mapping['ogbn-arxiv']:
        dataset = PygNodePropPredDataset(name ='ogbn-arxiv', root = dataset_dir)
        data = dataset[0]
        input_dim = data.x.shape[1]
        out_dim = dataset.num_classes
    else:
        raise ValueError(f'Unknown dataset: {dataname}')
    
    # Unified processing
    if dataname != 'ogbn-arxiv':
        data = dataset[0]
        input_dim = dataset.num_features
        out_dim = dataset.num_classes

    print(data)
    print(f'Dataloader: Loading {dataname} success.')
    print(f'—————————————————————————————\n')
    
    return data, input_dim, out_dim
