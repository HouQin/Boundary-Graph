# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:48:06 2020

@author: LENOVO
"""

import numpy as np
import torch
import sys
from inout import *
import os
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import sys
import pickle as pkl
import numpy as np
import json
import itertools
import networkx as nx
import os.path
from sparsegraph import SparseGraph

from torch_geometric.data import Data
from torch_geometric.datasets import WikiCS, Amazon, Planetoid
from torch_geometric.utils import remove_self_loops
import torch_geometric.transforms as T

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def train_test_split(graph_labels_dict, labelrate):

    idx_train = []
    idx_test = []
    idx_val = []
    val_count = 0

    n = len(graph_labels_dict)
    class_num = max(graph_labels_dict.values()) + 1
    train_num = class_num * labelrate

    idx = list(range(n))

    count = [0] * class_num
    for i in range(len(idx)):
        l = graph_labels_dict[idx[i]]
        if count[l] < labelrate:
            idx_train.append(idx[i])
            count[l] = count[l] + 1
        elif len(idx_train) == train_num and val_count < 500:
            idx_val.append(idx[i])
            val_count = val_count + 1
    for i in range(len(idx)-1000, len(idx)):
        idx_test.append(idx[i])
    idx_np = {}
    idx_np['train'] = idx_train
    idx_np['stopping'] = idx_val
    idx_np['valtest'] = idx_test

    return idx_np


def train_test_split_acm(graph_labels_dict, labelrate):

    idx_train = []
    idx_test = []
    idx_val = []
    val_count = 0

    n = len(graph_labels_dict)
    class_num = max(graph_labels_dict.values()) + 1
    train_num = class_num * labelrate

    idx = list(range(n))

    #random
    np.random.seed(20)
    np.random.shuffle(idx)
    count = [0] * class_num
    for i in range(len(idx)):
        l = graph_labels_dict[idx[i]]
        if count[l] < labelrate:
            idx_train.append(idx[i])
            count[l] = count[l] + 1
        elif len(idx_train) == train_num and val_count < 500:
            idx_val.append(idx[i])
            val_count = val_count + 1
    for i in range(len(idx)-1000, len(idx)):
        idx_test.append(idx[i])
    idx_np = {}
    idx_np['train'] = idx_train
    idx_np['stopping'] = idx_val
    idx_np['valtest'] = idx_test

    return idx_np


def load_new_data_wiki(labelrate):
    data = json.load(open('./data/wiki/data.json'))

    features = np.array(data['features'])
    labels = np.array(data['labels'])

    n_feats = features.shape[1]

    graph_node_features_dict = {}
    graph_labels_dict = {}
    for index in range(len(features)):
        graph_node_features_dict[index] = features[index]
        graph_labels_dict[index] = int(labels[index])

    g = nx.DiGraph()

    for index in range(len(features)):
        g.add_node(index, features=graph_node_features_dict[index],
                   label=graph_labels_dict[index])
    edge_list = list(itertools.chain(*[[(i, nb) for nb in nbs] for i, nbs in enumerate(data['links'])]))

    for edge in edge_list:
        g.add_edge(int(edge[0]), int(edge[1]))

    sG = networkx_to_sparsegraph_floatfeature(g, n_feats)

    idx_np = train_test_split(graph_labels_dict, labelrate)

    return sG, idx_np


def load_new_data_acm(labelrate):
    graph_adjacency_list_file_path = os.path.join('./data/acm/acm_PAP.edge')
    graph_node_features_file_path = os.path.join('./data/acm/acm.feature')
    graph_labels_file_path = os.path.join('./data/acm/acm.label')

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}
    index = 0
    with open(graph_node_features_file_path) as graph_node_features_file:
        for line in graph_node_features_file:
            assert (index not in graph_node_features_dict)
            graph_node_features_dict[index] = np.array(line.strip('\n').split(' '), dtype=np.uint8)
            index = index + 1
    index = 0
    with open(graph_labels_file_path) as graph_labels_file:
        for line in graph_labels_file:
            assert (index not in graph_labels_dict)
            graph_labels_dict[index] = int(line.strip('\n'))
            G.add_node(index , features=graph_node_features_dict[index], label=graph_labels_dict[index])
            index = index + 1

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        for line in graph_adjacency_list_file:
            line = line.rstrip().split(' ')
            assert (len(line) == 2)
            G.add_edge(int(line[0]), int(line[1]))

    sG = networkx_to_sparsegraph_acm(G, 1870)

    
    idx_np = train_test_split_acm(graph_labels_dict, labelrate)

    return sG, idx_np


def load_data_tkipf(dataset_str):

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./data/tkipf_data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./data/tkipf_data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]


    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    return adj, features, labels, idx_test, idx_train, idx_val


def load_new_data_tkipf(dataset_name, feature_dim, labelrate):
    adj, features, labels, idx_test, idx_train, idx_val = load_data_tkipf(dataset_name)
    labels = np.argmax(labels, axis=-1)
    features = features.todense()
    G = nx.DiGraph(adj)

    for index in range(len(labels)):
        G.add_node(index , features=features[index], label=labels[index])
    if dataset_name == 'pubmed':
        sG = networkx_to_sparsegraph_floatfeature(G, feature_dim)
    else:
        sG = networkx_to_sparsegraph_intfeature(G, feature_dim)

    graph_labels_dict = {}
    for index in range(len(labels)):
        graph_labels_dict[index] = int(labels[index])

    idx_np = {}
    if labelrate == 20:
        idx_np['train'] = idx_train
        idx_np['stopping'] = idx_val
        idx_np['valtest'] = idx_test
    else:
        # idx_np = train_test_split(graph_labels_dict, labelrate) # fixed split
        # trm, vam, tem = stratified_split_tkipf(adj, labels, train_ratio=0.6, val_ratio=0.2) # random split 60%/20%/20% without considering the classes.
        trm, vam, tem = stratified_split_tkipf(adj, labels, train_ratio=0.6, val_ratio=0.2) # random split 60%/20%/20% of each class
        idx_np['train'], idx_np['stopping'], idx_np['valtest'] = trm.numpy().tolist(), vam.numpy(), tem.numpy().tolist()

    return sG, idx_np

def load_OOD_dataset(name, data_dir='./', env_num=4, train_num=3, train_ratio=0.6, valid_ratio=0.2, combine=True):
    transform = T.NormalizeFeatures()
    torch_dataset = Planetoid(root=f'{data_dir}Planetoid',
                              name=name, transform=transform)
    preprocess_dir = os.path.join(data_dir, 'Planetoid', name)

    data = torch_dataset[0]
    x = data.x

    preprocess_dir = os.path.join(preprocess_dir, 'gen')
    if not os.path.exists(preprocess_dir):
        os.makedirs(preprocess_dir)
    spu_feat_num = 10
    class_num = data.y.max().item() + 1

    node_idx_list = [torch.arange(data.num_nodes) + i * data.num_nodes for i in range(env_num)]

    file_path = preprocess_dir + f'/{class_num}-{spu_feat_num}-{env_num}.pkl'
    print("using existing synthetic data...")
    with open(file_path, 'rb') as f:
        dataset = pkl.load(f)
    features = dataset['x']
    labels = dataset['y'].numpy()
    edge_index = dataset['edge_index']
    adj = sp.csr_matrix((np.ones(edge_index.shape[1]), (edge_index[0, :], edge_index[1, :])), shape=(features.shape[0], features.shape[0]))

    assert (train_num <= env_num - 1)

    ind_idx = torch.cat(node_idx_list[:train_num], dim=0)
    idx = torch.randperm(ind_idx.size(0))
    train_idx_ind = idx[:int(idx.size(0) * train_ratio)]
    valid_idx_ind = idx[int(idx.size(0) * train_ratio): int(idx.size(0) * (train_ratio + valid_ratio))]
    test_idx_ind = idx[int(idx.size(0) * (train_ratio + valid_ratio)):]
    idx_train = ind_idx[train_idx_ind].numpy()
    idx_stopping = ind_idx[valid_idx_ind].numpy()
    idx_valtest_in = ind_idx[test_idx_ind].numpy()

    idx_valtest_ood = [node_idx_list[-1]] if train_num == env_num - 1 else [
            torch.cat(node_idx_list[train_num:], dim=0)]  # Combine three ood environments

    G = nx.DiGraph(adj)

    for index in range(len(labels)):
        G.add_node(index, features=features[index], label=labels[index])
    if name == 'pubmed':
        sG = networkx_to_sparsegraph_floatfeature(G, dataset['x'].shape[1])
    else:
        sG = networkx_to_sparsegraph_intfeature(G, dataset['x'].shape[1])

    idx_np = {}
    idx_np['train'] = idx_train
    idx_np['stopping'] = idx_stopping
    idx_np['valtest'] = idx_valtest_in
    idx_np['valtest_ood'] = np.array(idx_valtest_ood[0])

    return sG, idx_np

def load_arxiv_dataset(train_num=3, train_ratio=0.5, valid_ratio=0.25, inductive=True):
    from ogb.nodeproppred import NodePropPredDataset

    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv', root=f'./ogb')

    node_years = ogb_dataset.graph['node_year']

    edge_index = torch.as_tensor(ogb_dataset.graph['edge_index'])
    node_feat = torch.as_tensor(ogb_dataset.graph['node_feat'])
    label = torch.as_tensor(ogb_dataset.labels)

    year_bound = [2005, 2010, 2012, 2014, 2016, 2018, 2021]
    env = torch.zeros(label.shape[0])
    for n in range(node_years.shape[0]):
        year = int(node_years[n])
        for i in range(len(year_bound)-1):
            if year >= year_bound[i+1]:
                continue
            else:
                env[n] = i
                break

    dataset = Data(x=node_feat, edge_index=edge_index, y=label)
    dataset.env = env
    dataset.env_num = len(year_bound)
    dataset.train_env_num = train_num

    ind_mask = (node_years < year_bound[train_num]).squeeze(1)
    idx = torch.arange(dataset.num_nodes)
    ind_idx = idx[ind_mask]
    idx_ = torch.randperm(ind_idx.size(0))
    train_idx_ind = idx_[:int(idx_.size(0) * train_ratio)]
    valid_idx_ind = idx_[int(idx_.size(0) * train_ratio): int(idx_.size(0) * (train_ratio + valid_ratio))]
    test_idx_ind = idx_[int(idx_.size(0) * (train_ratio + valid_ratio)):]
    dataset.train_idx = ind_idx[train_idx_ind]
    dataset.valid_idx = ind_idx[valid_idx_ind]
    dataset.test_in_idx = ind_idx[test_idx_ind]

    dataset.test_ood_idx = []

    for i in range(train_num, len(year_bound)-1):
        ood_mask_i = ((node_years >= year_bound[i]) * (node_years < year_bound[i+1])).squeeze(1)
        dataset.test_ood_idx.append(idx[ood_mask_i])

    features = dataset['x']
    labels = dataset['y'].numpy()
    labels = np.squeeze(labels, axis=-1)
    edge_index = dataset['edge_index']
    adj = sp.csr_matrix((np.ones(edge_index.shape[1]), (edge_index[0, :], edge_index[1, :])),
                        shape=(features.shape[0], features.shape[0]))

    G = nx.DiGraph(adj)

    for index in range(len(labels)):
        G.add_node(index, features=features[index], label=labels[index])
    sG = networkx_to_sparsegraph_floatfeature(G, dataset['x'].shape[1])

    idx_np = {}
    idx_np['train'] = dataset.train_idx.numpy()
    idx_np['stopping'] = dataset.valid_idx.numpy()
    idx_np['valtest'] = dataset.test_in_idx.numpy()
    idx_np['valtest_ood'] = torch.cat(dataset.test_ood_idx, dim=0).numpy()

    return sG, idx_np

def load_new_data_ms(labelrate):
    with np.load('./data/ms/ms_academic.npz', allow_pickle=True) as loader:
        loader = dict(loader)
        dataset = SparseGraph.from_flat_dict(loader)
        graph_labels_dict = {}
        for index in range(len(dataset.labels)):
            graph_labels_dict[index] = int(dataset.labels[index])
        idx_np = train_test_split(graph_labels_dict, labelrate)

        return dataset, idx_np

def load_new_data(dataset_name, train_label_rate, val_label_rate, test_label_rate, random_seed=123):
    dataset_folder = './data/new_data'

    dataset_path = os.path.join(dataset_folder, dataset_name)

    # 导入连接的点对信息
    edge_file = os.path.join(dataset_path, 'out1_graph_edges.txt')
    edges = np.loadtxt(edge_file, skiprows=1, dtype=int)
    num_nodes = np.max(edges) + 1

    # 创建稀疏矩阵
    adj = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                     shape=(num_nodes, num_nodes))

    # 导入节点特征和标签
    feature_file = os.path.join(dataset_path, 'out1_node_feature_label.txt')
    data = np.loadtxt(feature_file, skiprows=1, dtype=str)

    # 处理节点特征
    node_ids = data[:, 0].astype(int)
    if dataset_name == 'film':
        features = [list(map(lambda x: int(x) - 1, row.split(','))) for row in data[:, 1]]
    else:
        features = np.array([list(map(lambda x: int(x), row.split(','))) for row in data[:, 1]])
    labels = data[:, 2].astype(int)

    if dataset_name == 'film':
        # 创建空的特征矩阵
        feature_amount = 931  # 特征的总数量
        feature_matrix = np.zeros((len(data), feature_amount), dtype=int)

        # 填充特征矩阵中对应索引位置为1
        for i, feature_indices in enumerate(features):
            feature_matrix[i, feature_indices] = 1
        features = feature_matrix

    # 根据节点编号排序特征和标签
    sorted_indices = np.argsort(node_ids)
    node_ids = node_ids[sorted_indices]
    features = features[sorted_indices]
    labels = labels[sorted_indices]

    graph = SparseGraph(adj_matrix=adj, attr_matrix=features, labels=labels)
    graph = SparseGraph.to_unweighted(graph)
    graph = SparseGraph.to_undirected(graph)

    # 根据划分比例计算样本数量
    num_samples = len(labels)
    num_classes = len(np.unique(labels))
    num_train = int(num_samples * train_label_rate)
    percls_trn = int(round(num_samples * train_label_rate/num_classes))
    num_val = int(num_samples * val_label_rate)
    num_test = num_samples - num_train - num_val
    index = [i for i in range(0, num_samples)]

    train_idx = []
    rnd_state = np.random.RandomState(random_seed)
    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        if len(class_idx) < percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn, replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx = rnd_state.choice(rest_index, num_val, replace=False)
    test_idx = [i for i in rest_index if i not in val_idx]

    idx_np = {}
    idx_np['train'] = train_idx
    idx_np['stopping'] = val_idx
    idx_np['valtest'] = test_idx

    return graph, idx_np

def load_fixed_wikics():
    graph = WikiCS(root='./data/Wiki-CS', is_undirected=True)[0]
    graph.edge_index, _ = remove_self_loops(graph.edge_index)
    transform = T.Compose([T.AddSelfLoops(), T.ToUndirected(), ])
    graph = transform(graph)

    num_nodes = graph.x.size(0)

    idx_np = {}
    idx_np['train'], idx_np['stopping'], idx_np['valtest'] = fixed_split(graph, 100)

    adj = sp.csr_matrix((np.ones(graph.edge_index.shape[1]), (graph.edge_index[0, :], graph.edge_index[1, :])), shape=(num_nodes, num_nodes))

    features = np.array(graph.x)
    labels = np.array(graph.y)

    graph = SparseGraph(adj_matrix=adj, attr_matrix=features, labels=labels)
    graph = SparseGraph.to_unweighted(graph)
    graph = SparseGraph.to_undirected(graph)

    return graph, idx_np

def load_Amazon(dataset):
    graph = Amazon(root='./data', name=dataset.capitalize())[0]
    graph.edge_index, _ = remove_self_loops(graph.edge_index)
    transform = T.Compose([T.AddSelfLoops(), T.NormalizeFeatures()])
    graph = transform(graph)

    num_nodes = graph.x.size(0)

    idx_np = {}
    # trm, vam, tem = sparse_split(graph) # 2.5%/2.5%/95% whole dataset
    trm, vam, tem = stratified_sparse_split(graph) # 2.5%/2.5%/95% per class
    idx_np['train'], idx_np['stopping'], idx_np['valtest'] = trm.numpy().tolist(), vam.numpy(), tem.numpy().tolist()

    adj = sp.csr_matrix((np.ones(graph.edge_index.shape[1]), (graph.edge_index[0, :], graph.edge_index[1, :])), shape=(num_nodes, num_nodes))

    features = np.array(graph.x)
    labels = np.array(graph.y)

    graph = SparseGraph(adj_matrix=adj, attr_matrix=features, labels=labels)
    graph = SparseGraph.to_unweighted(graph)
    graph = SparseGraph.to_undirected(graph)

    return graph, idx_np

def fixed_split(graph, exp_num):
    num_splits = graph.train_mask.shape[1]
    split = exp_num % num_splits

    train_idx = torch.nonzero(graph.train_mask[:, split]).squeeze().numpy().tolist()
    test_idx = torch.nonzero(graph.test_mask).squeeze().numpy().tolist()
    val_idx = torch.nonzero(graph.val_mask[:, split]).squeeze().numpy()

    return train_idx, val_idx, test_idx

def sparse_split(graph, train_ratio=0.025, val_ratio=0.025):

    num_nodes = graph.x.size(0)
    num_labels = int(graph.y.max() + 1)

    nodes = torch.arange(num_nodes)
    nodes = nodes[torch.randperm(num_nodes)]

    num_train = int(num_nodes * train_ratio)
    num_val = int(num_nodes * val_ratio)

    train_idx = torch.LongTensor(nodes[0 : num_train])
    val_idx = torch.LongTensor(nodes[num_train : num_train+num_val])
    test_idx = torch.LongTensor(nodes[num_train+num_val : ])

    all_idx = torch.cat([train_idx, test_idx, val_idx])
    all_idx = torch.sort(all_idx)[0]
    assert torch.equal(all_idx, torch.arange(num_nodes))

    return train_idx, val_idx, test_idx

def stratified_sparse_split(graph, train_ratio=0.025, val_ratio=0.025, random_seed=123):
    num_nodes = graph.x.size(0)
    labels = graph.y
    num_labels = int(labels.max() + 1)

    # 创建空的训练、验证和测试索引列表
    train_idx = []
    val_idx = []
    test_idx = []

    rnd_state = np.random.RandomState(random_seed)

    # 对每个类别进行处理
    for c in range(num_labels):
        class_idx = (labels == c).nonzero(as_tuple=True)[0].numpy()
        class_size = len(class_idx)

        # 计算每个类别的训练和验证样本数量
        num_train_class = int(class_size * train_ratio)
        num_val_class = int(class_size * val_ratio)

        # 随机排列类别内的样本
        class_idx = class_idx[rnd_state.permutation(class_size)]

        # 添加训练、验证和测试样本的索引
        train_idx.extend(class_idx[:num_train_class])
        val_idx.extend(class_idx[num_train_class:num_train_class+num_val_class])
        test_idx.extend(class_idx[num_train_class+num_val_class:])

    # 将索引列表转换为LongTensor
    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, val_idx, test_idx


def sparse_split_tkipf(adj, labels, train_ratio=0.025, val_ratio=0.025):

    num_nodes = adj.shape[0]
    num_labels = int(labels.max() + 1)

    nodes = torch.arange(num_nodes)
    nodes = nodes[torch.randperm(num_nodes)]

    num_train = int(num_nodes * train_ratio)
    num_val = int(num_nodes * val_ratio)

    train_idx = torch.LongTensor(nodes[0 : num_train])
    val_idx = torch.LongTensor(nodes[num_train : num_train+num_val])
    test_idx = torch.LongTensor(nodes[num_train+num_val : ])

    all_idx = torch.cat([train_idx, test_idx, val_idx])
    all_idx = torch.sort(all_idx)[0]
    assert torch.equal(all_idx, torch.arange(num_nodes))

    return train_idx, val_idx, test_idx

def stratified_split_tkipf(adj, labels, train_ratio=0.025, val_ratio=0.025, random_seed=123):
    num_nodes = adj.shape[0]
    num_labels = int(labels.max() + 1)

    # 创建空的训练、验证和测试索引列表
    train_idx = []
    val_idx = []
    test_idx = []

    rnd_state = np.random.RandomState(random_seed)

    # 对每个类别进行处理
    for c in range(num_labels):
        class_idx = np.where(labels == c)[0]
        class_size = len(class_idx)

        # 计算每个类别的训练和验证样本数量
        num_train_class = int(class_size * train_ratio)
        num_val_class = int(class_size * val_ratio)

        # 随机排列类别内的样本
        class_idx = class_idx[rnd_state.permutation(class_size)]

        # 添加训练、验证和测试样本的索引
        train_idx.extend(class_idx[:num_train_class])
        val_idx.extend(class_idx[num_train_class:num_train_class+num_val_class])
        test_idx.extend(class_idx[num_train_class+num_val_class:])

    # 将索引列表转换为LongTensor
    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, val_idx, test_idx
