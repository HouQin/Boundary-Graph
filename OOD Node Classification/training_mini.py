from typing import Type, Tuple
import time
import logging
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader
from earlystopping import EarlyStopping, stopping_args
from utils import matrix_to_torch
from sparsegraph import SparseGraph
from torch_geometric.utils import subgraph

from propagation import HornerSparseIteration

def gen_seeds(size: int = None) -> np.ndarray:
    max_uint32 = np.iinfo(np.uint32).max
    return np.random.randint(max_uint32 + 1, size=size, dtype=np.uint32)


def get_dataloaders(idx, labels_np, batch_size=None):
    labels = torch.LongTensor(labels_np)
    if batch_size is None:
        batch_size = max((val.numel() for val in idx.values()))
    datasets = {phase: TensorDataset(ind, labels[ind]) for phase, ind in idx.items()}
    dataloaders = {phase: DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
                   for phase, dataset in datasets.items()}
    return dataloaders


def normalize_attributes(attr_matrix):
    epsilon = 1e-12
    if isinstance(attr_matrix, sp.csr_matrix):
        attr_norms = spla.norm(attr_matrix, ord=1, axis=1)
        attr_invnorms = 1 / np.maximum(attr_norms, epsilon)
        attr_mat_norm = attr_matrix.multiply(attr_invnorms[:, np.newaxis])
    else:
        attr_norms = np.linalg.norm(attr_matrix, ord=1, axis=1)
        attr_invnorms = 1 / np.maximum(attr_norms, epsilon)
        attr_mat_norm = attr_matrix * attr_invnorms[:, np.newaxis]
    return attr_mat_norm


def train_model_mini(
        batchsize: int, idx_np, name: str, model_class: Type[nn.Module], graph: SparseGraph, model_args: dict,
        args: dict, learning_rate: float, reg_lambda: float,
        stopping_args: dict = stopping_args,
        test: bool = True, device: str = 'cuda',
        torch_seed: int = None, print_interval: int = 10) -> Tuple[nn.Module, dict]:

    labels_all = graph.labels
    labels_all = torch.LongTensor(labels_all).to(device)
    idx_all = {key: torch.LongTensor(val) for key, val in idx_np.items()}
    edge_index = graph.list_edge

    logging.log(21, f"{model_class.__name__}: {model_args}")
    if torch_seed is None:
        torch_seed = gen_seeds()
    torch.manual_seed(seed=torch_seed)
    logging.log(22, f"PyTorch seed: {torch_seed}")

    nfeatures = graph.attr_matrix.shape[1]
    nclasses = max(labels_all) + 1
    model = model_class(nfeatures, nclasses, **model_args).to(device)

    reg_lambda = torch.tensor(reg_lambda, device=device)

    n = graph.attr_matrix.shape[0]
    # ---------------- from difformer --------------------------------
    train_mask = torch.zeros(n, dtype=torch.bool).to(device)
    train_mask[idx_all['train']] = True
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_batch = n // batchsize + 1
    # ---------------- from difformer --------------------------------

    start_time = time.time()
    last_time = start_time

    early_stopping = EarlyStopping(model, **stopping_args)
    epoch_stats = {'stopping': {}}

    for epoch in range(early_stopping.max_epochs):
        model.to(device)
        model.train()  # Set model to training mode

        # ---------------- from difformer --------------------------------
        rand_idx = torch.randperm(n).to(device)

        for i in range(num_batch):
            idx_i = rand_idx[i * batchsize : (i + 1) * batchsize]
            train_mask_i = train_mask[idx_i]
            x_i = normalize_attributes(graph.attr_matrix)
            x_i = matrix_to_torch(x_i).to(device)
            x_i = x_i.to_dense()
            x_i = x_i[idx_i]
            edge_index_i, _ = subgraph(idx_i, edge_index, num_nodes=n, relabel_nodes=True)
            edge_index_i = edge_index_i.to(device)
            y_i = labels_all[idx_i].to(device)
            propagation = HornerSparseIteration(edge_index_i, num_heads=args.num_heads,
                                        nclasses=max(graph.labels) + 1, niter=args.niter, npow=args.npow, npow_attn=args.npow_attn,
                                        nalpha=1, num_feature=nfeatures, num_hidden=64,
                                        device=device, niter_attn=args.niter_attn)
            model.reset_propagation(propagation)
            # ---------------- from difformer --------------------------------

            optimizer.zero_grad()
            log_preds = model(attr_mat_norm, idx)
            preds = torch.argmax(log_preds, dim=1)

            # Calculate loss
            cross_entropy_mean = F.nll_loss(log_preds[train_mask_i, :], y_i[train_mask_i])
            l2_reg = sum((torch.sum(param ** 2) for param in model.reg_params))
            loss = cross_entropy_mean + reg_lambda / 2 * l2_reg

            loss.backward()
            optimizer.step()

        # eval
        if epoch % 10 == 0:
            train_acc, stopping_acc, valtest_acc, stopping_loss = evaluate_cpu(model, graph, idx_all)
            train_acc = train_acc.item()
            stopping_acc = stopping_acc.item()
            valtest_acc = valtest_acc.item()
            epoch_stats['stopping']['loss'] = stopping_loss
            epoch_stats['stopping']['acc'] = stopping_acc
            print(f'Train Acc: {train_acc*100:.2f}, Valid Acc: {stopping_acc*100:.2f}, Test Acc: {valtest_acc*100:.2f}')

        if len(early_stopping.stop_vars) > 0:
            stop_vars = [epoch_stats['stopping'][key]
                         for key in early_stopping.stop_vars]
            if early_stopping.check(stop_vars, epoch):
                break

    model.load_state_dict(early_stopping.best_state, False)
    train_acc, stopping_acc, valtest_acc, stopping_loss = evaluate_cpu(model, graph, idx_all)
    train_acc = train_acc.item()
    stopping_acc = stopping_acc.item()
    valtest_acc = valtest_acc.item()
    print(f'Best Train Acc: {train_acc * 100:.2f}, Valid Acc: {stopping_acc * 100:.2f}, Test Acc: {valtest_acc * 100:.2f}')

    result = {}
    result['predictions'] = []
    result['train'] = {'accuracy': train_acc}
    result['early_stopping'] = {'accuracy': stopping_acc}
    result['valtest'] = {'accuracy': valtest_acc}
    result['runtime'] = 0
    result['runtime_perepoch'] = 0

    return model, result

@torch.no_grad()
def evaluate_cpu(model, dataset, split_idx):
    model.eval()

    model.to(torch.device("cpu"))
    edge_index, x = dataset.list_edge, dataset.attr_matrix
    x = normalize_attributes(x)
    x = matrix_to_torch(x).to(torch.device("cpu"))
    edge_index = edge_index.to(torch.device("cpu"))
    out = model(x, edge_index)
    preds = torch.argmax(out, dim=1)

    # ------------------------------ acc --------------------------------
    train_acc = torch.sum(preds[split_idx['train']] == torch.LongTensor(dataset.labels[split_idx['train']])) / len(split_idx['train'])
    valid_acc = torch.sum(preds[split_idx['stopping']] == torch.LongTensor(dataset.labels[split_idx['stopping']])) / len(split_idx['stopping'])
    test_acc = torch.sum(preds[split_idx['valtest']] == torch.LongTensor(dataset.labels[split_idx['valtest']])) / len(split_idx['valtest'])

    valid_loss = F.nll_loss(out[split_idx['stopping'], :], torch.LongTensor(dataset.labels[split_idx['stopping']]))

    return train_acc, valid_acc, test_acc, valid_loss