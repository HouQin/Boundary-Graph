import math
import numpy as np
import scipy.sparse as sp
import torch
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import LocallyLinearEmbedding as LLE


class SparseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        input_coal = input.coalesce()
        drop_val = F.dropout(input_coal._values(), self.p, self.training)
        return torch.sparse.FloatTensor(input_coal._indices(), drop_val, input.shape)


class MixedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        if input.is_sparse:
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)


class MixedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Our fan_in is interpreted by PyTorch as fan_out (swapped dimensions)
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.bias is None:
            if input.is_sparse:
                res = torch.sparse.mm(input, self.weight)
            else:
                res = input.matmul(self.weight)
        else:
            if input.is_sparse:
                res = torch.sparse.addmm(self.bias.expand(input.shape[0], -1), input, self.weight)
            else:
                res = torch.addmm(self.bias, input, self.weight)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None)


def sparse_matrix_to_torch(X):
    coo = X.tocoo()
    indices = np.array([coo.row, coo.col])
    return torch.sparse.FloatTensor(
            torch.LongTensor(indices/1.0),
            torch.FloatTensor(coo.data),
            coo.shape)


def matrix_to_torch(X):
    if sp.issparse(X):
        return sparse_matrix_to_torch(X)
    else:
        return torch.FloatTensor(X)


def calc_uncertainty(values: np.ndarray, n_boot: int = 1000, ci: int = 95) -> dict:
    stats = {}
    stats['mean'] = values.mean()
    boots_series = sns.algorithms.bootstrap(values, func=np.mean, n_boot=n_boot)
    stats['CI'] = sns.utils.ci(boots_series, ci)
    stats['uncertainty'] = np.max(np.abs(stats['CI'] - stats['mean']))
    return stats

######## get multiview graph #########
def get_PCA_graph(X, device=None, r=200, k=3, t=1e3, adjmode='DAD'):
    S = X.T @ X

    # 计算矩阵的特征值和特征向量
    eigenvalues, eigenvectors = torch.linalg.eigh(S+1e-10*torch.eye(S.size()[0]).to(device))

    # 获取特征值的排序索引
    sorted_indices = torch.argsort(eigenvalues, descending=True)

    # 按照特征值递减的顺序排列特征向量
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    W = sorted_eigenvectors[:, 0:r]

    feature = X @ W

    # graph = construct_graph(feature, k, t)
    graph = get_epsilon_graph(feature)

    graph = (graph + graph.T) / 2  + 1e-10
    if adjmode == 'DAD':
        return torch.diag((torch.sum(graph, dim=0)) ** (-1/2)) @ graph @ torch.diag((torch.sum(graph, dim=0)) ** (-1/2))
    else:
        return torch.diag((torch.sum(graph, dim=0)) ** (-1)) @ graph

def get_LLE_graph(X, r=150, adjmode='DAD', device='cuda:0'):
    X_prime = X.clone().detach().cpu().numpy()
    embedding = LLE(n_components=r)
    X_lle = torch.from_numpy(embedding.fit(X_prime).embedding_).to(device)
    graph = get_epsilon_graph(X_lle)
    graph = (graph + graph.T) / 2
    if adjmode == 'DAD':
        return torch.diag((torch.sum(graph, dim=0) + 1e-5) ** (-1 / 2)) @ graph @ torch.diag((torch.sum(graph, dim=0) + 1e-5) ** (-1 / 2))
    else:
        return torch.diag((torch.sum(graph, dim=0) + 1e-5) ** (-1)) @ graph

def get_cosine_graph(X, adjmode='DAD'):
    inner_product = X @ X.T
    X_norm = torch.linalg.norm(X, dim=1).unsqueeze(1)
    norm_mat = X_norm @ X_norm.T + 1e-5
    norm_mat_einv = torch.reciprocal(norm_mat)
    cossim = inner_product * norm_mat_einv
    # cossim = cossim.detach().cpu().numpy()
    # return torch.where(cossim > 0, 1, 0)
    if adjmode == 'DAD':
        return torch.diag((torch.sum(cossim, dim=0) + 1e-5) ** (-1 / 2)) @ cossim @ torch.diag((torch.sum(cossim, dim=0) + 1e-5) ** (-1 / 2))
    else:
        return torch.diag((torch.sum(cossim, dim=0) + 1e-5) ** (-1)) @ cossim

def get_epsilon_graph(X, adjmode='DAD'):
    W = matrix_calculation(X, X)
    W = torch.where(W > torch.mean(W), 1, 0)
    graph = (W + W.T) / 2
    if adjmode == 'DAD':
        return torch.diag((torch.sum(graph, dim=0) + 1e-5) ** (-1 / 2)) @ graph @ torch.diag((torch.sum(graph, dim=0) + 1e-5) ** (-1 / 2))
    else:
        return torch.diag((torch.sum(graph, dim=0) + 1e-5) ** (-1)) @ graph

def create_symmetric_random_matrix(A, random_rate):
    size = A.shape[0]
    B = torch.zeros(size, size)
    num_ones = int(size * random_rate)
    for i in range(size):
        ones_idx = np.random.choice(size, num_ones, replace=False)
        B[i, ones_idx] = 1
    B = torch.triu(B) + torch.triu(B).T - torch.diag(torch.diag(B))
    D_12 = torch.diag(torch.clamp(torch.sum(B, dim=1), min=1e-5) ** (-1 / 2))
    B = D_12 @ B @ D_12
    return B

def matrix_calculation(A, B):
    # n1 x d | n2 x d
    # calculate dot products of A and B
    AB = A @ B.T

    # calculate squared norms of A and B
    Anorm = A.norm(dim=1, keepdim=True)
    Bnorm = B.norm(dim=1, keepdim=True)

    # calculate pairwise distances
    C = torch.abs((Anorm * Anorm) -2*AB + (Bnorm * Bnorm).T)
    C = C.sqrt()
    return C
