from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import MixedLinear, MixedDropout

def rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma=1.0):
    M, N = x.size(0), y.size(0)
    dist_matrix = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(M, N) + \
        torch.pow(y, 2).sum(dim=1, keepdim=True).expand(N, M).t()
    dist_matrix.addmm_(1, -2, x, y.t())
    return torch.exp(-gamma * dist_matrix)

class KernelFunctionLayer(nn.Module):
    def __init__(self, in_features, out_features, n_centers, kernel=rbf_kernel) -> None:
        super().__init__()
        self.z = nn.Parameter(torch.randn(n_centers, in_features) / (n_centers)**0.5)
        self.alpha = nn.Linear(in_features=n_centers, out_features=out_features, bias=False)
        self.kernel = kernel

    def forward(self, x: torch.Tensor):
        x = self.kernel(x, self.z)
        x = self.alpha(x)
        return x

class GNNs(nn.Module):
    def __init__(self, nfeatures: int, nclasses: int, hiddenunits: List[int], drop_prob: float,
                 propagation: nn.Module, bias: bool = False):
        super().__init__()

        fcs = [MixedLinear(nfeatures, hiddenunits[0], bias=bias)]
        for i in range(1, len(hiddenunits)):
            fcs.append(nn.Linear(hiddenunits[i - 1], hiddenunits[i], bias=bias))
        fcs.append(nn.Linear(hiddenunits[-1], nclasses, bias=bias))
        # fcs = [KernelFunctionLayer(nfeatures, hiddenunits[0], n_centers=32)]
        # for i in range(1, len(hiddenunits)):
        #     fcs.append(KernelFunctionLayer(hiddenunits[i-1], hiddenunits[i], n_centers=32))
        # fcs.append(KernelFunctionLayer(hiddenunits[-1], nclasses, n_centers=32))
        self.fcs = nn.ModuleList(fcs)

        self.reg_params = list(self.fcs[0].parameters())

        if drop_prob is 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)
        self.act_fn = nn.ReLU()

        self.propagation = propagation

    def _transform_features(self, attr_matrix: torch.sparse.FloatTensor):
        layer_inner = self.act_fn(self.fcs[0](self.dropout(attr_matrix)))
        for fc in self.fcs[1:-1]:
            layer_inner = self.act_fn(fc(layer_inner))
        res = self.fcs[-1](self.dropout(layer_inner))
        return res

    def forward(self, attr_matrix: torch.sparse.FloatTensor, idx: torch.LongTensor):
        local_logits = self._transform_features(attr_matrix)
        final_logits = self.propagation(local_logits, idx, attr_matrix)
        # print("finallogit={} {}".format(final_logits.shape , final_logits))
        return F.log_softmax(final_logits, dim=-1)
