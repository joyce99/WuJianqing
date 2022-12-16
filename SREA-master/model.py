import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spmm
from torch_geometric.utils import *
import math


class GCN(nn.Module):
    def __init__(self, hidden):
        super(GCN, self).__init__()
        self.w = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, edge_index):
        edge_index_j, edge_index_i = edge_index
        deg = degree(edge_index_i, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[edge_index_j]*deg_inv_sqrt[edge_index_i]
        x = self.w(x)
        x = F.relu(spmm(edge_index[[1, 0]], norm, x.size(0), x.size(0), x))
        return x

    
class Highway(nn.Module):
    def __init__(self, x_hidden):
        super(Highway, self).__init__()
        self.lin = nn.Linear(x_hidden, x_hidden)

    def forward(self, x1, x2):
        gate = torch.sigmoid(self.lin(x1))
        x = torch.mul(gate, x2)+torch.mul(1-gate, x1)
        return x


class L_GAT(nn.Module):
    def __init__(self, hidden):
        super(L_GAT, self).__init__()
        self.a_i = nn.Linear(hidden, 1, bias=False)
        self.a_j = nn.Linear(hidden, 1, bias=False)

    def forward(self, x, edge_index, val=None):
        edge_index_j, edge_index_i = edge_index
        val_i = softmax(val, edge_index_i)
        val_j = softmax(val, edge_index_j)
        e = val_i + val_j
        alpha = softmax(e.float(), edge_index_j)
        x = F.relu(spmm(edge_index[[1, 0]], alpha, x.size(0), x.size(0), x))
        return x


class GAT(nn.Module):
    def __init__(self, hidden, r_hidden):
        super(GAT, self).__init__()
        self.a_i = nn.Linear(hidden, 1, bias=False)
        self.a_j = nn.Linear(hidden, 1, bias=False)
        self.a_r = nn.Linear(r_hidden, 1, bias=False)
        
    def forward(self, x, r, edge_index, rel):
        edge_index_j, edge_index_i = edge_index
        e_i = self.a_i(x).squeeze()[edge_index_i]
        e_j = self.a_j(x).squeeze()[edge_index_j]
        r = self.a_r(r).squeeze()[rel]
        e = e_i+e_j+r
        alpha = softmax(F.leaky_relu(e).float(), edge_index_i)
        x = F.relu(spmm(edge_index[[1, 0]], alpha, x.size(0), x.size(0), x))
        return x


class GAT_R_to_E(nn.Module):
    def __init__(self, e_hidden, r_hidden):
        super(GAT_R_to_E, self).__init__()
        self.a_h = nn.Linear(e_hidden, 1, bias=False)
        self.a_t = nn.Linear(e_hidden, 1, bias=False)
        self.a_r = nn.Linear(r_hidden, 1, bias=False)

    def forward(self, x_e, x_r, edge_index, rel):
        edge_index_h, edge_index_t = edge_index
        e_h = self.a_h(x_e).squeeze()[edge_index_h]
        e_t = self.a_t(x_e).squeeze()[edge_index_t]
        e_r = self.a_r(x_r).squeeze()[rel]
        alpha = softmax(F.leaky_relu(e_h + e_r).float(), edge_index_h)
        x_e_h = spmm(torch.cat([edge_index_h.view(1, -1), rel.view(1, -1)], dim=0), alpha, x_e.size(0), x_r.size(0),
                     x_r)
        alpha = softmax(F.leaky_relu(e_t + e_r).float(), edge_index_t)
        x_e_t = spmm(torch.cat([edge_index_t.view(1, -1), rel.view(1, -1)], dim=0), alpha, x_e.size(0), x_r.size(0),
                     x_r)
        x = torch.cat([x_e_h, x_e_t], dim=1)
        return x


class SREA(nn.Module):
    def __init__(self, e_hidden, r_hidden, rel1_size, rel2_size):
        super(SREA, self).__init__()
        self.gcn1 = GCN(300)
        self.highway1 = Highway(300)
        self.gcn2 = GCN(300)
        self.highway2 = Highway(300)

        self.l_gat_adj = L_GAT(r_hidden)
        self.l_gat_tri = L_GAT(r_hidden)

        self.highway_rel = Highway(r_hidden)
        self.gat_r_to_e = GAT_R_to_E(e_hidden*2, r_hidden)
        self.gat = GAT(300, r_hidden*2)

        self.rel_emb_out1 = nn.Parameter(nn.init.sparse_(torch.empty(rel1_size, r_hidden), sparsity=0.15))
        self.rel_emb_out2 = nn.Parameter(nn.init.sparse_(torch.empty(rel2_size, r_hidden), sparsity=0.15))
        self.rel_emb_tri1 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(rel1_size, r_hidden), gain=math.sqrt(1.5)))
        self.rel_emb_tri2 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(rel2_size, r_hidden), gain=math.sqrt(1.5)))


    def forward(self, x_e, edge_index, rel, edge_index_all, rel_all, lg_merge, lg_triangular, merge_val, tri_val):
        x_e = self.highway1(x_e, self.gcn1(x_e, edge_index_all))
        x_e = self.highway2(x_e, self.gcn2(x_e, edge_index_all))

        if rel.max()+1 == self.rel_emb_out1.size(0):
            rel_emb_merge = self.l_gat_adj(self.rel_emb_out1, lg_merge, merge_val)
            rel_emb_tri = self.l_gat_tri(self.rel_emb_tri1, lg_triangular, tri_val)

        else:
            rel_emb_merge = self.l_gat_adj(self.rel_emb_out2, lg_merge, merge_val)
            rel_emb_tri = self.l_gat_tri(self.rel_emb_tri2, lg_triangular, tri_val)

        x_r = self.highway_rel(rel_emb_merge, rel_emb_tri)
        rel_emb = torch.cat([torch.cat([rel_emb_merge, rel_emb_tri], dim=1), torch.cat([rel_emb_merge, rel_emb_tri], dim=1)], dim=0)
        x_e = torch.cat([x_e, self.gat(x_e, rel_emb, edge_index_all, rel_all)], dim=1)
        x_e = torch.cat([x_e, self.gat_r_to_e(x_e, x_r, edge_index, rel)], dim=1)
        return x_e
