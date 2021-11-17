import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spmm
from torch_geometric.utils import *
from torch_scatter import scatter


  
class GCN(nn.Module):
    def __init__(self, hidden):
        super(GCN, self).__init__()
        self.w = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, edge_index):
        edge_index_j, edge_index_i = edge_index
        deg = degree(edge_index_i, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[edge_index_j]*deg_inv_sqrt[edge_index_i]
        x = F.relu(spmm(edge_index[[1, 0]], norm, x.size(0), x.size(0), x))
        x = self.w(x)
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

    def forward(self, x, edge_index):
        edge_index_j, edge_index_i = edge_index
        e_i = self.a_i(x).squeeze()[edge_index_i]
        e_j = self.a_j(x).squeeze()[edge_index_j]
        e = e_i + e_j
        alpha = softmax(F.leaky_relu(e).float(), edge_index_j)
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
        return F.relu(x)


class T_wise_graphattention(nn.Module):
    def __init__(self, e_hidden, r_hidden):
        super(T_wise_graphattention, self).__init__()
        self.e_hidden = e_hidden
        self.r_hidden = r_hidden
        self.ww = nn.Linear(e_hidden*2+r_hidden, 1, bias=False)


    def forward(self, x, edge_index_all, rel_all, rel_emb):
        outputs = []
        edge_index_i, edge_index_j = edge_index_all
        outputs.append(x)
        e_head = x[edge_index_i]
        e_tail = x[edge_index_j]
        e_rel = rel_emb[rel_all]
        att = self.ww(torch.cat([e_head, e_rel, e_tail], dim=1)).squeeze()
        att = softmax(att, edge_index_i)
        x_e = scatter(torch.cat([e_head, e_rel, e_tail], dim=1) * torch.unsqueeze(att, dim=-1), edge_index_i, dim=0, reduce='sum')
        e_features = F.relu(x_e)
        outputs.append(e_features)
        return torch.cat(outputs, dim=1)


class MREA(nn.Module):
    def __init__(self, e_hidden, r_hidden, rel1_size, rel2_size):
        super(MREA, self).__init__()
        self.gcn1 = GCN(e_hidden)
        self.highway1 = Highway(e_hidden)
        self.gcn2 = GCN(e_hidden)
        self.highway2 = Highway(e_hidden)
        self.l_gat = L_GAT(r_hidden)
        self.T_wise_graphattention = T_wise_graphattention(e_hidden, r_hidden)
        self.gat = GAT(e_hidden*3+r_hidden, r_hidden)

        self.rel_emb1 = nn.Parameter(nn.init.sparse_(torch.empty(rel1_size, r_hidden), sparsity=0.15))
        self.rel_emb2 = nn.Parameter(nn.init.sparse_(torch.empty(rel2_size, r_hidden), sparsity=0.15))

    def forward(self, x_e, edge_index, rel, edge_index_all, rel_all, line_graph_index_out, line_graph_index_in):
        x_e = self.highway1(x_e, self.gcn1(x_e, edge_index_all))
        x_e = self.highway2(x_e, self.gcn2(x_e, edge_index_all))
        if rel.max()+1 == self.rel_emb1.size(0):
            rel_emb_out = self.l_gat(self.rel_emb1, line_graph_index_out)
            rel_emb_in = self.l_gat(self.rel_emb1, line_graph_index_in)
        else:
            rel_emb_out = self.l_gat(self.rel_emb2, line_graph_index_out)
            rel_emb_in = self.l_gat(self.rel_emb2, line_graph_index_in)
        rel_emb = torch.cat([rel_emb_out, rel_emb_in], dim=0)

        x_rel = self.T_wise_graphattention(x_e, edge_index_all, rel_all, rel_emb)

        edge_index_all, rel_all = remove_self_loops(edge_index_all, rel_all)
        x_e = torch.cat([x_rel, self.gat(x_rel, rel_emb, edge_index_all, rel_all)], dim=1)
        return x_e
