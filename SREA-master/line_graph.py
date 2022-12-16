from torch_geometric.io import read_txt_array
import torch
import numpy as np
import itertools
from tqdm import trange
from torch_geometric.utils import *



def process():
    dataset = 'DBP15K'  # DBP15K || SRPRS
    language = 'zh_en'

    save_merge_path1 = './data/' + dataset + '/' + language + '/lg_merge1.npy'
    save_merge_path2 = './data/' + dataset + '/' + language + '/lg_merge2.npy'
    save_triangular_path1 = './data/' + dataset + '/' + language + '/lg_triangular1.npy'
    save_triangular_path2 = './data/' + dataset + '/' + language + '/lg_triangular2.npy'

    ent_ids_1 = './data/' + dataset + '/' + language + '/ent_ids_1'
    ent_ids_2 = './data/' + dataset + '/' + language + '/ent_ids_2'
    triples_1 = './data/' + dataset + '/' + language + '/triples_1'
    triples_2 = './data/' + dataset + '/' + language + '/triples_2'

    print('process KG1')
    lg_merge = process_graph(triples_1, ent_ids_1)
    np.save(save_merge_path1, lg_merge)
    lg_triangular = get_line_graph_ring2(triples_1, ent_ids_1)
    np.save(save_triangular_path1, lg_triangular)

    print('process KG2')
    lg_merge = process_graph(triples_2, ent_ids_2)
    np.save(save_merge_path2, lg_merge)
    lg_triangular = get_line_graph_ring2(triples_2, ent_ids_2)
    np.save(save_triangular_path2, lg_triangular)


def process_graph(triple_path, ent_path):
    g = read_txt_array(triple_path, sep='\t', dtype=torch.long)
    subj, rel, obj = g.t()

    assoc = torch.full((rel.max().item() + 1,), -1, dtype=torch.long)
    assoc[rel.unique()] = torch.arange(rel.unique().size(0))
    rel = assoc[rel]

    idx = []
    with open(ent_path, 'r') as f:
        for line in f:
            info = line.strip().split('\t')
            idx.append(int(info[0]))
    idx = torch.tensor(idx)

    assoc = torch.full((idx.max().item() + 1,), -1, dtype=torch.long)
    assoc[idx] = torch.arange(idx.size(0))
    subj, obj = assoc[subj], assoc[obj]

    print('process lg_out')
    edge_index1 = torch.stack([subj, obj], dim=0)
    edge_index1, rel1 = sort_edge_index(edge_index1, rel)
    deg1 = degree(edge_index1[0]).numpy()
    edge_index1, rel1 = edge_index1.numpy().tolist(), rel1.numpy().tolist()

    line_graph_out = get_line_graph(edge_index1, rel1, deg1, self_loop=True)

    print('process lg_in')
    edge_index2 = torch.stack([obj, subj], dim=0)
    edge_index2, rel2 = sort_edge_index(edge_index2, rel)
    deg2 = degree(edge_index2[0]).numpy()
    edge_index2, rel2 = edge_index2.numpy().tolist(), rel2.numpy().tolist()
    line_graph_in = get_line_graph(edge_index2, rel2, deg2, self_loop=True)

    line_graph = line_graph_out + line_graph_in
    return line_graph


def get_line_graph(edge_index, rel, deg, self_loop=True):
    line_graph = np.zeros((max(rel)+1, max(rel)+1))
    length = 0
    for i in trange(max(edge_index[0])+1):
        head = edge_index[0][length]
        rel_index = []
        tail_index = []
        j = 0
        while j + length < len(edge_index[0]) and edge_index[0][j + length] == i:
            rel_index.append(rel[j + length])
            tail_index.append(edge_index[1][j + length])
            length += 1
        for idx1 in range(len(rel_index)):
            for idx2 in range(idx1 + 1, len(rel_index)):
                if self_loop is True or rel_index[idx1] != rel_index[idx2]:
                    weight = 1 / deg[head]
                    line_graph[rel_index[idx1]][rel_index[idx2]] += weight
                    line_graph[rel_index[idx2]][rel_index[idx1]] += weight
    return line_graph


def get_line_graph_ring2(triple_path, ent_path):
    g = read_txt_array(triple_path, sep='\t', dtype=torch.long)
    subj, rel, obj = g.t()

    assoc = torch.full((rel.max().item() + 1,), -1, dtype=torch.long)
    assoc[rel.unique()] = torch.arange(rel.unique().size(0))
    rel = assoc[rel]

    idx = []
    with open(ent_path, 'r') as f:
        for line in f:
            info = line.strip().split('\t')
            idx.append(int(info[0]))
    idx = torch.tensor(idx)

    assoc = torch.full((idx.max().item() + 1,), -1, dtype=torch.long)
    assoc[idx] = torch.arange(idx.size(0))
    subj, obj = assoc[subj], assoc[obj]
    edge_index = torch.stack([subj, obj], dim=0)
    # remove_self_loops
    edge_index, rel = remove_self_loops(edge_index, rel)
    edge_index, rel = sort_edge_index(edge_index, rel)

    deg_out = degree(edge_index[0]).numpy()
    deg_in = degree(edge_index[1]).numpy()

    edge_index_i, edge_index_j = edge_index.numpy()
    rel_list = rel.numpy().tolist()
    edge_index_i = edge_index_i.tolist()
    edge_index_j = edge_index_j.tolist()

    line_graph = np.zeros((max(rel)+1, max(rel)+1))
    # ring structure composed of two different entities
    length = 0
    neighs_hop1 = dict()
    for i in trange(max(edge_index_i) + 1):
        rel_index = []
        j = 0
        while j + length < len(edge_index_i) and edge_index_i[j + length] == i:
            rel_index.append([rel_list[j + length], edge_index_j[j + length]])
            length += 1
        neighs_hop1[i] = rel_index

    c = 0
    for key, val in neighs_hop1.items():
        for i in range(len(val)):
            for j in range(i + 1, len(val)):
                if val[i][1] == val[j][1] and val[i][1] != key:
                    weight = 1 / deg_out[key]
                    line_graph[val[i][0]][val[j][0]] += weight
                    line_graph[val[j][0]][val[i][0]] += weight
                    c += 1
    print('two_entities_loop numbers1:' + str(c))

    c = 0
    for key, val in neighs_hop1.items():
        for rel_out, tail_out in val:
            tail_val = neighs_hop1.get(tail_out)
            if tail_val is None:
                continue
            for rel_in, tail_in in tail_val:
                if tail_in == key and rel_in != rel_out:
                    weight = (1 / deg_out[key] + 1 / deg_in[key] + 1 / deg_out[tail_out] + 1 / deg_in[tail_out]) / 4
                    line_graph[rel_in][rel_out] += weight
                    line_graph[rel_out][rel_in] += weight
                    c += 1
    print('two_entities_loop numbers2:' + str(c))

    # triangular ring structure
    edge_index_all, rel = add_inverse_rels(edge_index, rel)
    edge_index_all, rel = sort_edge_index(edge_index_all, rel)

    deg = degree(edge_index_all[0]).numpy()

    edge_index_all_i, edge_index_all_j = edge_index_all.numpy()
    rel_list = rel.numpy().tolist()
    edge_index_all_i = edge_index_all_i.tolist()
    edge_index_all_j = edge_index_all_j.tolist()

    c = 0
    for i in trange(len(edge_index_all_i)):
        s1 = edge_index_all_i[i]
        r1 = rel_list[i]
        t1 = edge_index_all_j[i]
        idx1 = edge_index_all_i.index(t1)
        for j in range(idx1, len(edge_index_all_i)):
            if edge_index_all_i[j] != t1:
                break
            s2 = edge_index_all_i[j]
            r2 = rel_list[j]
            t2 = edge_index_all_j[j]
            idx2 = edge_index_all_i.index(t2)
            if t2 != s1:
                for k in range(idx2, len(edge_index_all_i)):
                    if edge_index_all_i[k] != t2:
                        break
                    s3 = edge_index_all_i[k]
                    r3 = rel_list[k]
                    t3 = edge_index_all_j[k]
                    if t3 == s1:  # 添加边的关联到线图
                        if True:
                            weight = (1 / deg[s1] + 1 / deg[s2] + 1 / deg[s3]) / 3
                            for index in itertools.combinations([r1, r2, r3], 2):
                                c += 1
                                line_graph[index[0]][index[1]] += weight
                                line_graph[index[1]][index[0]] += weight
    print('triangular numbers:' + str(c))
    return line_graph


def add_inverse_rels(edge_index, rel=None):
    edge_index_all = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    if rel is not None:
        rel_all = torch.cat([rel, rel])
        return edge_index_all, rel_all
    else:
        return edge_index_all


if __name__ == '__main__':
    process()
