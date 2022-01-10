import torch

def add_inverse_rels(edge_index, rel):
    edge_index_all = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    rel_all = torch.cat([rel, rel+rel.max()+1])
    return edge_index_all, rel_all


def get_train_batch(x1, x2, train_set, k=5):
    e1_neg1 = torch.cdist(x1[train_set[:, 0]], x1, p=1).topk(k+1, largest=False)[1].t()[1:]
    e1_neg2 = torch.cdist(x1[train_set[:, 0]], x2, p=1).topk(k+1, largest=False)[1].t()[1:]
    e2_neg1 = torch.cdist(x2[train_set[:, 1]], x2, p=1).topk(k+1, largest=False)[1].t()[1:]
    e2_neg2 = torch.cdist(x2[train_set[:, 1]], x1, p=1).topk(k+1, largest=False)[1].t()[1:]
    train_batch = torch.stack([e1_neg1, e1_neg2, e2_neg1, e2_neg2], dim=0)
    return train_batch


def get_hits(x1, x2, pair, dist='L1', Hn_nums=(1, 10)):
    pair_num = pair.size(0)
    S = torch.cdist(x1[pair[:, 0]], x2[pair[:, 1]], p=1)
    print('Left:\t',end='')
    for k in Hn_nums:
        pred_topk= S.topk(k, largest=False)[1]
        Hk = (pred_topk == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item()/pair_num
        print('Hits@%d: %.2f%%    ' % (k, Hk*100),end='')
    rank = torch.where(S.sort()[1] == torch.arange(pair_num, device=S.device).view(-1, 1))[1].float()
    MRR = (1/(rank+1)).mean().item()
    print('MRR: %.3f' % MRR)
    print('Right:\t',end='')
    for k in Hn_nums:
        pred_topk= S.t().topk(k, largest=False)[1]
        Hk = (pred_topk == torch.arange(pair_num, device=S.device).view(-1, 1)).sum().item()/pair_num
        print('Hits@%d: %.2f%%    ' % (k, Hk*100),end='')
    rank = torch.where(S.t().sort()[1] == torch.arange(pair_num, device=S.device).view(-1, 1))[1].float()
    MRR = (1/(rank+1)).mean().item()
    print('MRR: %.3f' % MRR)




