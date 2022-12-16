from sinkhorn_loss_wasserstein import *
from CSLS import eval_alignment_by_sim_mat


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



def get_hits_np(Lvec, Rvec, test_pair, wrank=None, top_k=(1, 5, 10)):
    Lvec = Lvec.to('cpu').detach().numpy()
    Rvec = Rvec.to('cpu').detach().numpy()
    test_pair = test_pair.numpy().tolist()
    Lvec = np.array([Lvec[e1] for e1, e2 in test_pair])
    Rvec = np.array([Rvec[e2] for e1, e2 in test_pair])
    Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
    sim_o = -Lvec.dot(Rvec.T)
    sim = sim_o.argsort(-1)
    if wrank is not None:
        srank = np.zeros_like(sim)
        for i in range(srank.shape[0]):
            for j in range(srank.shape[1]):
                srank[i, sim[i, j]] = j
        rank = np.max(np.concatenate([np.expand_dims(srank, -1), np.expand_dims(wrank, -1)], -1), axis=-1)
        sim = rank.argsort(-1)
    top_lr = [0] * len(top_k)
    MRR_lr = 0
    for i in range(Lvec.shape[0]):
        rank = sim[i, :]
        rank_index = np.where(rank == i)[0][0]
        MRR_lr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    top_rl = [0] * len(top_k)
    MRR_rl = 0
    sim = sim_o.argsort(0)
    for i in range(Rvec.shape[0]):
        rank = sim[:, i]
        rank_index = np.where(rank == i)[0][0]
        MRR_rl += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('MRR: %.3f' % (MRR_lr / Lvec.shape[0]))
    print('For each right:')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
    print('MRR: %.3f' % (MRR_rl / Rvec.shape[0]))


def get_hits(x1, x2, pair, dist='L1', Hn_nums=(1, 10)):
    pair_num = pair.size(0)
    S = torch.cdist(x1[pair[:, 0]], x2[pair[:, 1]], p=1)
    print('Left:\t',end='')
    for k in Hn_nums:
        pred_topk = S.topk(k, largest=False)[1]
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
    return S


def CSLS_test(x1, x2, test_set, thread_number=16, csls=10, accurate=True):
    x1 = x1.detach().cpu().numpy()
    x2 = x2.detach().cpu().numpy()
    Lvec = np.array([x1[e1] for e1, e2 in test_set])
    Rvec = np.array([x2[e2] for e1, e2 in test_set])
    Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
    eval_alignment_by_sim_mat(Lvec, Rvec, [1, 10], thread_number, csls=csls, accurate=accurate)
    return None


def get_hits_stable(x1, x2, pair, S=None):
    import time
    pos = time.time()
    pair_num = pair.size(0)
    if S is None:
        S = -torch.cdist(x1[pair[:, 0]], x2[pair[:, 1]], p=1).cpu()
    index = (S.softmax(1)+S.softmax(0)).flatten().argsort(descending=True)
    index_e1 = index//pair_num
    index_e2 = index%pair_num
    aligned_e1 = torch.zeros(pair_num, dtype=torch.bool)
    aligned_e2 = torch.zeros(pair_num, dtype=torch.bool)
    true_aligned = 0
    for _ in range(pair_num*100):
        if aligned_e1[index_e1[_]] or aligned_e2[index_e2[_]]:
            continue
        if index_e1[_] == index_e2[_]:
            true_aligned += 1
        aligned_e1[index_e1[_]] = True
        aligned_e2[index_e2[_]] = True
    print('Both:\tHits@Stable: %.2f%%    ' % (true_aligned/pair_num*100))
    print("Stable time: {}".format(time.time() - pos))


def get_hits_sinkhorn(test_pair, S, top_k=(1, 10)):
    dev_s = test_pair[:, 0].to('cpu').numpy().tolist()
    dev_t = test_pair[:, 1].to('cpu').numpy().tolist()

    mu, nu = torch.ones(len(dev_s)), torch.ones(len(dev_t))
    sim = sinkhorn(mu, nu, S.to('cpu'), 0.05, stopThr=1e-3)
    if sim is None:
        S = torch.sqrt(S)
        sim = sinkhorn(mu, nu, S.to('cpu'), 0.05, stopThr=1e-3)
    sim = -sim

    credible_pairs_s2t, credible_pairs_t2s = set(), set()

    top_lr = [0] * len(top_k)
    MRR_lr = 0
    for i in range(len(dev_s)):
        rank = sim[i, :].argsort()
        credible_pairs_s2t.add((dev_s[i], dev_t[rank[0].numpy().tolist()]))
        rank_index = np.where(rank == i)[0][0]
        MRR_lr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1

    top_rl = [0] * len(top_k)
    MRR_rl = 0
    for i in range(len(dev_t)):
        rank = sim[:, i].argsort()
        credible_pairs_t2s.add((dev_s[rank[0].numpy().tolist()], dev_t[i]))
        rank_index = np.where(rank == i)[0][0]
        MRR_rl += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
    print('Left:', end=" ")
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100), end='\t')
    print('MRR: %.3f' % (MRR_lr / len(dev_s)))
    print('Right:', end=" ")
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100), end='\t')
    print('MRR: %.3f' % (MRR_rl / len(dev_t)))
    # intersection
    final_credible_pairs = credible_pairs_s2t.intersection(credible_pairs_t2s)
    return final_credible_pairs


def P_R_F1(credible_pairs, golden_pairs):
    hit = 0
    for p in credible_pairs:
        if p in golden_pairs:
            hit += 1
    P = hit/len(credible_pairs)
    R = hit/len(golden_pairs)
    F1 = 2*P*R/(P+R)
    print(f"Precision: {(P):.3f}, Recall: {(R):.3f}, F1: {(F1):.3f}")
    return P, R, F1

