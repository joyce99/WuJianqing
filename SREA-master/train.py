import argparse
import itertools

import apex

from model import SREA
from data import DBP15K
from loss import L1_Loss
from utils.utils import *
from sinkhorn_loss_wasserstein import *
import numpy as np
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--data", default="data/DBP15K")
    parser.add_argument("--lang", default="zh_en")
    # parser.add_argument("--lang", default="ja_en")
    # parser.add_argument("--lang", default="fr_en")

    # parser.add_argument("--data", default="data/SRPRS")
    # parser.add_argument("--lang", default="en_fr")
    # parser.add_argument("--lang", default="en_de")

    parser.add_argument("--rate", type=float, default=0.3)

    parser.add_argument("--r_hidden", type=int, default=200)

    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=3)

    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--neg_epoch", type=int, default=10)
    parser.add_argument("--test_epoch", type=int, default=20)
    parser.add_argument("--csls_test", action="store_true", default=True)
    parser.add_argument("--stable_test", action="store_true", default=True)
    parser.add_argument("--ot_test", action="store_true", default=True)
    args = parser.parse_args()
    return args


def init_data(args, device):
    data = DBP15K(args.data, args.lang, rate=args.rate)[0]

    data.x1 = F.normalize(data.x1, dim=1, p=2).to(device).requires_grad_()
    data.x2 = F.normalize(data.x2, dim=1, p=2).to(device).requires_grad_()
    data.edge_index_all1, data.rel_all1 = add_inverse_rels(data.edge_index1, data.rel1)
    data.edge_index_all2, data.rel_all2 = add_inverse_rels(data.edge_index2, data.rel2)

    lg_merge1 = np.load(args.data + '/' + args.lang + '/lg_merge1.npy')
    lg_merge2 = np.load(args.data + '/' + args.lang + '/lg_merge2.npy')
    data.lg_merge1 = torch.tensor(lg_merge1.nonzero())
    data.lg_merge2 = torch.tensor(lg_merge2.nonzero())
    data.lg_merge1_val = torch.tensor(lg_merge1[lg_merge1.nonzero()])
    data.lg_merge2_val = torch.tensor(lg_merge2[lg_merge2.nonzero()])

    lg_triangular1 = np.load(args.data + '/' + args.lang + '/lg_triangular1.npy')
    lg_triangular2 = np.load(args.data + '/' + args.lang + '/lg_triangular2.npy')
    data.lg_triangular1 = torch.tensor(lg_triangular1.nonzero())
    data.lg_triangular2 = torch.tensor(lg_triangular2.nonzero())
    data.lg_triangular1_val = torch.tensor(lg_triangular1[lg_triangular1.nonzero()])
    data.lg_triangular2_val = torch.tensor(lg_triangular2[lg_triangular2.nonzero()])
    return data


def get_emb(model, data):
    model.eval()
    with torch.no_grad():
        x1 = model(data.x1, data.edge_index1, data.rel1, data.edge_index_all1, data.rel_all1,
                       data.lg_merge1, data.lg_triangular1, data.lg_merge1_val, data.lg_triangular1_val)
        x2 = model(data.x2, data.edge_index2, data.rel2, data.edge_index_all2, data.rel_all2,
                       data.lg_merge2, data.lg_triangular2, data.lg_merge2_val, data.lg_triangular2_val)
    return x1, x2



def train(model, criterion, optimizer, data, train_batch):
    model.train()
    x1 = model(data.x1, data.edge_index1, data.rel1, data.edge_index_all1, data.rel_all1,
               data.lg_merge1, data.lg_triangular1, data.lg_merge1_val, data.lg_triangular1_val)
    x2 = model(data.x2, data.edge_index2, data.rel2, data.edge_index_all2, data.rel_all2,
               data.lg_merge2, data.lg_triangular2, data.lg_merge2_val, data.lg_triangular2_val)
    loss = criterion(x1, x2, data.train_set, train_batch)
    optimizer.zero_grad()
    with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    optimizer.step()
    return loss


def test(model, data, csls=False, stable=False, sinkhorn=False):
    x1, x2 = get_emb(model, data)
    print('-' * 16 + 'Test_set' + '-' * 16)
    S = get_hits(x1, x2, data.test_set)
    if csls is True:
        CSLS_test(x1, x2, data.test_set)
    if stable is True:
        get_hits_stable(x1, x2, data.test_set, -S)
    if sinkhorn is True:
        print('-' * 16 + 'Test_set_sinkhorn' + '-' * 16)
        final_credible_pairs = get_hits_sinkhorn(data.test_set, S)

        print('-' * 16 + 'Test_set_sinkhorn_P_R_F1' + '-' * 16)
        test_set = data.test_set.to('cpu').numpy().tolist()
        golden_pairs = set()
        for pair in test_set:
            golden_pairs.add((pair[0], pair[1]))
        P_R_F1(final_credible_pairs, golden_pairs)


def main(args):
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    data = init_data(args, device).to(device)

    model = SREA(data.x1.size(1), args.r_hidden, data.rel1.max() + 1, data.rel2.max() + 1).to(device)
    optimizer = torch.optim.Adam(itertools.chain(model.parameters()))
    model, optimizer = apex.amp.initialize(model, optimizer)
    criterion = L1_Loss(args.gamma)

    for epoch in range(args.epoch):
        if epoch % args.neg_epoch == 0:
            x1, x2 = get_emb(model, data)
            train_batch = get_train_batch(x1, x2, data.train_set, args.k)
        loss = train(model, criterion, optimizer, data, train_batch)
        print('Epoch:', epoch + 1, '/', args.epoch, '\tLoss: %.3f' % loss, '\r',end='')
        if (epoch + 1) % args.test_epoch == 0:
            print()
            test(model, data)
    print()
    test(model, data, csls=args.csls_test, stable=args.stable_test, sinkhorn=args.ot_test)


if __name__ == '__main__':
    args = parse_args()
    main(args)
