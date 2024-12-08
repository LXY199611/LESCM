import os
import argparse
import os.path as p
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from CausalModel import Causal_Encoder_1, Causal_Encoder_2, Causal_Decoder_1, Causal_Decoder_2
from datasets import GetDataset
from sklearn.preprocessing import scale
import scipy.io as sio
import numpy as np
from measures import *
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

def log_standard_categorical(p, reduction="mean"):

    # Uniform prior over y
    prior = F.softmax(torch.ones_like(p), dim=1)
    prior.requires_grad = False
    cross_entropy = -torch.sum(p * torch.log(prior + 1e-8), dim=1)
    # print(cross_entropy)
  
    if reduction=="mean":
        cross_entropy = torch.mean(cross_entropy)
    else:
        cross_entropy = torch.sum(cross_entropy)
    
    return cross_entropy




def train(src_path1, src_path2, ds):

    data_f_d = sio.loadmat(src_path1)
    data_l = sio.loadmat(src_path2)

    num_sample = len(data_f_d['features'].T[0])
    candidate = int(num_sample / args.batch_size)
    features = data_f_d['features']
    f_dim = len(data_f_d['features'][0])
    dis_label_gt = data_f_d['labels']
    d_dim = len(data_f_d['labels'][0])
    log_label = data_l['logicalLabel']
    features_tmp = scale(features)
    x_data = torch.from_numpy(features_tmp).float().to(device)
    log_label = torch.from_numpy(log_label).float().to(device)

    LESCM_enc_1 = Causal_Encoder_1(f_dim, d_dim).to(device)
    LESCM_enc_2 = Causal_Encoder_2(f_dim + d_dim, args.h_dim).to(device)
    LESCM_dec_1 = Causal_Decoder_1(d_dim + f_dim, d_dim).to(device)
    LESCM_dec_2 = Causal_Decoder_2(args.h_dim + d_dim, f_dim).to(device)

    dataset = GetDataset(x_data, log_label)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    op = optim.Adam(list(LESCM_enc_1.parameters()) + list(LESCM_enc_2.parameters()) + list(LESCM_dec_1.parameters()) 
    + list(LESCM_dec_2.parameters()), lr=args.lr[0])

    lr_s = torch.optim.lr_scheduler.MultiStepLR(op, milestones=[50, 60, 70, 80, 90], gamma=0.9)

    best_cheb = 99
    best_all = []

    for epoch in range(args.epochs[0]):
        print(epoch)
        for batch_idx, (batch, log_l, _) in enumerate(train_loader):
            loss_all = 0
            d_recovered = LESCM_enc_1(batch)
            input_enc_2 = torch.cat((d_recovered, batch),1)
            mu, sigma_ = LESCM_enc_2(input_enc_2)
            z = mu + (torch.randn(sigma_.size()).to(device))*torch.exp(0.5*sigma_)
            input_dec_2 = torch.cat((d_recovered, z),1)
            x_hat = LESCM_dec_2(input_dec_2)

            input_dec_1 = torch.cat((d_recovered, x_hat),1)
            l_hat = LESCM_dec_1(input_dec_1)

            l1 = F.mse_loss(x_hat, batch, reduction="mean")
            l2 = -log_standard_categorical(d_recovered, reduction="mean")
            l3 = F.cross_entropy(log_l, l_hat, reduction="mean")
            l4 = -0.5*torch.sum(1 + sigma_ - mu.pow(2) - sigma_.exp())
            l5 = F.mse_loss(d_recovered, log_l, reduction="mean")
            lpp = 0
            th = 0.0
            for i in range(d_recovered.shape[0]):
                neg_index = (log_l[i] == 0).to(device)
                pos_index = (log_l[i] == 1).to(device)
                if torch.sum(pos_index) == 0 or torch.sum(neg_index) == 0: continue
                lpp += torch.maximum(d_recovered[i][neg_index][torch.argmax(d_recovered[i][neg_index])] - d_recovered[i][pos_index][torch.argmin(d_recovered[i][pos_index])] + th, torch.tensor([0]).to(device)).to(device)

            lpp = lpp.mean()

            loss_all = loss_all + l1 + l2 + l3 + l4 + args.para_hyper[0]*l5 + args.para_hyper[1]*lpp
            op.zero_grad()
            loss_all.backward()
            op.step()
            lr_s.step()
            if batch_idx % args.log_interval == 0:
                with torch.no_grad():
                        if epoch == args.epochs[0] - args.early:
                            LESCM_enc_1.eval()
                            d = LESCM_enc_1(x_data)
                            preds = d.data.cpu().numpy()
                            preds = softmax(preds)
                            dists = []
                            dist1 = chebyshev(dis_label_gt, preds)
                            dist2 = clark(dis_label_gt, preds)
                            dist3 = canberra(dis_label_gt, preds)
                            dist4 = kl_dist(dis_label_gt, preds)
                            dist5 = cosine(dis_label_gt, preds)
                            dist6 = intersection(dis_label_gt, preds)
                            dists.append(dist1)
                            dists.append(dist2)
                            dists.append(dist3)
                            dists.append(dist4)
                            dists.append(dist5)
                            dists.append(dist6)
                            if dists[0] <  best_cheb:
                                best_cheb = dists[0]
                                best_all = dists
                                state = LESCM_enc_1.state_dict()
                                torch.save(state,'./models/' + ds +'.pth')
    # evaluate 
    checkpoint = torch.load('./models/' + ds + '.pth')
    LESCM_enc_1.load_state_dict(checkpoint)
    LESCM_enc_1.eval()
    d = LESCM_enc_1(x_data)
    distri_pre_tmp = []
    distri_pre_tmp.extend(d.data.cpu().numpy())
    preds = softmax(distri_pre_tmp)
    dists = []
    dist1 = chebyshev(dis_label_gt, preds)
    dist2 = clark(dis_label_gt, preds)
    dist3 = canberra(dis_label_gt, preds)
    dist4 = kl_dist(dis_label_gt, preds)
    dist5 = cosine(dis_label_gt, preds)
    dist6 = intersection(dis_label_gt, preds)

    dists.append(dist1)
    dists.append(dist2)
    dists.append(dist3)
    dists.append(dist4)
    dists.append(dist5)
    dists.append(dist6)
    return dists

def softmax(d, t=1):
    for i in range(len(d)):
        d[i] = d[i]*t
        d[i] = np.exp(d[i])/sum(np.exp(d[i]))
    return d

current_ds = 'unkown'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--h_dim', type=int, default=256, metavar='N',
                        help='input batch size for training [default: 2000]')
    parser.add_argument('--para_hyper', type=int, default=[1e3, 1e-1], metavar='N',
                        help='input batch size for training [default: 2000]')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training [default: 2000]')
    parser.add_argument('--epochs', type=int, default=[150], metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--lr', type=float, default=[0.005], metavar='LR',
                        help='learning rate [default: 1e-3]')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training [default: False]')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='choose CUDA device [default: cuda:1]')
    parser.add_argument('--seed', '-seed', type=int, default=3,
                        help='random seed (default: 0)')
    parser.add_argument('--early', '-early', type=int, default=2,
                        help='early stop')

    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    device = torch.device(args.device if args.cuda else 'cpu')

    SRC_PATH = p.join(p.dirname(__file__), 'dataset')
    DST_PATH = p.join(p.dirname(__file__), 'results_with_causal')

    setup_seed(args.seed)

    dataset = 'Movie'

    if not p.isdir(DST_PATH):
        os.mkdir(DST_PATH)

    src_file1 = p.join(SRC_PATH, dataset + '.mat')
    src_file2 = p.join(SRC_PATH, dataset + '_binary.mat')
    dst_file = p.join(DST_PATH, dataset + '_with_causal_results_log.txt')

    print("Label enhancement process with causal of " + dataset)

    # setup_seed(args.seed)
    dists = train(src_file1, src_file2, dataset)

    args_lists = vars(parser.parse_args())
    print('Param:' + str(args.para_hyper[0]) + "\t" +str(args.para_hyper[1]))
    print("args: " + str(args_lists))
    print("chebyshev, clark, canberra, kl_dist, cosine, intersection: ")
    print(np.round(dists, 4))
    with open(dst_file, 'a', encoding='utf-8') as file:
        file.write("Label enhancement process with causal of " + dataset + "\n")
        file.write("args: " + str(args_lists) + "\n")
        file.write("chebyshev, clark, canberra, kl_dist, cosine, intersection: " + "\n")
        file.write(' '.join(map(str, np.round(dists, 4).ravel().tolist())) + "\n")
        file.write("----------------------------------------------------------------------------\n")


