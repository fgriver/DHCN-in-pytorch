import pickle
import argparse

import numpy as np
import numpy.random
import torch
from torch import nn
from dataloader import MyDataset, get_dataloader, split_validation, collate_fn
import random
import datetime
import os
from utils import compute_node_num
from model import DHCN
from tqdm import tqdm
from torch.utils.data import DataLoader
from itertools import islice


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# register device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--beta', type=float, default=0.01, help='the impact of cl loss')
parser.add_argument('--shuffle', default=True)
parser.add_argument('--hg_conv_step', type=int, default=3, help='hypergraph convolution steps')
parser.add_argument('--line_conv_steps', type=int, default=3, help='linear convolution steps')
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_rate', type=float, default=0.1, help='the rate of validation')
parser.add_argument('--log_dir', default='log/', help='log directory')
parser.add_argument('--patience', type=int, default=3, help='the patience of early stop')
opt = parser.parse_args()


def main():
    # print parse
    print(opt)

    # set seed
    set_seed(opt.seed)

    # handle the saved files
    log_dir = opt.log_dir
    timestape = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_path = os.path.join(log_dir, 'log_' + opt.dataset + timestape)
    # TODO: save the log files

    # load data
    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))
    all_data = pickle.load(open('datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    n_node = compute_node_num(all_data)

    # print('----')
    train_data = MyDataset(rawdata=train_data, n_node=n_node)
    test_data = MyDataset(rawdata=test_data, n_node=n_node)

    # print('test')
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=opt.shuffle, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, collate_fn=collate_fn)

    # init model
    model = DHCN(opt=opt, n_node=n_node, hg_adj_matrix=train_data.hg_adj_matrix)
    model.to(device)

    # init optimizer and scheduler
    adm = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=adm,
                                                   step_size=opt.lr_dc_step,
                                                   gamma=opt.lr_dc,
                                                   verbose=True)

    # load trainer
    train_test(model=model,
               trainDataloader=train_loader,
               testDataloader=test_loader,
               loss_func=nn.CrossEntropyLoss(),
               optimizer=adm,
               epochs=opt.epoch,
               scheduler=lr_scheduler)


def train_test(model, trainDataloader, testDataloader, loss_func, optimizer, epochs, scheduler, **kwargs):
    # define the best results of this model
    best_precision = 0.0
    best_mrr = 0.0
    best_epoch = [0, 0]
    patience = 0

    for epoch in tqdm(range(epochs), desc='Epoch'):
        # load model to 'TRAIN'
        model.train()
        epoch_total_loss = 0.0

        # the batches in one epoch
        with tqdm(total=len(trainDataloader), desc='training') as tbar:
            display_result = {'loss': 0.0, 'aver_loss': 0.0}
            for targets, session_lens, session_info_inputs, reversed_session_info, masks, line_adj_matrix, degree in trainDataloader:
                targets = torch.tensor(targets, dtype=torch.long, device=device)
                session_lens = torch.tensor(session_lens, dtype=torch.long, device=device)
                session_info_inputs = torch.tensor(session_info_inputs, dtype=torch.long, device=device)
                reversed_session_info = torch.tensor(reversed_session_info, dtype=torch.long, device=device)
                masks = torch.tensor(masks, dtype=torch.long, device=device)

                line_adj_matrix = torch.tensor(line_adj_matrix, dtype=torch.float32, device=device)
                degree = torch.tensor(degree, dtype=torch.float32, device=device)

                optimizer.zero_grad()

                hg_conved_session_emb, cl_loss, hg_conved_item_emb = model(session_info_inputs,
                                                                           reversed_session_info,
                                                                           session_lens,
                                                                           masks,
                                                                           line_adj_matrix,
                                                                           degree)
                scores = torch.mm(hg_conved_session_emb, torch.transpose(hg_conved_item_emb, 1, 0))
                rec_loss = loss_func(scores, targets)
                loss = rec_loss + cl_loss
                loss.backward()
                optimizer.step()

                display_result['loss'] = loss.item()
                # tbar.set_postfix(display_result)
                epoch_total_loss += loss.item()
                tbar.update(1)
            display_result['aver_loss'] = epoch_total_loss / len(trainDataloader)
            tbar.set_postfix(display_result)

        # load model to 'TEST'
        model.eval()
        mrr = []
        precision = []
        with torch.no_grad():
            display_result = {'P@20': 0.0, 'MRR@20': 0.0}
            with tqdm(total=len(testDataloader), desc='testing') as tbar:
                # iterate the batches
                for targets, session_lens, session_info_inputs, reversed_session_info, masks, line_adj_matrix, degree in testDataloader:
                    targets = np.asarray(targets)
                    session_lens = torch.tensor(session_lens, dtype=torch.long, device=device)
                    session_info_inputs = torch.tensor(session_info_inputs, dtype=torch.long, device=device)
                    reversed_session_info = torch.tensor(reversed_session_info, dtype=torch.long, device=device)
                    masks = torch.tensor(masks, dtype=torch.long, device=device)
                    line_adj_matrix = torch.tensor(line_adj_matrix, dtype=torch.float32, device=device)
                    degree = torch.tensor(degree, dtype=torch.float32, device=device)

                    hg_conved_session_emb, cl_loss, hg_conved_item_emb = model(session_info_inputs,
                                                                               reversed_session_info,
                                                                               session_lens,
                                                                               masks,
                                                                               line_adj_matrix,
                                                                               degree)
                    scores = torch.mm(hg_conved_session_emb, torch.transpose(hg_conved_item_emb, 1, 0))

                    # top_k_scores 没用
                    top_k_score, top_k_indices = torch.topk(scores, k=20)
                    top_k_indices = top_k_indices.cpu().detach().numpy()

                    for score_indices, target in zip(top_k_indices, targets):
                        precision.append(np.isin(target, score_indices))
                        if len(np.where(target == score_indices)[0]) == 0:
                            mrr.append(0)
                        else:
                            mrr.append(1. / (np.where(target == score_indices)[0][0] + 1))
                    tbar.update(1)

                # calculate
                # precision = torch.mean(torch.concat(precision).to(torch.float32))
                # mrr = torch.mean(torch.concat(mrr).to(torch.float32))
                precision = np.mean(precision) * 100
                mrr = np.mean(mrr) * 100
                display_result['P@20'] = precision.item()
                display_result['MRR@20'] = mrr.item()
                tbar.set_postfix(display_result)
                # early stop
                if precision.item() > best_precision:
                    best_precision = precision.item()
                    best_epoch[0] = epoch
                    print(f'best precision:{best_precision}')
                if mrr.item() > best_mrr:
                    best_mrr = mrr.item()
                    best_epoch[1] = epoch
                    print(f'best mrr:{best_mrr}')
                else:
                    patience += 1

                if patience >= opt.patience:
                    print(f'early stop epoch: {epoch}')
                    break
        # 是否需要加？
        # scheduler.step()


def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    main()
