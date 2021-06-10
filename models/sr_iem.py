'''
SR-IEM
'''
from abc import ABC

import torch
import math
import datetime
from torch import nn
from torch.nn import Module, Parameter
from tqdm import tqdm
from utils import *


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


class SrIEM(Module, ABC):
    def __init__(self, opt, n_node):
        super(SrIEM, self).__init__()
        self.hidden_size = opt.hidden_size
        self.batch_size = opt.batch_size
        self.n_node = n_node
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)

        # transform 2d(att+xt) -> d
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        # IEM
        self.attention_dim = 100
        self.Q = nn.Linear(self.hidden_size, self.attention_dim, False)
        self.K = nn.Linear(self.hidden_size, self.attention_dim, False)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def iem(self, inputs, masks):
        inputs_emb = self.embedding(inputs)  # [b,n,d]
        Q = torch.sigmoid(self.Q(inputs_emb))  # [b,n,l]
        K = torch.sigmoid(self.K(inputs_emb))  # [b.n,l]
        KT = K.permute(0, 2, 1)  # [b,l,n]
        affinity_matrix = torch.div(torch.bmm(Q, KT),
                                    torch.sqrt(torch.tensor(self.attention_dim).cuda().float()))  # [b,n,n]
        diag_element = torch.diagonal(affinity_matrix, dim1=-2, dim2=-1)  # [b,n]
        diag_matrix = torch.diag_embed(diag_element)  # [b,n,n]
        masked_affinity = affinity_matrix - diag_matrix  # [b,n,n]  after masked diagonal elements
        # sum by column to get importance
        col_mask = masks.view([self.batch_size, 1, -1])  # [b,1,n]
        col_masked_affinity = torch.mul(masked_affinity, col_mask)
        item_score = torch.sum(col_masked_affinity, 2)  # [b,n]
        item_score = torch.mul(item_score, masks)
        item_score = item_score.softmax(dim=1)
        item_score = torch.mul(item_score, masks)
        norm = item_score.sum(dim=1, keepdim=True)
        item_score = item_score / norm
        alpha = item_score.view([self.batch_size, -1, 1])
        final = torch.sum(alpha * inputs_emb, 1)  # [b,d]  mean sum
        return final, affinity_matrix

    def forward(self, inputs, masks):
        last_id = [inputs[i][j - 1] for i, j in zip(range(len(inputs)), torch.sum(masks, 1))]
        last_id = torch.Tensor(last_id).long().cuda()
        att_final, affinity_matrix = self.iem(inputs, masks)
        last_inputs_emb = self.embedding(last_id)
        final_rep = self.linear_transform(torch.cat([att_final, last_inputs_emb], 1))
        b = self.embedding.weight[1:]
        scores = torch.matmul(final_rep, b.transpose(1, 0))
        return scores, affinity_matrix


def train_and_test(model, train_data, test_data, opt):
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        m_print('epoch: {}, ==========================================='.format(epoch))
        flag = 0

        # training
        m_print('start training: {}'.format(datetime.datetime.now()))
        slices = train_data.generate_batch(opt.batch_size)
        train_loss = 0.0
        for i, j in zip(slices, tqdm(np.arange(len(slices)))):
            model.optimizer.zero_grad()
            inputs, masks, targets = train_data.get_slice(i)
            inputs = trans_to_cuda(torch.Tensor(inputs).long())
            masks = trans_to_cuda(torch.Tensor(masks).long())
            targets = trans_to_cuda(torch.Tensor(targets).long())
            scores = model(inputs, masks)
            loss = model.criterion(scores, targets - 1)
            # print(loss,loss_cl)
            loss.backward()
            model.optimizer.step()
            train_loss += loss
            if j % int(len(slices) / 5 + 1) == 0:
                print('Loss: %.4f' % (loss.item()))
        print('\tLoss:\t%.3f' % train_loss)
        print("learning rate is ", model.optimizer.param_groups[0]["lr"])

        # predicting
        if opt.is_train_eval:  # eval train data
            m_print('start predicting train data: {}'.format(datetime.datetime.now()))
            hit_dict, mrr_dict = {}, {}
            hit, mrr = [], []
            slices = train_data.generate_batch(opt.batch_size)
            with torch.no_grad():
                for i, j in zip(slices, tqdm(np.arange(len(slices)))):
                    inputs, masks, targets = train_data.get_slice(i)
                    length = np.sum(masks, axis=1)
                    inputs = trans_to_cuda(torch.Tensor(inputs).long())
                    masks = trans_to_cuda(torch.Tensor(masks).long())
                    targets = trans_to_cuda(torch.Tensor(targets).long())
                    scores = model(inputs, masks)
                    sub_scores = scores.topk(20)[1].cpu().numpy()
                    targets = targets.cpu().numpy()
                    for score, target, _len in zip(sub_scores, targets, length):
                        hit.append(np.isin(target - 1, score))
                        hit_dict.setdefault(_len, []).append(np.isin(target - 1, score))
                        if len(np.where(score == target - 1)[0]) == 0:
                            mrr.append(0)
                            mrr_dict.setdefault(_len, []).append(0)
                        else:
                            mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
                            mrr_dict.setdefault(_len, []).append(1 / (np.where(score == target - 1)[0][0] + 1))
            hit = np.mean(hit) * 100
            mrr = np.mean(mrr) * 100
            m_print("current train result:")
            m_print("\tRecall@20:\t{}\tMMR@20:\t{}\tEpoch:\t{}".format(hit, mrr, epoch))
        # eval test data
        m_print('start predicting: {}'.format(datetime.datetime.now()))
        hit, mrr = [], []
        hit_dict, mrr_dict = {}, {}
        slices = test_data.generate_batch(opt.batch_size)
        with torch.no_grad():
            for i, j in zip(slices, tqdm(np.arange(len(slices)))):
                inputs, masks, targets = test_data.get_slice(i)
                length = np.sum(masks, axis=1)
                inputs = trans_to_cuda(torch.Tensor(inputs).long())
                masks = trans_to_cuda(torch.Tensor(masks).long())
                targets = trans_to_cuda(torch.Tensor(targets).long())
                scores = model(inputs, masks)
                sub_scores = scores.topk(20)[1].cpu().numpy()
                targets = targets.cpu().numpy()
                for score, target, _len in zip(sub_scores, targets, length):
                    hit.append(np.isin(target - 1, score))
                    hit_dict.setdefault(_len, []).append(np.isin(target - 1, score))
                    if len(np.where(score == target - 1)[0]) == 0:
                        mrr.append(0)
                        mrr_dict.setdefault(_len, []).append(0)
                    else:
                        mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
                        mrr_dict.setdefault(_len, []).append(1 / (np.where(score == target - 1)[0][0] + 1))
        hit = np.mean(hit) * 100
        mrr = np.mean(mrr) * 100
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        # save checkpoint
        save_checkpoints(
            epoch=epoch,
            path=os.path.join('results', model.__class__.__name__),
            is_best=flag,
            model=model,
            optimizer=model.optimizer)
        bad_counter += 1 - flag

        m_print("current test result:")
        m_print("\tRecall@20:\t{}\tMMR@20:\t{}\tEpoch:\t{}".format(hit, mrr, epoch))
        if bad_counter >= opt.patience:
            break
        model.scheduler.step()

        # # print length info
        # for key_1, key_2 in zip(hit_dict.keys(), mrr_dict.keys()):
        #     print(key_1, np.mean(hit_dict[key_1]) * 100)
        #     print(key_2, np.mean(mrr_dict[key_2]) * 100)

    return best_result[0], best_result[1]
