'''
Realize an improved STAMP
use basic attention  wT*σ(w1x1+w2x2+b)
then use linear transform to ht+xt
'''
from abc import ABC

import torch
import math
import datetime
from torch import nn
from torch.nn import Module, Parameter
from tqdm import tqdm
from utils import *
import numpy as np


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


class ImprovedSTAMP(Module, ABC):
    def __init__(self, opt, n_node):
        super(ImprovedSTAMP, self).__init__()
        self.hidden_size = opt.hidden_size
        self.batch_size = opt.batch_size
        self.n_node = n_node
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)

        # basic attention
        self.att_linear_one = nn.Linear(self.hidden_size,self.hidden_size,True)
        self.att_linear_two = nn.Linear(self.hidden_size,self.hidden_size,True)
        self.att_linear_three = nn.Linear(self.hidden_size,1,False)

        # transform 2d(att+xt) -> d
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def basic_att(self, inputs, masks, anchor):
        anchor_emb = self.embedding(anchor)  # [b,d]
        inputs_emb = self.embedding(inputs)  # [b,n,d]
        q1 = self.att_linear_one(anchor_emb).view(self.batch_size, 1, self.hidden_size)
        q2 = self.att_linear_two(inputs_emb)
        alpha = self.att_linear_three(torch.sigmoid(q1+q2))  # [b,n,1]
        final = torch.mean(alpha * inputs_emb * masks.view(self.batch_size, -1, 1).float(), 1)  # [b,d]  mean sum
        return final

    def forward(self, inputs, masks, pos_masks, targets):
        last_id = [inputs[i][j - 1] for i, j in zip(range(len(inputs)), torch.sum(masks, 1))]
        last_id = torch.Tensor(last_id).long().cuda()
        att_final = self.basic_att(inputs,masks,last_id)
        last_inputs_emb = self.embedding(last_id)
        final_rep = self.linear_transform(torch.cat([att_final, last_inputs_emb], 1))
        b = self.embedding.weight[1:]
        scores = torch.matmul(final_rep, b.transpose(1, 0))
        # add cl
        cl_mask = self.cl_block(targets)
        cl_mask = trans_to_cuda(torch.Tensor(cl_mask).long())
        cl_score = self.contrastive(final_rep, targets, cl_mask)
        return scores, cl_score

    def contrastive(self, sr, targets, cl_mask):
        targets_emb = self.embedding(targets)  # [b,d]
        sr = sr.permute(1,0)
        cl_matrix = torch.mm(targets_emb,sr)  # [b,b]
        pos_masked_cl_matrix = torch.mul(cl_matrix, cl_mask)
        neg_masked_cl_matrix = torch.mul(cl_matrix, 1-cl_mask)
        pos_score = torch.mean(pos_masked_cl_matrix, dim=1)
        neg_score = torch.mean(neg_masked_cl_matrix, dim=1)
        cl_score = torch.stack((pos_score, neg_score), -1)  # [b,2]
        return cl_score

    def cl_block(self, targets):
        # targets [b]  return [b,b]
        targets = targets.cpu().numpy()
        cl_dict = {}
        for i, target in enumerate(targets):
            cl_dict.setdefault(target, []).append(i)
        mask = np.zeros((len(targets), len(targets)))
        for i, target in enumerate(targets):
            for j in cl_dict[target]:
                mask[i][j] = 1
        return mask

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
            inputs, masks, targets, pos_masks = train_data.get_slice(i)
            inputs = trans_to_cuda(torch.Tensor(inputs).long())
            masks = trans_to_cuda(torch.Tensor(masks).long())
            pos_masks = trans_to_cuda(torch.Tensor(pos_masks).long())
            targets = trans_to_cuda(torch.Tensor(targets).long())
            scores, cl_score = model(inputs, masks, pos_masks, targets)
            loss = model.criterion(scores, targets - 1)
            cl_target = torch.zeros(opt.batch_size,dtype=torch.long).cuda()
            loss_cl = model.criterion(cl_score, cl_target)
            total_loss = loss + opt.cl_lambda*loss_cl
            loss.backward()
            model.optimizer.step()
            train_loss += total_loss
            if j % int(len(slices) / 5 + 1) == 0:
                print('Loss: %.4f\t%.4f' % (loss.item(), opt.cl_lambda*loss_cl.item()))

        print('\tLoss:\t%.3f' % train_loss)
        print("learning rate is ", model.optimizer.param_groups[0]["lr"])

        # predicting
        if opt.is_train_eval:  # eval train data
            m_print('start predicting train data: {}'.format(datetime.datetime.now()))
            hit, mrr = [], []
            slices = train_data.generate_batch(opt.batch_size)
            with torch.no_grad():
                for i, j in zip(slices, tqdm(np.arange(len(slices)))):
                    inputs, masks, targets, pos_masks = train_data.get_slice(i)
                    inputs = trans_to_cuda(torch.Tensor(inputs).long())
                    masks = trans_to_cuda(torch.Tensor(masks).long())
                    pos_masks = trans_to_cuda(torch.Tensor(pos_masks).long())
                    targets = trans_to_cuda(torch.Tensor(targets).long())
                    scores, cl_score = model(inputs, masks, pos_masks, targets)
                    sub_scores = scores.topk(20)[1].cpu().numpy()
                    targets = targets.cpu().numpy()
                    for score, target in zip(sub_scores, targets):
                        hit.append(np.isin(target - 1, score))
                        if len(np.where(score == target - 1)[0]) == 0:
                            mrr.append(0)
                        else:
                            mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
            hit = np.mean(hit) * 100
            mrr = np.mean(mrr) * 100
            m_print("current train result:")
            m_print("\tRecall@20:\t{}\tMMR@20:\t{}\tEpoch:\t{}".format(hit, mrr, epoch))
        # eval test data
        m_print('start predicting: {}'.format(datetime.datetime.now()))
        hit, mrr = [], []
        slices = test_data.generate_batch(opt.batch_size)
        with torch.no_grad():
            for i, j in zip(slices, tqdm(np.arange(len(slices)))):
                inputs, masks, targets, pos_masks = test_data.get_slice(i)
                inputs = trans_to_cuda(torch.Tensor(inputs).long())
                masks = trans_to_cuda(torch.Tensor(masks).long())
                pos_masks = trans_to_cuda(torch.Tensor(pos_masks).long())
                targets = trans_to_cuda(torch.Tensor(targets).long())
                scores, cl_score = model(inputs, masks, pos_masks, targets)
                sub_scores = scores.topk(20)[1].cpu().numpy()
                targets = targets.cpu().numpy()
                for score, target in zip(sub_scores, targets):
                    hit.append(np.isin(target - 1, score))
                    if len(np.where(score == target - 1)[0]) == 0:
                        mrr.append(0)
                    else:
                        mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
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

    return best_result[0], best_result[1]