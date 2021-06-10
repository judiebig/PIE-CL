'''
Realize Repeat-Net without GRU
a efficient yet effective version same in RE-GNN
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


class ImprovedRepeatNet(Module, ABC):
    def __init__(self, opt, n_node):
        super(ImprovedRepeatNet, self).__init__()
        self.hidden_size = opt.hidden_size
        self.batch_size = opt.batch_size
        self.n_node = n_node
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)

        # session representation
        self.Wre1 = nn.Linear(self.hidden_size, 1, bias=False)
        self.Wre2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.Wre3 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        # self.Wr4 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.Wre = nn.Linear(self.hidden_size * 2, 2, bias=False)

        # repeat mode
        self.Wr1 = nn.Linear(self.hidden_size, 1, bias=False)
        self.Wr2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.Wr3 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.Wr4 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

        # explore mode
        self.We1 = nn.Linear(self.hidden_size, 1, bias=False)
        self.We2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.We3 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.We4 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

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

    def re_ratio(self, last_id, inputs, masks):
        # return repeat ratio and explore ratio
        last_inputs_emb = self.embedding(last_id)  # [b,d]
        inputs_emb = self.embedding(inputs)  # [b,n,d]
        alpha_temp_1 = self.Wre2(inputs_emb)  # [b,n,d]
        alpha_temp_2 = self.Wre3(last_inputs_emb).view(self.batch_size, 1, self.hidden_size)  # [b,1,d]
        alpha_temp_3 = torch.sigmoid(alpha_temp_1+alpha_temp_2)  # [b,n,d]
        alpha_temp_4 = torch.mul(self.Wre1(alpha_temp_3).view(self.batch_size, -1), masks)  # [b,n]
        alpha = alpha_temp_4.softmax(dim=1)
        alpha = torch.mul(alpha, masks)
        norm = alpha.sum(dim=1,keepdim=True)
        alpha = (alpha/norm).view(self.batch_size,-1,1)  # [b,n,1]
        sr_global = torch.sum(alpha*inputs_emb, dim=1)
        sr = torch.cat([sr_global, last_inputs_emb], 1)  # [b,2d]
        re_weight = self.Wre(sr).softmax(dim=1)
        return re_weight

    def repeat_mode(self, last_id, inputs, masks):
        # return repeat score
        last_inputs_emb = self.embedding(last_id)  # [b,d]
        inputs_emb = self.embedding(inputs)  # [b,n,d]
        alpha_temp_1 = self.Wr2(inputs_emb)  # [b,n,d]
        alpha_temp_2 = self.Wr3(last_inputs_emb).view(self.batch_size, 1, self.hidden_size)  # [b,1,d]
        alpha_temp_3 = torch.sigmoid(alpha_temp_1+alpha_temp_2)  # [b,n,d]
        alpha_temp_4 = torch.mul(self.Wr1(alpha_temp_3).view(self.batch_size, -1), masks)  # [b,n]
        alpha = alpha_temp_4.softmax(dim=1)
        alpha = torch.mul(alpha, masks)
        norm = alpha.sum(dim=1,keepdim=True)
        alpha = (alpha/norm).view(self.batch_size,-1,1)  # [b,n,1]
        sr_global = torch.sum(alpha*inputs_emb, dim=1)
        sr = torch.cat([sr_global, last_inputs_emb], 1)  # [b,2d]
        sr = self.Wr4(sr)  # [b,d]
        return sr

    def explore_mode(self, last_id, inputs, masks):
        # return explore score
        last_inputs_emb = self.embedding(last_id)  # [b,d]
        inputs_emb = self.embedding(inputs)  # [b,n,d]
        alpha_temp_1 = self.We2(inputs_emb)  # [b,n,d]
        alpha_temp_2 = self.We3(last_inputs_emb).view(self.batch_size, 1, self.hidden_size)  # [b,1,d]
        alpha_temp_3 = torch.sigmoid(alpha_temp_1+alpha_temp_2)  # [b,n,d]
        alpha_temp_4 = torch.mul(self.We1(alpha_temp_3).view(self.batch_size, -1), masks)  # [b,n]
        alpha = alpha_temp_4.softmax(dim=1)
        alpha = torch.mul(alpha, masks)
        norm = alpha.sum(dim=1,keepdim=True)
        alpha = (alpha/norm).view(self.batch_size,-1,1)  # [b,n,1]
        sr_global = torch.sum(alpha*inputs_emb, dim=1)
        sr = torch.cat([sr_global, last_inputs_emb], 1)  # [b,2d]
        sr = self.We4(sr)  # [b,d]
        return sr

    def forward(self, inputs, masks, re_masks):
        last_id = [inputs[i][j - 1] for i, j in zip(range(len(inputs)), torch.sum(masks, 1))]
        last_id = torch.Tensor(last_id).long().cuda()
        b = self.embedding.weight[1:]

        # ratio
        re_weight = self.re_ratio(last_id, inputs, masks)  # [b,2]

        # repeat
        repeat_sr = self.repeat_mode(last_id, inputs, masks)
        repeat_scores = torch.matmul(repeat_sr, b.transpose(1, 0))  # [b,N-1]
        repeat_scores = torch.mul(repeat_scores, re_masks)
        # explore
        explore_sr = self.explore_mode(last_id, inputs, masks)
        explore_scores = torch.matmul(explore_sr, b.transpose(1, 0))  # [b,N-1]
        explore_scores = torch.mul(explore_scores, 1-re_masks)

        repeat_weight, explore_weight = torch.chunk(re_weight, 2, dim=1)  # [b,1]

        scores = repeat_weight * repeat_scores + explore_weight * explore_scores

        return scores


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
            re_masks = np.zeros([opt.batch_size, model.n_node-1])
            for i, input in enumerate(inputs):
                for item in input:
                    if item != 0:
                        re_masks[i][item-1] = 1
            inputs = trans_to_cuda(torch.Tensor(inputs).long())
            masks = trans_to_cuda(torch.Tensor(masks).long())
            targets = trans_to_cuda(torch.Tensor(targets).long())
            re_masks = trans_to_cuda(torch.Tensor(re_masks).long())
            scores = model(inputs, masks, re_masks)
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
            hit, mrr = [], []
            slices = train_data.generate_batch(opt.batch_size)
            with torch.no_grad():
                for i, j in zip(slices, tqdm(np.arange(len(slices)))):
                    inputs, masks, targets = train_data.get_slice(i)
                    re_masks = np.zeros([opt.batch_size, model.n_node - 1])
                    for i, input in enumerate(inputs):
                        for item in input:
                            if item != 0:
                                re_masks[i][item - 1] = 1
                    inputs = trans_to_cuda(torch.Tensor(inputs).long())
                    masks = trans_to_cuda(torch.Tensor(masks).long())
                    targets = trans_to_cuda(torch.Tensor(targets).long())
                    re_masks = trans_to_cuda(torch.Tensor(re_masks).long())
                    scores = model(inputs, masks, re_masks)
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
                inputs, masks, targets = test_data.get_slice(i)
                re_masks = np.zeros([opt.batch_size, model.n_node - 1])
                for i, input in enumerate(inputs):
                    for item in input:
                        if item != 0:
                            re_masks[i][item - 1] = 1
                inputs = trans_to_cuda(torch.Tensor(inputs).long())
                masks = trans_to_cuda(torch.Tensor(masks).long())
                targets = trans_to_cuda(torch.Tensor(targets).long())
                re_masks = trans_to_cuda(torch.Tensor(re_masks).long())
                scores = model(inputs, masks, re_masks)
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

    return best_result[0], best_result[1]