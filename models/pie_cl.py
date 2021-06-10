'''
PIE-CL
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


class PIECL(Module, ABC):
    def __init__(self, opt, n_node):
        super(PIECL, self).__init__()
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

        # position
        self.pos_embedding = nn.Embedding(opt.max_len+1, self.hidden_size)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def iem_pos(self, inputs, masks, pos_masks):
        inputs_emb = self.embedding(inputs)  # [b,n,d]
        pos_emb = self.pos_embedding(pos_masks) # [b,n,d]
        pos_emb = torch.mul(pos_emb, masks.view(self.batch_size, -1, 1))
        inputs_emb = inputs_emb + pos_emb
        Q = torch.tanh(self.Q(inputs_emb))  # [b,n,l]
        K = torch.tanh(self.K(inputs_emb))  # [b.n,l]
        KT = K.permute(0,2,1)  # [b,l,n]
        affinity_matrix = torch.div(torch.bmm(Q,KT),torch.sqrt(torch.tensor(self.attention_dim).cuda().float()))  # [b,n,n]
        diag_element = torch.diagonal(affinity_matrix,dim1=-2,dim2=-1)  # [b,n]
        diag_matrix = torch.diag_embed(diag_element)  # [b,n,n]
        masked_affinity = affinity_matrix - diag_matrix  # [b,n,n]  after masked diagonal elements
        # sum by column to get importance
        col_mask = masks.view([self.batch_size,1,-1]) # [b,1,n]
        col_masked_affinity = torch.mul(masked_affinity, col_mask)
        item_score = torch.sum(col_masked_affinity,2)  # [b,n]
        item_score = torch.mul(item_score, masks)
        item_score = item_score.softmax(dim=1)
        item_score = torch.mul(item_score, masks)
        norm = item_score.sum(dim=1, keepdim=True)
        item_score = item_score/norm
        alpha = item_score.view([self.batch_size,-1,1])
        final = torch.sum(alpha * inputs_emb, 1)  # [b,d]  mean sum
        return final, affinity_matrix

    def forward(self, inputs, masks, pos_masks, targets):
        last_id = [inputs[i][j - 1] for i, j in zip(range(len(inputs)), torch.sum(masks, 1))]
        last_id = torch.Tensor(last_id).long().cuda()
        att_final, affinity_matrix = self.iem_pos(inputs, masks, pos_masks)
        last_inputs_emb = self.embedding(last_id)
        final_rep = self.linear_transform(torch.cat([att_final, last_inputs_emb], 1))
        b = self.embedding.weight[1:]
        scores = torch.matmul(final_rep, b.transpose(1, 0))
        cl_mask = self.cl_block(targets)
        cl_mask = trans_to_cuda(torch.Tensor(cl_mask).long())
        cl_score = self.contrastive(final_rep, targets, cl_mask)
        return scores, cl_score, affinity_matrix

    def contrastive(self, sr, targets, cl_mask):
        targets_emb = self.embedding(targets)  # [b,d]
        sr = sr.permute(1,0)
        cl_matrix = torch.mm(targets_emb,sr)  # [b,b]
        pos_masked_cl_matrix = torch.mul(cl_matrix, cl_mask)
        neg_masked_cl_matrix = torch.mul(cl_matrix, 1-cl_mask)
        pos_score = torch.mean(pos_masked_cl_matrix, dim=1)
        neg_score = torch.mean(neg_masked_cl_matrix, dim=1)
        cl_score = torch.stack((pos_score,neg_score),-1)  # [b,2]
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
    cl_lambda = 0
    for epoch in range(opt.epoch):
        cl_lambda += 1
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
            # total_loss.backward()
            model.optimizer.step()
            train_loss += total_loss
            if j % int(len(slices) / 5 + 1) == 0:
                print('Loss: %.4f\t%.4f' % (loss.item(), opt.cl_lambda*loss_cl.item()))
                # print('Loss: %.4f\t%.4f' % (loss.item(), cl_lambda*loss_cl.item()))
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