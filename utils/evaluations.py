import numpy as np


def hit_k(k, scores, targets):
    # Hit Rate $ Recall & Precision
    if type(scores) is not np.ndarray:
        c_scores = scores.topk(k)[1].cpu().numpy()
    else:
        c_scores = scores
    if type(targets) is not np.ndarray:
        c_targets = targets.cpu().numpy()
    else:
        c_targets = targets
    hit = []
    for score, target in zip(c_scores, c_targets):
        hit.append(np.isin(target - 1, score))
    hit = np.mean(hit)
    return hit


def mrr_k(k, scores, targets):
    # Mean Reciprocal Rank,MRR
    if type(scores) is not np.ndarray:
        c_scores = scores.topk(k)[1].cpu().numpy()
    else:
        c_scores = scores
    if type(targets) is not np.ndarray:
        c_targets = targets.cpu().numpy()
    else:
        c_targets = targets
    mrr = []
    for score, target in zip(c_scores, c_targets):
        if len(np.where(score == target - 1)[0]) == 0:
            mrr.append(0)
        else:
            mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    mrr = np.mean(mrr)
    return mrr


def ndcg_k(k, scores, targets):
    # Normalized Discounted Cumulative Gain，NDCG
    if type(scores) is not np.ndarray:
        c_scores = scores.topk(k)[1].cpu().numpy()
    else:
        c_scores = scores
    if type(targets) is not np.ndarray:
        c_targets = targets.cpu().numpy()
    else:
        c_targets = targets
    ndcg = []
    for score, target in zip(c_scores, c_targets):
        if len(np.where(score == target - 1)[0]) == 0:
            ndcg.append(0)
        else:
            ndcg.append(1 / np.log2((np.where(score == target - 1)[0][0] + 1) + 1))
    ndcg = np.mean(ndcg)
    return ndcg


def coverage_k(k, scores, targets, true_set, pred_set):
    # Coverage
    # count, no need for minus 1
    if type(scores) is not np.ndarray:
        c_scores = scores.topk(k)[1].cpu().numpy()
    else:
        c_scores = scores
    new_c = []
    for row in c_scores:
        for item in row:
            new_c.append(item)
    if type(targets) is not np.ndarray:
        c_targets = targets.cpu().numpy()
    else:
        c_targets = targets
    true_set.update(list(c_targets.tolist()))
    pred_set.update(list(new_c))
    return true_set, pred_set


def popularity():
    pass