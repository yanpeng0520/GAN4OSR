""" Evaluation metrics: accuracy, confidence, AUC, AUOC"""

import numpy as np
import torch
import scipy
import seaborn as sns
from sklearn.metrics import roc_auc_score, auc, roc_curve

def accuracy(predict, target):
    #predict can be logit or softmax score
    with torch.no_grad():
        known = target >= 0
        total = torch.sum(known, dtype=int)
        if total:
            correct = torch.sum(torch.max(predict[known], axis=1).indices == target[known], dtype=int)
        else:
            correct = 0
    return torch.tensor((correct, total))

def roc_auc(pred, y):
    scores = torch.ones_like(y, dtype=torch.float)
    target = torch.ones_like(y)
    # binary roc_auc with label 0 for unknowns and label 1 for knowns
    for i in range(len(y)):
        if y[i] == -1:
            scores[i] = torch.max(pred[i]).item()
            target[i] = 0

        elif y[i] == 10:
            pred_bg = pred[:, :-1]
            scores[i] = pred[i][y[i]].item()
            target[i] = 0
        else:
            scores[i] = pred[i][y[i]].item()

    scores = scores.detach().numpy()
    target = target.detach().numpy()
    fpr, tpr, thresholds = roc_curve(target, scores, pos_label=1)
    return auc(fpr, tpr)

def auoc(pred, y, BG=False):
    """
    auoc function computes the area of OSCR curve
    :param pred: the prediction of each class
    :param y: original class label
    :param BG: set True if the method uses background
    """

    if BG:
        # remove the labels for the unknown class in case of BG-softmax
        pred = pred[:,:-1]

    # vary thresholds
    ccr, fprt = [], []
    neg_class = 10 if BG else -1
    test = pred[y == neg_class]    # negetive probility of testing data

    positives = pred[y != neg_class]
    gt = y[y != neg_class]
    for tau in sorted(positives[range(len(gt)), gt]):
         # correct classification rate
         ccr.append(np.sum(np.logical_and(
           np.argmax(positives, axis=1) == gt,
           positives[range(len(gt)),gt] >= tau
         )) / len(positives))
         # false positive rate
         fprt.append(np.sum(np.max(test, axis=1) >= tau) / len(test))

    auoc = 0
    fprt.reverse()
    ccr.reverse()
    y=ccr[0]
    i = 1

    # compute the AUOC (area under OSCR curve)
    while i<len(fprt):
        if fprt[i] == fprt[i-1]:
            y=ccr[i]
        elif fprt[i] > fprt[i-1]:
            auoc += (fprt[i]-fprt[i-1])*y
            y=ccr[i]
        i +=1

    return ccr, fprt, auoc

def get_probs(pnts, pred_weights):
    e_ = np.exp(np.dot(pnts, pred_weights))
    e_ = e_ / np.sum(e_, axis=1)[:, None]
    res = np.max(e_, axis=1)
    return res

def entropy(data):
    entropy_data = scipy.stats.entropy(data, axis=1)
    sns.distplot(a=entropy_data, hist=False, color='green')
    sns.distplot(a=data, hist=False, color='green')  # discribution, can clear see located in some areas
    # plt.show()