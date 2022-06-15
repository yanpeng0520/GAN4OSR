
import torch.nn.functional as F
import torch

def remove_unknown(X,y):
    correct = y != -1
    return X[correct], y[correct]

def filter_correct(X, y, pred):
    correct = torch.argmax(pred, dim=1) == y
    return X[correct], y[correct], y

def filter_threshold(X, y, pred, thresh=0.9):
    correct = torch.diagonal(torch.index_select(F.softmax(pred, dim=1), dim=1, index=y) > thresh)
    return X[correct], y[correct], y

def filter_gan_threshold(X, y, y1, pred, thresh=0.9):
    correct = torch.diagonal(torch.index_select(F.softmax(pred, dim=1), dim=1, index=y) < thresh) & torch.diagonal(torch.index_select(F.softmax(pred, dim=1), dim=1, index=y1) < thresh)
    return X[correct]