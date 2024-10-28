import numpy as np
import torch
from torch import nn
from itertools import chain


class Loss(nn.Module):
    def __init__(self, scores, targets):
        self.scores = scores
        self.targets = targets

    def calculate_loss(self):
        return nn.CrossEntropyLoss(self.scores, self.targets)


def compute_node_num(all_data):
    # 返回一个chain对象
    n_node = set(node for session in all_data for node in session)

    return len(list(n_node))


class Metrics(nn.Module):
    def __init__(self, score, k):
        self.score = score
        self.k = k

    def calculate_mrr_k(self):
        print(self.score)
