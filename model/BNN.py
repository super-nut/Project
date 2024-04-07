# -*- coding: utf-8 -*-
"""
@author: JHL
"""

import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 定义权重的先验分布的参数
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))

        # 定义偏置的先验分布的参数
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_sigma, 0.1)
        nn.init.uniform_(self.bias_mu, -1 / sqrt(self.in_features), 1 / sqrt(self.in_features))
        nn.init.constant_(self.bias_sigma, 0.1)

    def forward(self, input):
        # 在前向传播时，为每个参数采样
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon

        return F.linear(input, weight, bias)


class BNNModel(BaseModel):
    def __init__(self, in_features=64, out_features=2):
        super().__init__()
        self.linear1 = BayesianLinear(in_features, 32)
        self.relu = nn.ReLU()
        self.linear2 = BayesianLinear(32, 32)
        self.relu2 = nn.ReLU()
        self.linear3 = BayesianLinear(32, out_features)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x



