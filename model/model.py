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


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MLPModel(BaseModel):
    def __init__(self, in_features=64, out_features=2):
        super().__init__()
        self.linear1 = nn.Linear(in_features, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, 32)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(32, out_features)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x


class BayesianLinear(nn.Module):
    def __init__(self, in_features=64, out_features=2):
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


class T_BnnModel(BaseModel):
    def __init__(self,
                 dim_model=128,
                 nhead=4,
                 num_encoder_layers=3,
                 num_decoder_layers=3,
                 dim_feedforward=256,
                 dropout=0.1):
        super(T_BnnModel, self).__init__()

        # 输入特征嵌入
        self.feature_embedding = nn.Linear(64, dim_model)

        # Transformer
        self.transformer = nn.Transformer(d_model=dim_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(dim_model, 64),
            nn.ReLU(),
            BayesianLinear(64, 16),
            nn.ReLU(),
            BayesianLinear(16, 2)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, src):
        # 嵌入层处理输入
        src = self.feature_embedding(src)

        ## 增加一个虚拟的序列长度维度，并确保其形状为(seq_len, batch, dim_model)
        src = src.unsqueeze(0)  # 注意这里是unsqueeze(1)
        # 生成一个相同尺寸的“dummy”解码器输入，因为PyTorch的Transformer实现需要它
        # 实际上我们不会使用解码器输出
        dummy_tgt = torch.zeros_like(src)

        # Transformer处理
        transformer_output = self.transformer(src, dummy_tgt)

        # 将输出转换回(batch_size, dim_model)
        transformer_output = transformer_output.squeeze(0)

        # 输出层处理，得到最终的分类结果
        output = self.output_layer(transformer_output)

        return output
