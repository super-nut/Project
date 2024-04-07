# -*- coding: utf-8 -*-
"""
@author: JHL
"""
import pandas as pd
from sklearn.model_selection import train_test_split  # 分割数据集
from sklearn.preprocessing import MinMaxScaler

# 读取数据
data = pd.read_excel('data/DQACC/raw/dqacc数据集-02-不平衡数据处理.xlsx', header=None)
# 列名变为第一行
data.columns = data.iloc[0, :]
# 去除第一行
data = data.drop(index=0)
# 变换数据类型
for i, label in enumerate(data.columns):
    if label != 'label':
        data[label] = data[label].astype('float32')
    else:
        data[label] = data[label].astype('int32')

# 简单划分训练测试集
features = data.iloc[:, :-1]  # 第二行到最后一行，第三列到倒数第二列为特征列
labels = data.iloc[:, -1]  # 最后一列为标签列
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features)
import torch

X_train, X_test, Y_train, Y_test = train_test_split(normalized_features, labels, test_size=0.2, random_state=42)
print(Y_train)
X = torch.tensor(X_train, dtype=torch.float32)
y = torch.tensor(Y_train.to_numpy(), dtype=torch.long).reshape(-1)

X_t = torch.tensor(X_test, dtype=torch.float32)
y_t = torch.tensor(Y_test.to_numpy(), dtype=torch.long).reshape(-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X, y = X.to(device), y.to(device)
X_t, y_t = X_t.to(device), y_t.to(device)

from torch.utils.data import Dataset, DataLoader


class Data(Dataset):
    def __init__(self, mode='train'):

        if mode == 'train':
            # 第一列包含图像文件的名称
            self.train_data = X
            self.labels = y

        elif mode == 'test':
            # 第一列包含图像文件的名称
            self.train_data = X_t
            self.labels = y_t

        self.real_len = len(self.labels)

        print('Finished reading the {} set of Leaves Dataset ({} samples found)'
              .format(mode, self.real_len))

    def __getitem__(self, index):

        feature = self.train_data[index]
        # 得到 feature label
        label = self.labels[index]
        return feature, label

    def __len__(self):
        return self.real_len

    def __str__(self):
        return super().__str__() + f":  {self.__len__()} samples found"


train_dataset = Data(mode='train')
valid_dataset = Data(mode='test')

train_loader = DataLoader(train_dataset, batch_size=128)
valid_dataset = DataLoader(valid_dataset, batch_size=128)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt


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


class SimpleTransformer(nn.Module):
    def __init__(self, num_features, dim_model=128, nhead=4, num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=256, dropout=0.1):
        super(SimpleTransformer, self).__init__()
        # 输入特征嵌入
        self.feature_embedding = nn.Linear(64, dim_model)

        # Transformer
        self.transformer = nn.Transformer(d_model=dim_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward, dropout=dropout)
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(dim_model, 64),
            nn.ReLU(),
            BayesianLinear(64, 16),
            nn.ReLU(),
            BayesianLinear(16, 2)
        )

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

        return output  # 确保输出形状是(batch_size,)


TRM = SimpleTransformer(num_features=64).to(device)


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        nn.init.kaiming_uniform_(m.weight)


TRM.apply(init_weights)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(TRM.parameters(), lr=0.0008, weight_decay=9e-4)
num_epochs = 1
min_acc = 0
min_acc_train = 0

for epoch in range(num_epochs):
    accu_train_sum = [0, 0]
    TRM.train()
    for i, data in enumerate(train_loader):
        features, label = data
        pre = TRM(features)
        # 测试精度
        with torch.no_grad():
            cmp = torch.argmax(pre, axis=1)
            cmp = (cmp == label)

            accu_train_sum[0] += cmp.sum()
            accu_train_sum[1] += len(label)
        # 梯度下降
        optimizer.zero_grad()
        cross_loss = criterion(pre, label)
        cross_loss.sum().backward()
        optimizer.step()

    train_acc = accu_train_sum[0] / accu_train_sum[1]
    if epoch % 100 == 1:
        print('step', epoch, 'train_accu: ', train_acc.item(), "details", accu_train_sum)

    accu_sum = [0, 0]
    TRM.eval()
    for i, data_test in enumerate(valid_dataset):
        features_test, label_test = data_test
        pre_valid = TRM(features_test)
        # 测试精度
        with torch.no_grad():
            cmp = torch.argmax(pre_valid, axis=1)
            cmp = (cmp == label_test)
            accu_sum[0] += cmp.sum()
            accu_sum[1] += len(label_test)

    accu = accu_sum[0] / accu_sum[1]
    if accu > min_acc:
        print('details', accu_sum, 'valid: ', accu, 'step', epoch, 'train_acc', train_acc)
        min_acc = accu
