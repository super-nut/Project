# -*- coding: utf-8 -*-
"""
@author: JHL
"""
import os
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class dqacc_dataset(Dataset):
    """
    加载dqacc数据集，并进行预处理
    """
    random_seed = 0
    train_ratio = 0.8

    def __init__(
            self,
            xlsx_file: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        self.fpath = xlsx_file
        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform

        if not os.path.isfile(xlsx_file):
            raise RuntimeError(f"Data file not found. Please check the file path <{xlsx_file}>")
        if not xlsx_file.endswith(".xlsx"):
            raise RuntimeError(f"Wrong file type. Only .xlsx file is allowed")
        self.data, self.targets = self._load_data()

    def _load_data(self):
        """
        这里默认讲数据进行随机的8：2划分成训练集和测试集
        """
        raw_pd = pd.read_excel(self.fpath, header=0)
        raw = torch.from_numpy(raw_pd.to_numpy())
        data = raw[:, :-1].float()
        targets = raw[:, -1:].long()

        "原始数据归一化"
        data = (data - data.min(dim=0)[0]) / data.max(dim=0)[0]

        # "训练集和测试集划分"
        # shuffled_indices = np.random.permutation(len(raw))
        # train_set_size = int(len(raw) * self.train_ratio)
        # train_indices = shuffled_indices[:train_set_size]
        # test_indices = shuffled_indices[train_set_size:]
        #
        # if self.train:
        #     data = data[train_indices]
        #     targets = targets[train_indices]
        # else:
        #     data = data[test_indices]
        #     targets = targets[test_indices]

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data, target = self.data[index], int(self.targets[index])

        return data, target

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self is not None:
            body.append(f"Data file location: {self.fpath}")

        lines = [head] + ["    " + line for line in body]

        return "\n".join(lines)


class DqaccDataLoader(BaseDataLoader):
    """
    使用BaseDataLoader封装并转载 dqacc 数据
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.fpath = os.path.join(data_dir, "DQACC/raw/dqacc数据集-02-不平衡数据处理.xlsx")

        self.dataset = dqacc_dataset(self.fpath, train=training)
        print(self.dataset)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

if __name__ == '__main__':
    pass
