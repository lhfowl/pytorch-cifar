import torch
import random
import numpy as np


class sub_dataset(torch.utils.data.Dataset):
    def __init__(self, trainset, num, num_classes=10):
        self.num = num
        self.indcs = []
        self.trainset = trainset
        self.num_classes = num_classes
        self._get_indices()

    def _get_indices(self):
        temp = np.zeros(self.num_classes)
        train_indcs = list(range(len(self.trainset)))
        random.shuffle(train_indcs)
        for i in train_indcs:
            _, label = self.trainset[i]
            if temp[label] < self.num:
                self.indcs.append(i)
                temp[label] += 1
            if temp.sum() == self.num_classes * self.num:
                break

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return self.trainset[idx]
