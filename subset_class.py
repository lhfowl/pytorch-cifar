import torch
import random
import numpy as np


class sub_dataset(torch.utils.data.Dataset):
    def __init__(self, trainset, num, num_classes=10):
        self.num = num
        self.indcs = []
        self._get_indices()
        self.trainset = trainset
        self.num_classes = num_classes

    def _get_indices(self):
        temp = np.zeros(self.num_classes)
        for i in random.randrange(len(self.trainset)):
            _, label = trainset[i]
            if temp[label] < self.num:
                self.indcs.append(i)
            if temp.sum() == self.num_classes * self.num:
                break

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return self.trainset[idx]
