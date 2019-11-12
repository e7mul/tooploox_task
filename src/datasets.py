import torch
import numpy as np
from torchvision.datasets import CIFAR10


class TripletDataset(CIFAR10):
    def __init__(self, root, device, train=True,  transform=None,
                 target_transform=None, download=True):
        super(TripletDataset, self).__init__(root, train=True,
                                             transform=None,
                                             target_transform=None,
                                             download=True)
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.numpy()
        anch = np.array(self.data[idx])
        anch_c = self.targets[idx]
        pos_idx = np.random.choice(np.arange(len(self))[np.array(self.targets)
                                                        == anch_c])
        neg_idx = np.random.choice(np.arange(len(self))[np.array(self.targets)
                                                        != anch_c])
        pos, pos_c = self.data[pos_idx], self.targets[pos_idx]
        neg, neg_c = self.data[neg_idx], self.targets[neg_idx]
        if self.transform:
            pos = self.transform(pos)
            neg = self.transform(neg)
            anch = self.transform(anch)
        return anch, anch_c, pos, pos_c, neg, neg_c


if __name__ == '__main__':
    pass
