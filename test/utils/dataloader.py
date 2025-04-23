import torch
from semilearn.utils.data import SSLCylicLoader

import torch

from semilearn.datasets import Cifar10

cifar10 = Cifar10()

cifar10.get_lbl_dataset()


trainloader = SSLCylicLoader(cifar10.get_lbl_dataset(), cifar10.get_ulbl_dataset(), 4, 8)

for (i, batch) in enumerate(trainloader):
    print(batch.y_lbl)
    # print(batch.X_lbl)

    if (i >= 100):
        break

