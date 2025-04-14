import torch

from semilearn.datasets import Cifar10

cifar10 = Cifar10()

cifar10.get_lbl_dataset()


# from semilearn.datasets import Cifar100

# cifar100 = Cifar100()

# cifar100.get_lbl_dataset()


from semilearn.utils.data import SSLDataLoader

trainloader = SSLDataLoader(cifar10.get_lbl_dataset(), cifar10.get_ulbl_dataset(), 4, 8)

print(len(cifar10.get_ulbl_dataset()))

for batch in trainloader:
    print(batch.y_lbl)
    print(batch.X_lbl)
