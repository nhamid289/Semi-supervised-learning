from torchvision.datasets import CIFAR10
from torchvision import transforms

from semilearn.datasets import SSLDataset
from semilearn.utils.data import BaseDataset, split_lb_ulb_balanced
from semilearn.datasets.augmentation import RandAugment

import numpy as np

class Cifar10(SSLDataset):

    def __init__(self, num_lbl=4, num_ulbl=None, seed=None,
                 crop_size=32, crop_ratio=1,
                 data_dir = "~/SSLDatasets/CIFAR10", download=True):

        self._define_transforms(crop_size, crop_ratio)

        self._get_dataset(num_lbl, num_ulbl, seed, data_dir, download)

    def _define_transforms(self, crop_size, crop_ratio):

        self.weak_transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomCrop(crop_size,
                                  padding=int(crop_size * (1 - crop_ratio)),
                                  padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.medium_transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            RandAugment(1, 5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.strong_transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            RandAugment(3, 5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _get_dataset(self, num_lbl, num_ulbl, seed, data_dir, download):

        cifar10_tr = CIFAR10(data_dir, train=True, download=download)
        X, y = cifar10_tr.data, cifar10_tr.targets
        X_lb, y_lb, X_ulb, y_ulb = split_lb_ulb_balanced(X, y, num_lbl=num_lbl,
                                                         num_ulbl=num_ulbl,
                                                         seed=seed)

        self.lbl_dataset = BaseDataset(X=X_lb, y=y_lb, num_classes=10,
                                       weak_transform=self.weak_transform,
                                       medium_transform=self.strong_transform,
                                       strong_transform=self.strong_transform)
        self.ulbl_dataset = BaseDataset(X=X_ulb, y=y_ulb, num_classes=10,
                                       weak_transform=self.weak_transform,
                                       medium_transform=self.medium_transform,
                                       strong_transform=self.strong_transform)

        cifar10_test = CIFAR10(data_dir, train=False, download=download)

        X, y = cifar10_test.data, cifar10_test.targets
        X, y = np.array(X), np.array(y)
        self.eval_dataset = BaseDataset(X=X, y=y, num_classes=10,
                                        weak_transform=self.eval_transform)

# class Cifar10(SSLDatasetInterface):

#     def __init__(self, config):

#         self.config = config
#         self._create_transforms()
#         self._get_dataset()

#         lbl_loader = DataLoader(self.lbl_dataset,
#                                      self.config.batch_size,
#                                      self.config.shuffle,
#                                      self.config.num_workers)

#         ulbl_loader = DataLoader(self.lbl_dataset,
#                                      self.config.batch_size * self.config.uratio,
#                                      self.config.shuffle,
#                                      self.config.num_workers)

#         eval_loader = DataLoader(self.eval_dataset,
#                                       self.config.eval_batch_size)

#         super().__init__(lbl_loader=lbl_loader, ulbl_loader=ulbl_loader,
#                          eval_loader=eval_loader)

#     def _create_transforms(self):
#         crop_size = self.config.crop_size
#         crop_ratio = self.config.crop_ratio

#         self.weak_transform = transforms.Compose([
#             transforms.Resize(crop_size),
#             transforms.RandomCrop(crop_size,
#                                   padding=int(crop_size * (1 - crop_ratio)),
#                                   padding_mode='reflect'),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])

#         self.medium_transform = transforms.Compose([
#             transforms.Resize(crop_size),
#             transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
#             transforms.RandomHorizontalFlip(),
#             RandAugment(1, 5),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])

#         self.strong_transform = transforms.Compose([
#             transforms.Resize(crop_size),
#             transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
#             transforms.RandomHorizontalFlip(),
#             RandAugment(3, 5),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])

#         self.eval_transform = transforms.Compose([
#             transforms.Resize(crop_size),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])

#     def _get_dataset(self):

#         cifar10_tr = CIFAR10(self.config.data_dir, train=True, download=True)
#         X, y = cifar10_tr.data, cifar10_tr.targets
#         X_lb, y_lb, X_ulb, y_ulb = split_lb_ulb_balanced(X, y,
#                                                          self.config.num_lbl,
#                                                          self.config.num_ulbl,
#                                                          self.config.seed)

#         self.lbl_dataset = BaseDataset(X_lb, y=y_lb, num_classes=10,
#                                        weak_transform=self.weak_transform,
#                                        medium_transform=self.strong_transform,
#                                        strong_transform=self.strong_transform)
#         self.ulbl_dataset = BaseDataset(X_ulb, y_ulb, num_classes=10,
#                                        weak_transform=self.weak_transform,
#                                        medium_transform=self.medium_transform,
#                                        strong_transform=self.strong_transform)

#         cifar10_test = CIFAR10(self.config.data_dir, train=False, download=True)
#         X, y = cifar10_tr.data, cifar10_tr.targets
#         self.eval_dataset = BaseDataset(X, y=y, num_classes=10,
#                                         weak_transform=self.eval_transform)

#     def _create_loaders(self):
#         lbl_loader = DataLoader(self.lbl_dataset,
#                                      self.config.batch_size,
#                                      self.config.shuffle,
#                                      self.config.num_workers)

#         ulbl_loader = DataLoader(self.lbl_dataset,
#                                      self.config.batch_size * self.config.uratio,
#                                      self.config.shuffle,
#                                      self.config.num_workers)

#         eval_loader = DataLoader(self.eval_dataset,
#                                       self.config.eval_batch_size)

#         return lbl_loader, ulbl_loader, eval_loader