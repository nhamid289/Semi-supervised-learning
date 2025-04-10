import torch
import numpy as np

from semilearn.utils.data import BaseDataset

class SSLDataset():

    def __init__(self):
        self.lbl_dataset = None
        self.ulbl_dataset = None
        self.eval_dataset = None

    def get_lbl_dataset(self):
        return self.lbl_dataset

    def get_ulbl_dataset(self):
        return self.ulbl_dataset

    def get_eval_dataset(self):
        return self.eval_dataset


# class SSLDatasetInterface:

#     def __init__(self, lbl_dataset=None, ulbl_dataset=None,
#                  lbl_loader=None, ulbl_loader=None,
#                  eval_dataset=None, eval_loader=None):

#         self.lbl_dataset = lbl_dataset
#         self.ulbl_dataset = ulbl_dataset
#         self.lbl_loader = lbl_loader
#         self.ulbl_loader = ulbl_loader
#         self.eval_dataset = eval_dataset
#         self.get_eval_loader = eval_loader

#     def get_lbl_dataset(self):
#         return self.lbl_dataset

#     def get_ulbl_dataset(self):
#         return self.ulbl_dataset

#     def get_lbl_loader(self):
#         return self.lbl_loader

#     def get_ulbl_loader(self):
#         return self.ulbl_loader

#     def get_eval_dataset(self):
#         return self.eval_dataset

#     def get_eval_loader(self):
#         return self.eval_loader





