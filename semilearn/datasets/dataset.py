import torch
import numpy as np

from torch.utils.data import Dataset


def split_lb_ulb_balanced(X, y, num_lbl, num_ulbl = None,
                 lbl_idx=None, ulbl_idx=None, lbl_in_ulbl=True,
                 return_idx = False, seed=None):
    """
    A function to split features and labels into separate labelled and
    unlabelled sets.

    Args:
        X: the features
        y: the labels
        num_classes: The number of target classes
        num_lbl: The number of samples per class to be labelled
        num_ulbl: The number of samples per class to be unlabelled.
            If left unspecified, all remaining unlabelled data is taken
        lbl_idx: The specific indices to include in labelled data.
        ulbl_indx: The specific indices to include in unlabelled data.

    Returns
        If return_idx is True:
            Returns a tuple of lists containing the labelled and unlabelled
            indices
        Else:
            Returns a 4-tuple containing the labelled features and labels,
            and unlabelled features and labels
    """

    lbl_idx = [] if lbl_idx is None else lbl_idx
    ulbl_idx = [] if ulbl_idx is None else ulbl_idx
    if seed is not None:
        np.random.seed(seed)

    for label in np.unique(y):
        idx = np.where(target = label)
        np.random.shuffle(idx)
        # take the first num_lbl from shuffled indices
        lbl_idx.extend(idx[:num_lbl])
        if num_ulbl is None:
            ulbl_idx.extend(idx[num_lbl:])
        else:
            ulbl_idx.extend(idx[num_lbl: num_lbl + num_ulbl])

    if return_idx:
        return lbl_idx, ulbl_idx

    return X[lbl_idx], y[lbl_idx], X[ulbl_idx], y[ulbl_idx]

def split_lb_ulb_imbalanced():
    pass

class BaseDataset(Dataset):
    """
    A class to store a dataset and apply any transformations required by
    an algorithm
    """
    def __init__(self, X, y=None, num_classes=None,
                 weak_transform=None, medium_transform=None,
                 strong_transform=None, onehot=False):
        """
        Initialise an SSL dataset. This can be either a labelled or unlabelled
        dataset.

        X: the features of the data
        y: the class labels
        num_classes: the number of target classes in the data
        weak_transform: a basic transformation that is always applied
        medium_transform: a transformation that may be applied
        strong_transform: a transformation that may be applied
        onehot: True if the labels should be converted to one-hot vectors
        """

        super().__init__()

        self.X = X
        self.y = y
        self.num_classes = num_classes

        self.weak_transform = weak_transform
        self.medium_transform = medium_transform
        self.strong_transform = strong_transform


    def __len__(self):
        """
        Return the number of elements in the dataset
        """
        return(len(self.X))


    def __getitem__(self, index):
        """
        Returns an item from the dataset with any transformations applied

        Args:
            index: The index of the observation to return
        Returns:
            A 4-tuple (X_w, y, X_m, X_s) of the weakly transformed data, label,
            medium and strong transformed data. If a weak transform is not
            specified, return the unmodified observation. If the data is
            unlabelled, or medium/strong is not specified, these all are None
        """


        X = self.X[index]
        y = self.y[index] if self.y is not None else None

        X_w = self.weak_transform(X) if self.weak_tranform is not None else X
        X_m = self.medium_transform(X) if self.medium_tranform is not None else None
        X_s = self.strong_transform(X) if self.strong_tranform is not None else None

        return X_w, y, X_m, X_s


class SSLDatasetInterface:

    def __init__(self, lbl_dataset=None, ulbl_dataset=None,
                 lbl_loader=None, ulbl_loader=None,
                 eval_dataset=None, eval_loader=None):

        self.lbl_dataset = lbl_dataset
        self.ulbl_dataset = ulbl_dataset
        self.lbl_loader = lbl_loader
        self.ulbl_loader = ulbl_loader
        self.eval_dataset = eval_dataset
        self.get_eval_loader = eval_loader

    def get_lbl_dataset(self):
        return self.lbl_dataset

    def get_ulbl_dataset(self):
        return self.ulbl_dataset

    def get_lbl_loader(self):
        return self.lbl_loader

    def get_ulbl_loader(self):
        return self.ulbl_loader

    def get_eval_dataset(self):
        return self.eval_dataset

    def get_eval_loader(self):
        return self.eval_loader





