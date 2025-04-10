from torch.utils.data import Dataset

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



