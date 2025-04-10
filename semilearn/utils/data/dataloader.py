from torch.utils.data import DataLoader


class SSLBatch():
    def __init__(self, X_lbl_weak, X_lbl_medium, X_lbl_strong, y_lbl,
                 X_ulbl_weak, X_ulbl_medium, X_ulbl_strong, y_ulbl):

        self.X_lbl_weak = X_lbl_weak
        self.X_lbl_medium = X_lbl_medium
        self.X_lbl_strong = X_lbl_strong
        self.y_lbl = y_lbl
        self.X_ulbl_weak = X_ulbl_weak
        self.X_ulbl_medium = X_ulbl_medium
        self.X_ulbl_strong = X_ulbl_strong
        self.y_ulbl = y_ulbl

class SSLDataLoader(DataLoader):

    def __init__(self, lbl_dataset, ulbl_dataset,
                 lbl_batch_size, ulbl_batch_size,
                 shuffle_lbl=True, shuffle_ulbl=True,
                 num_workers=1):

        self.lbl_dataset = lbl_dataset
        self.ulbl_dataset = ulbl_dataset

        self.lbl_loader = DataLoader(lbl_dataset, batch_size=lbl_batch_size,
                                     shuffle=shuffle_lbl, num_workers=num_workers)
        self.ulbl_loader = DataLoader(ulbl_dataset, batch_size=ulbl_batch_size,
                                      shuffle=shuffle_ulbl, num_workers=num_workers)

    def __iter__(self):
        self.lbl_iter = iter(self.lbl_loader)
        self.ulbl_iter = iter(self.ulbl_loader)
        return self

    def __next__(self):
        lbl_batch = next(self.lbl_iter)
        ulbl_batch = next(self.ulbl_iter)
        return SSLBatch(*lbl_batch, *ulbl_batch)

    def __len__(self):
        return min(len(self.lbl_loader), len(self.ulbl_loader))