import torch
from semilearn.algorithms import SSLAlgorithm
from semilearn.utils.data import SSLBatch
from semilearn.core.criterions import CELoss

class SSLFullySupervised(SSLAlgorithm):

    def __init__(self, loss_func=None):
        super().__init__()

        if loss_func is None:
            ce_loss = CELoss()
            self.loss_func = lambda y, y_pred: ce_loss(y, y_pred, reduction="mean")
        else:
            self.loss_func = loss_func

    def train_step(self, model, batch:SSLBatch):

        X = torch.cat([batch.X_lbl, batch.X_ulbl], dim=0)
        y = torch.cat([batch.y_lbl, batch.y_ulbl], dim=0)
        out_X = model(X)

        loss = self.loss_func(out_X['logits'], y)

        return loss





