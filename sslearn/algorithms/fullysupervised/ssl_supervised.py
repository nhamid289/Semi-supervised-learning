import torch
from semilearn.algorithms import SSLAlgorithm
from semilearn.utils.data import SSLBatch
from semilearn.core.criterions import CELoss

class SSLSupervised(SSLAlgorithm):

    def __init__(self, loss_func=None):
        super().__init__()

        if loss_func is None:
            ce_loss = CELoss()
            self.loss_func = lambda y, y_pred: ce_loss(y, y_pred, reduction="mean")
        else:
            self.loss_func = loss_func

    def train_step(self, model, batch:SSLBatch):

        out_X = model(batch.X_lbl)

        loss = self.loss_func(out_X['logits'], batch.y_lbl)

        return loss
