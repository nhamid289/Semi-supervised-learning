import torch
import numpy as np
from inspect import signature


from semilearn.utils.data import SSLBatch


class SSLAlgorithm:

    def __init__(self):
        pass



    def train_step(self, model, batch:SSLBatch):
        """
        train_step specific to each algorithm
        """
        # implement train step for each algorithm
        # compute loss
        # update model
        # record log_dict
        # return log_dict
        raise NotImplementedError

