import torch
import numpy as np
from inspect import signature

from semilearn.algorithms.utils import smooth_targets



class SSLAlgorithm:

    def __init__(self, config, **kwargs):
        pass



    def train_step(self, model, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s,
                   **kwargs):
        """
        train_step specific to each algorithm
        """
        # implement train step for each algorithm
        # compute loss
        # update model
        # record log_dict
        # return log_dict
        raise NotImplementedError

    def compute_prob(self, logits):
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def gen_ulb_targets(self, logits, use_hard_label=True, T=1.0, softmax=True,
                        label_smoothing=0.0):
        """
        generate pseudo-labels from logits/probs

        Args:
            algorithm: base algorithm
            logits: logits (or probs, need to set softmax to False)
            use_hard_label: flag of using hard labels instead of soft labels
            T: temperature parameters
            softmax: flag of using softmax on logits
            label_smoothing: label_smoothing parameter
        """

        logits = logits.detach()
        if use_hard_label:
            # return hard label directly
            pseudo_label = torch.argmax(logits, dim=-1)
            if label_smoothing:
                pseudo_label = smooth_targets(logits, pseudo_label,
                                              label_smoothing)

        # return soft label
        elif softmax:
            # pseudo_label = torch.softmax(logits / T, dim=-1)
            pseudo_label = self.compute_prob(logits / T)

        else:
            # inputs logits converted to probabilities already
            pseudo_label = logits

        return pseudo_label

    def process_batch(self, input_args=None, **kwargs):
        """
        process batch data, send data to cuda
        NOTE **kwargs should have the same arguments to train_step function as keys to work properly
        """
        if input_args is None:
            input_args = signature(self.train_step).parameters
            input_args = list(input_args.keys())

        input_dict = {}

        for arg, var in kwargs.items():
            if not arg in input_args:
                continue

            if var is None:
                continue

            # send var to cuda
            if isinstance(var, dict):
                var = {k: v.cuda(self.config.gpu) for k, v in var.items()}
            else:
                var = var.cuda(self.config.gpu)
            input_dict[arg] = var
        return input_dict