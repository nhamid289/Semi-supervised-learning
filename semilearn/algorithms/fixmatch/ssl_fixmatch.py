import torch
from semilearn.algorithms import SSLAlgorithm
from semilearn.algorithms.utils import smooth_targets
from semilearn.utils.data import SSLBatch
from semilearn.core.criterions import CELoss, ConsistencyLoss

class SSLFixMatch(SSLAlgorithm):

    def __init__(self, use_hard_label=False, T=1.0, lambda_u=1.0,
                 conf_threshold=0.95, sup_loss_func = CELoss(),
                 unsup_loss_func = ConsistencyLoss()):
        """

        """
        super().__init__()

        self.use_hard_label = use_hard_label
        self.T = T
        self.lambda_u = lambda_u
        self.conf_threshold = conf_threshold

        self.sup_loss_func = sup_loss_func
        self.unsup_loss_func = unsup_loss_func

    def train_step(self, model, batch:SSLBatch):
        """

        """
        out_lbl_weak = model(batch.X_lbl_weak)
        out_ulbl_strong = model(batch.X_ulbl_strong)

        with torch.no_grad():
            out_ulbl_weak = model(batch.X_ulbl_weak)

        sup_loss = self.sup_loss_func(out_lbl_weak['logits'], batch.y_lbl, reduction='mean')

        probs_ulbl_w = self._compute_prob(out_ulbl_weak['logits'].detach())

        with torch.no_grad():
            max_probs, _ = torch.max(probs_ulbl_w, dim=-1)
            mask = max_probs.ge(self.conf_threshold).to(max_probs.dtype)

        pseudo_label = self._gen_ulb_targets(logits=probs_ulbl_w,
                                            use_hard_label=self.use_hard_label,
                                            T=self.T,
                                            softmax=False)

        unsup_loss = self.unsup_loss_func(out_ulbl_strong['logits'], pseudo_label, 'ce', mask=mask)

        total_loss = sup_loss + self.lambda_u * unsup_loss

        return total_loss


    def _mask(self, logits_x_ulb, softmax_x_ulb=True, *args, **kwargs):
        with torch.no_grad():
            if softmax_x_ulb:
                # probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
                probs_x_ulb = self._compute_prob(logits_x_ulb.detach())
            else:
                # logits is already probs
                probs_x_ulb = logits_x_ulb.detach()
            max_probs, _ = torch.max(probs_x_ulb, dim=-1)
            mask = max_probs.ge(self.p_cutoff).to(max_probs.dtype)
            return mask

    def _compute_prob(self, logits):
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def _gen_ulb_targets(self, logits, use_hard_label=True, T=1.0, softmax=True,
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
            pseudo_label = self._compute_prob(logits / T)

        else:
            # inputs logits converted to probabilities already
            pseudo_label = logits

        return pseudo_label
