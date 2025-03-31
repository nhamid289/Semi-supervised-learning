



import torch
from semilearn.algorithms import SSLAlgorithm
from semilearn.core.criterions import CELoss, ConsistencyLoss

class SSLFixMatch(SSLAlgorithm):

    def __init__(self, config, **kwargs):
        """

        """
        super().__init__(config, **kwargs)
        self.config = config
        self.ce_loss = CELoss()
        self.consistency_loss = ConsistencyLoss()

    def train_step(self, model, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        """

        """
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = model(x_lb)
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())

            # compute mask
            # mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # # generate unlabeled targets using pseudo label hook
            # pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
            #                               logits=probs_x_ulb_w,
            #                               use_hard_label=self.use_hard_label,
            #                               T=self.T,
            #                               softmax=False)

            pseudo_label = self.gen_ulb_targets(logits=probs_x_ulb_w,
                                                use_hard_label=self.config.use_hard_label,
                                                T=self.config.T,
                                                softmax=False)

            unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label, 'ce')

            total_loss = sup_loss + self.lambda_u * unsup_loss

        return total_loss
        # out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        # log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
        #                                  unsup_loss=unsup_loss.item(),
        #                                  total_loss=total_loss.item(),
        #                                  util_ratio=mask.float().mean().item())
        # return out_dict, log_dict


