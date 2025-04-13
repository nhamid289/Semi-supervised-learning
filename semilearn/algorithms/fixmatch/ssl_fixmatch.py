



import torch
from semilearn.algorithms import SSLAlgorithm
from semilearn.utils.data import SSLBatch
from semilearn.core.criterions import CELoss, ConsistencyLoss

class SSLFixMatch(SSLAlgorithm):

    def __init__(self, use_hard_label=False, T=0.5, lambda_u = 0.5,
                 sup_loss_func = CELoss(), unsup_loss_func = ConsistencyLoss()):
        """

        """
        super().__init__()

        self.use_hard_label = use_hard_label
        self.T = T
        self.lambda_u = lambda_u

        self.sup_loss_func = sup_loss_func
        self.unsup_loss_func = unsup_loss_func

    def train_step(self, model, batch:SSLBatch):
        """

        """
        out_lbl = model(batch.X_lbl)
        out_lbl_weak = model(batch.X_lbl_weak)
        out_ulbl_strong = model(batch.X_ulbl_strong)

        with torch.no_grad():
            out_ulbl = model(batch.X_ulbl_weak)

        sup_loss = self.sup_loss_func(out_lbl['logits'], batch.y_lbl, reduction='mean')

        probs_ulbl_w = self.compute_prob(out_ulbl['logits'].detach())

        pseudo_label = self.gen_ulb_targets(logits=probs_ulbl_w,
                                            use_hard_label=self.use_hard_label,
                                            T=self.T,
                                            softmax=False)

        unsup_loss = self.unsup_loss_func(out_ulbl_strong['logits'], pseudo_label, 'ce')

        total_loss = sup_loss + self.lambda_u * unsup_loss

        return total_loss


        # logits_x_lb = outs_x_lb['logits']
        # feats_x_lb = outs_x_lb['feat']

        # outs_x_ulb_s = model(x_ulb_s)

        # logits_x_ulb_s = outs_x_ulb_s['logits']
        # feats_x_ulb_s = outs_x_ulb_s['feat']




        # # inference and calculate sup/unsup losses
        # with config.amp_cm():
        #     if config.use_cat:
        #         inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
        #         outputs = model(inputs)
        #         logits_x_lb = outputs['logits'][:num_lb]
        #         logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
        #         feats_x_lb = outputs['feat'][:num_lb]
        #         feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
        #     else:
        #         outs_x_lb = model(x_lb)
        #         logits_x_lb = outs_x_lb['logits']
        #         feats_x_lb = outs_x_lb['feat']
        #         outs_x_ulb_s = model(x_ulb_s)
        #         logits_x_ulb_s = outs_x_ulb_s['logits']
        #         feats_x_ulb_s = outs_x_ulb_s['feat']
        #         with torch.no_grad():
        #             outs_x_ulb_w = model(x_ulb_w)
        #             logits_x_ulb_w = outs_x_ulb_w['logits']
        #             feats_x_ulb_w = outs_x_ulb_w['feat']
        # feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}



        # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
        # probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())

        # compute mask
        # mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

        # # generate unlabeled targets using pseudo label hook
        # pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
        #                               logits=probs_x_ulb_w,
        #                               use_hard_label=self.use_hard_label,
        #                               T=self.T,
        #                               softmax=False)

        # pseudo_label = self.gen_ulb_targets(logits=probs_x_ulb_w,
        #                                     use_hard_label=self.config.use_hard_label,
        #                                     T=self.config.T,
        #                                     softmax=False)

        # unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label, 'ce')

        # total_loss = sup_loss + self.config.lambda_u * unsup_loss

        # return total_loss
        # out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        # log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
        #                                  unsup_loss=unsup_loss.item(),
        #                                  total_loss=total_loss.item(),
        #                                  util_ratio=mask.float().mean().item())
        # return out_dict, log_dict


