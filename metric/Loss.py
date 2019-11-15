import torch.nn as nn
import torch.nn.functional as F


class CosRankingLoss(nn.Module):
    def __init__(self, opt):
        super(CosRankingLoss, self).__init__()
        self.margin = opt.cos_ranking_loss_margin

    def forward(self, code_repr, good_comment_repr, bad_comment_repr):
        good_sim = F.cosine_similarity(code_repr, good_comment_repr)
        bad_sim = F.cosine_similarity(code_repr, bad_comment_repr)

        loss = (self.margin - good_sim + bad_sim).clamp(min=1e-6).mean()

        return loss
