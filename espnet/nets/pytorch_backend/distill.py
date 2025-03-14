import torch
import torch.nn as nn



class DistillLoss(nn.Module):
    def __init__(self, l2_weight, l1_weight, cos_weight, cos_type):
        super().__init__()
        self.l2_weight = l2_weight
        self.l1_weight = l1_weight
        self.cos_weight = cos_weight
        self.cos_type = cos_type
        assert cos_type in ["raw", "log_sig"], cos_type

        if l2_weight != 0:
            self.mse_loss = nn.MSELoss()
        if l1_weight != 0:
            self.l1_loss = nn.L1Loss()
        if cos_weight != 0:
            self.cos_sim = nn.CosineSimilarity(dim=-1)
    
    def __repr__(self) -> str:
        return "{}(l2={}, l1={}, {}_cos={})".format(
            self.__class__.__name__,
            self.l2_weight,
            self.l1_weight,
            self.cos_type,
            self.cos_weight,
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Args:
            input: (batch, layer, time, feature)
            target: same shape as input
        """
        loss_mse = 0
        loss_l1 = 0
        loss_cos = 0
        if self.l2_weight != 0:
            loss_mse = self.mse_loss(input, target)
        if self.l1_weight != 0:
            loss_l1 = self.l1_loss(input, target)
        if self.cos_weight != 0:    # maximize cosine similarity
            if self.cos_type == "raw":
                loss_cos = -self.cos_sim(input, target).mean()
            elif self.cos_type == "log_sig":
                loss_cos = -self.cos_sim(input, target).sigmoid().log().mean()
            else:
                raise ValueError

        loss = self.l2_weight * loss_mse + self.l1_weight * loss_l1 + self.cos_weight * loss_cos

        return loss, (loss_mse, loss_l1, loss_cos)