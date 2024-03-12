import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LogitMargin(nn.Module):
    def __init__(self,
                 margin: float = 0.35,
                 alpha: float = 0.2,
                 ignore_index: int = -100,
                 schedule: str = "",
                 mu: float = 0,
                 max_alpha: float = 100.0,
                 step_size: int = 100):
        super().__init__()
        assert schedule in ("", "add", "multiply", "step")
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.mu = mu
        self.schedule = schedule
        self.max_alpha = max_alpha
        self.step_size = step_size

        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "loss_margin_l1"

    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        return diff

    def forward(self, inputs, targets):

        inputs = inputs.view(-1, 4)
        targets = targets.view(-1)

        loss_ce = self.cross_entropy(inputs, targets)
        # get logit distance
        diff = self.get_diff(inputs)
        # linear penalty where logit distances are larger than the margin
        loss_margin = F.relu(diff-self.margin).mean()
        loss = loss_ce + self.alpha * loss_margin

        return loss

def npl_loss(logits_w, logits_s, k, num_labels, threshold=0.95):
    softmax_w = torch.softmax(logits_w, dim=-1).to("cuda").view(-1, num_labels)
    softmax_s = torch.softmax(logits_s, dim=-1).to("cuda").view(-1, num_labels)

    _, top_indices_w = torch.topk(-softmax_w, k, largest=True)

    pre_negative_label1 = torch.zeros_like(softmax_w).scatter(
        dim=-1, index=top_indices_w, value=1
    )

    pre_negative_label2 = torch.where(
        softmax_w < 1 - threshold**3,
        torch.ones_like(softmax_w),
        torch.zeros_like(softmax_w),
    )

    pseudo_negative_label = torch.where(
        (pre_negative_label1 == 1) & (pre_negative_label2 == 1),
        torch.ones_like(softmax_w),
        torch.zeros_like(softmax_w),
    )

    loss = -torch.log(1 - softmax_s) * pseudo_negative_label
    loss = loss.sum(axis=-1).mean()
    print("loss------------", loss)
    return loss
