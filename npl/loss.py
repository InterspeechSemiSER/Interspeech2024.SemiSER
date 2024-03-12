import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
