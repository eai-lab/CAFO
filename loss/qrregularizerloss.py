import torch
import torch.nn as nn
import torch.nn.functional as F


class QRRegularizerLoss(nn.Module):
    def __init__(self, cfg):
        super(QRRegularizerLoss, self).__init__()
        self.cfg = cfg
        self.gamma = cfg.loss.gamma

    def forward(self, features, labels=None):
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        # squeeze dim
        features = features.squeeze()

        # construct class prototypes
        unique_labels = torch.unique(labels)
        prototypes = torch.zeros(unique_labels.shape[0], features.shape[-1]).to(device)
        for i, label in enumerate(unique_labels):
            prototypes[i] = features[labels == label].mean(dim=0)

        if self.cfg.loss.feature_wise:
            _, R = torch.qr(prototypes)
        else:
            _, R = torch.qr(prototypes.t())
        # get only the offdiagnoal elements of R
        R_offdiagonal = abs(torch.triu(R, diagonal=1))
        # count the number of offdiagonal elements
        counts = abs(torch.triu(torch.ones_like(R_offdiagonal), diagonal=1)).sum().item()

        # compute the mean of the offdiagonal elements
        mean_loss = R_offdiagonal.sum() / counts

        return self.gamma * mean_loss
