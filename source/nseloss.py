import torch


class NSELoss(torch.nn.Module):

    def __init__(self):
        super(NSELoss, self).__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        SS_res = torch.sum(torch.square(y_true - y_pred))
        SS_tot = torch.sum(torch.square(y_true - torch.mean(y_true)))
        return (1 - SS_res / (SS_tot + 1e-7))
