import torch


class QuantileLoss(torch.nn.Module):
    def __init__(self, quantiles, device, weights=None):
        super().__init__()
        self.quantiles = (
            torch.tensor(quantiles).float().view(1, -1).to(device)
        )  # shape: (1, num_quantiles)
        self.weights = (
            None
            if weights is None
            else torch.tensor(weights, dtype=torch.float32, device=device).view(1, -1)
        )

    def forward(self, preds, target):
        # preds: (batch_size, num_quantiles)
        # target: (batch_size, 1)
        target = target.unsqueeze(1)  # make sure shape is (batch_size, 1)
        errors = target - preds
        losses = torch.maximum(self.quantiles * errors, (self.quantiles - 1) * errors)
        if self.weights is not None:
            losses = losses * self.weights
        return losses.mean()
