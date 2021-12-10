import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPLoss(nn.Module):

    def __init__(self, n_class: int, n_layers: int = 3, hidden_size: int = 20) -> nn.Module:
        super(MLPLoss, self).__init__()

        self.n_class = n_class

        layers = [nn.Linear(2 * n_class, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, y_hat: torch.Tensor, y_star: torch.Tensor) -> torch.Tensor:
        y_star = F.one_hot(y_star, self.n_class)
        contat_logits = torch.cat([y_hat, y_star], dim=-1)
        loss_foreach_batch = self.layers(contat_logits)
        return loss_foreach_batch.sum()
