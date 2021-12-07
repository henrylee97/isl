import torch.nn as nn


def DenseNet(in_features: int, n_class: int, n_hidden: int = 512, n_layers: int = 6, bias: bool = True) -> nn.Module:
    layers = [nn.Linear(in_features, n_hidden, bias=bias), nn.ReLU()]
    for _ in range(n_layers - 2):
        layers.append(nn.Linear(n_hidden, n_hidden, bias=bias))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(n_hidden, n_class, bias=bias))
    layers.append(nn.Softmax())
    return nn.Sequential(*layers)
