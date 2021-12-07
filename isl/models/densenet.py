from torch import nn


def DenseNet(in_features: int, n_class: int, bias: bool = True) -> nn.Module:
    model = nn.Sequential(
        nn.Linear(in_features, 512, bias=bias),
        nn.ReLU(),
        nn.Linear(512, 256, bias=bias),
        nn.ReLU(),
        nn.Linear(256, 128, bias=bias),
        nn.ReLU(),
        nn.Linear(128, 64, bias=bias),
        nn.ReLU(),
        nn.Linear(64, 32, bias=bias),
        nn.ReLU(),
        nn.Linear(32, n_class, bias=bias),
        nn.Softmax(),
    )
    return model
