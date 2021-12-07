import torch.nn as nn


def LeNet5(in_channels: int, n_class: int) -> nn.Module:
    model = nn.Sequential(
        nn.Conv2d(in_channels, 6, kernel_size=5, stride=1),
        nn.Tanh(),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(6, 16, kernel_size=5, stride=1),
        nn.Tanh(),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(16, 120, kernel_size=5, stride=1),
        nn.Tanh(),
        nn.Flatten(),
        nn.Linear(120, 84),
        nn.Tanh(),
        nn.Linear(84, n_class),
        nn.Softmax()
    )
    return model
