import torch.nn as nn


def LeNet5(in_channels: int, n_class: int, use_relu: bool = False) -> nn.Module:
    activation_function = nn.ReLU if use_relu else nn.Tanh
    model = nn.Sequential(
        nn.Conv2d(in_channels, 6, kernel_size=5, stride=1),
        activation_function(),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(6, 16, kernel_size=5, stride=1),
        activation_function(),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(16, 120, kernel_size=5, stride=1),
        activation_function(),
        nn.Flatten(),
        nn.Linear(120, 84),
        activation_function(),
        nn.Linear(84, n_class),
        nn.Softmax()
    )
    return model
