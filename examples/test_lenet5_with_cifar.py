from argparse import ArgumentParser
from typing import Union
from isl.datasets import CIFAR10
from isl.datasets import CIFAR100
from isl.models import LeNet5
from isl.trainer import simple_validation
from pathlib import Path
from torch.utils.data import DataLoader
import sys
import torch
import torch.nn as nn


def load_cifar(batch_size: int = 100, use_gpu: bool = False, use_cifar100: bool = False) -> DataLoader:
    CIFAR = CIFAR100 if use_cifar100 else CIFAR10
    dataset = CIFAR(download=True)
    test = DataLoader(dataset.test, batch_size=batch_size,
                      shuffle=True, pin_memory=use_gpu)
    return test


def make_model(n_class: int, use_relu: bool = False) -> nn.Module:
    return LeNet5(in_channels=3, n_class=n_class, use_relu=use_relu)


def load_model(path: Union[str, Path], model: nn.Module) -> nn.Module:
    path = Path(path)
    loaded = torch.load(path)
    model.load_state_dict(loaded['model_state_dict'])
    return model


def main(*argv) -> None:

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True, metavar='<path>',
                        help='Path to save trained models (required)')
    parser.add_argument('--cuda', action='store_true',
                        help='Use GPU if avaiable')
    parser.add_argument('--cifar100', action='store_true',
                        help='Use CIFAR100 instead of CIFAR10')
    parser.add_argument('--relu', action='store_true',
                        help='Use ReLU as activation function for hidden layers')
    parser.add_argument('--batch-size', type=int, default=100,
                        metavar='<int>', help='Set batch size (default=100)')
    args = parser.parse_args(argv)

    use_gpu = args.cuda and torch.cuda.is_available()

    test = load_cifar(args.batch_size, use_gpu, args.cifar100)

    model = make_model(100 if args.cifar100 else 10, args.relu)
    model = load_model(args.model, model)
    print(f'Model loaded: {args.model}')

    criterion = nn.CrossEntropyLoss()

    if use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()

    loss, accuracy = simple_validation(
        model, criterion, test, verbose=1, use_gpu=use_gpu)
    print(f'Total loss: {loss:.4f} total accuracy: {accuracy * 100:.2f}%')


if __name__ == '__main__':
    main(*sys.argv[1:])
