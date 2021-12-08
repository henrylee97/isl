from argparse import ArgumentParser
from typing import Union
from isl.datasets import CIFAR10
from isl.datasets import CIFAR100
from isl.models import LeNet5
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

    parser = ArgumentParser(
        description='ISL Example Script: Test LeNet5 with CIFAR')
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
    parser.add_argument('--verbose-every', type=int, default=10,
                        metavar='<int>', help='Print status every <int> steps (default=10)')
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

    model.eval()
    criterion.eval()

    loss = 0
    accuracy = 0

    with torch.no_grad():
        for current_step, (x, y_star) in enumerate(test):

            if use_gpu:
                x = x.cuda()
                y_star = y_star.cuda()

            y_hat = model(x)

            current_loss = criterion(y_hat, y_star).cpu().item()
            loss += current_loss

            _, y_hat = torch.max(y_hat, 1)
            current_accuracy = (y_hat == y_star).sum().cpu().item()
            current_accuracy /= args.batch_size
            accuracy += current_accuracy

            print(f'\r[{current_step + 1}/{len(test)}] loss: {current_loss:.4f} '
                  f'accuracy: {current_accuracy * 100:.2f}%', end='', flush=True)
        print()
    print(f'Total loss: {loss / len(test):.4f} '
          f'total accuracy: {accuracy / len(test) * 100:.2f}%')


if __name__ == '__main__':
    main(*sys.argv[1:])
