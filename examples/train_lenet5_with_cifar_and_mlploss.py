from argparse import ArgumentParser
from typing import Tuple
from typing import Union
from isl.datasets import CIFAR10
from isl.datasets import CIFAR100
from isl.loss import MLPLoss
from isl.models import LeNet5
from pathlib import Path
from torch.utils.data import DataLoader
import sys
import torch
import torch.nn as nn
import torch.optim as optim


def load_cifar(batch_size: int = 100, use_gpu: bool = False, use_cifar100: bool = False) -> DataLoader:
    CIFAR = CIFAR100 if use_cifar100 else CIFAR10
    dataset = CIFAR(download=True)
    train = DataLoader(dataset.train, batch_size=batch_size,
                       shuffle=True, pin_memory=use_gpu)
    return train


def make_model(n_class: int, use_relu: bool = False) -> nn.Module:
    return LeNet5(in_channels=3, n_class=n_class, use_relu=use_relu)


def load_model(path: Union[str, Path], model: nn.Module, optimizer: optim.Optimizer) -> Tuple[nn.Module, optim.Optimizer, int]:
    path = Path(path)
    loaded = torch.load(path)
    model.load_state_dict(loaded['model_state_dict'])
    optimizer.load_state_dict(loaded['optimizer_state_dict'])
    return model, optimizer, loaded['current_step']


def save_model(path: Union[str, Path], model: nn.Module, optimizer: optim.Optimizer, current_step: int) -> None:
    path = Path(path)
    torch.save({
        'current_step': current_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)


def main(*argv) -> None:

    parser = ArgumentParser(
        description='ISL Example Script: Train LeNet5 with CIFAR')
    parser.add_argument('--cuda', action='store_true',
                        help='Use GPU if avaiable')
    parser.add_argument('--cifar100', action='store_true',
                        help='Use CIFAR100 instead of CIFAR10')
    parser.add_argument('--relu', action='store_true',
                        help='Use ReLU as activation function for hidden layers')
    parser.add_argument('--batch-size', type=int, default=100,
                        metavar='<int>', help='Set batch size (default=100)')
    parser.add_argument('-s', '--steps', type=int, default=500,
                        metavar='<int>', help='How many times to update model (default=500)')
    parser.add_argument('--save-every', type=int, default=500,
                        metavar='<int>', help='Save every <int> steps (default=500)')
    parser.add_argument('--verbose-every', type=int, default=10,
                        metavar='<int>', help='Print status every <int> steps (default=10)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.1,
                        metavar='<float>', help='Set learning rate (default=0.1)')
    parser.add_argument('--path', type=str, default='./result', metavar='<path>',
                        help='Path to save trained models (default=./result)')
    parser.add_argument('--continual', type=str, default=None, metavar='<path>',
                        help='Continual training from existing model in <path>')
    args = parser.parse_args(argv)

    use_gpu = args.cuda and torch.cuda.is_available()

    path = Path(args.path)
    path.mkdir(parents=True, exist_ok=True)

    train = load_cifar(args.batch_size, use_gpu, args.cifar100)
    iterator = iter(train)

    n_class = 100 if args.cifar100 else 10
    model = make_model(n_class, args.relu)
    criterion = MLPLoss(n_class, 3, 20)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    current_step = 0

    if args.continual is not None:
        model, optimizer, current_step = load_model(
            args.continual, model, optimizer)
        print(f'Model loaded: {args.continual}')

    if use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()

    model.train()
    criterion.train()

    verbose_running_loss = 0
    verbose_running_accuracy = 0

    save_running_accuracy = 0
    save_best_accuracy = 0
    while current_step < args.steps:
        current_step += 1

        try:
            x, y_star = next(iterator)
        except StopIteration:
            iterator = iter(train)
            x, y_star = next(iterator)
        if use_gpu:
            x = x.cuda()
            y_star = y_star.cuda()

        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y_star)
        loss.backward()
        optimizer.step()

        verbose_running_loss += loss.cpu().item()

        _, y_hat = torch.max(y_hat, 1)
        accuracy = (y_hat == y_star).sum().item()
        verbose_running_accuracy += accuracy / args.batch_size
        save_running_accuracy += accuracy / args.batch_size

        if current_step % args.verbose_every == 0:
            print(f'[{current_step}/{args.steps}] loss: {verbose_running_loss / args.verbose_every:.4f} accuracy: {verbose_running_accuracy / args.verbose_every * 100:.2f}%')
            verbose_running_loss = 0
            verbose_running_accuracy = 0

        if current_step % args.save_every == 0:
            model_path = path / \
                f'lenet5{"_relu" if args.relu else ""}_with_cifar{100 if args.cifar100 else 10}_{current_step}steps.pt'
            save_model(model_path, model, optimizer, current_step)
            print(f'Model saved: {model_path}')

            if save_running_accuracy / args.verbose_every > save_best_accuracy:
                save_best_accuracy = save_running_accuracy / args.verbose_every
                model_path = path / \
                    f'lenet5{"_relu" if args.relu else ""}_with_cifar{100 if args.cifar100 else 10}_best.pt'
                save_model(model_path, model, optimizer, current_step)
                print(f'Model saved: {model_path}')
            save_running_accuracy = 0


if __name__ == '__main__':
    main(*sys.argv[1:])
