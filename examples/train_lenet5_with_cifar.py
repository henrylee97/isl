from argparse import ArgumentParser
from isl.datasets import CIFAR10
from isl.datasets import CIFAR100
from isl.models import LeNet5
from torch.utils.data import DataLoader
import sys
import torch
import torch.nn as nn
import torch.optim as optim


def load_cifar(batch_size: int = 100, use_gpu: bool = False, use_cifar100: bool = False) -> DataLoader:
    CIFAR = CIFAR100 if use_cifar100 else CIFAR10
    dataset = CIFAR(download=True)
    train = DataLoader(dataset.train, batch_size=batch_size, shuffle=True, pin_memory=use_gpu)
    return train


def load_model() -> nn.Module:
    return LeNet5(in_channels=3, n_class=100)


def main(*argv):

    parser = ArgumentParser(description='ISL Example Script: Train LeNet5 with CIFAR')
    parser.add_argument('--cuda', action='store_true', help='Use GPU if avaiable')
    parser.add_argument('--cifar100', action='store_true', help='Use CIFAR100 instead of CIFAR10')
    parser.add_argument('--batch-size', type=int, default=100, nargs=1, metavar='<int>', help='Set batch size (default=100)')
    parser.add_argument('-s', '--steps', type=int, default=500, metavar='<int>', help='How many times to update model (default=500)')
    parser.add_argument('--save-every', type=int, default=500, metavar='<int>', help='Save every <int> steps (default=500)')
    parser.add_argument('--verbose-every', type=int, default=10, metavar='<int>', help='Print status every <int> steps (default=10)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.1, metavar='<float>', help='Set learning rate (default=0.1)')
    args = parser.parse_args(argv)

    use_gpu = args.cuda and torch.cuda.is_available()

    train = load_cifar(args.batch_size, use_gpu)
    iterator = iter(train)

    model = load_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    if use_gpu:
        model = model.cuda()

    current_step = 0
    running_loss = 0
    running_accuracy = 0
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

        running_loss += loss.cpu().item()

        _, y_hat = torch.max(y_hat, 1)
        accuracy = (y_hat == y_star).sum().item()
        running_accuracy += accuracy / args.batch_size

        if current_step % args.verbose_every == 0:
            print(f'[{current_step}/{args.steps}] loss: {running_loss / args.verbose_every:.4f} accuracy: {running_accuracy / args.verbose_every * 100:.2f}%')
            running_loss = 0
            running_accuracy = 0


if __name__ == '__main__':
    main(*sys.argv[1:])
