from argparse import ArgumentParser
from typing import Tuple
from typing import Union
from isl.datasets import CIFAR10
from isl.datasets import CIFAR100
from isl.models import LeNet5
from isl.trainer import simple_trainer
from isl.trainer import simple_validation
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
    test = DataLoader(dataset.test, batch_size=batch_size, pin_memory=use_gpu)
    return train, test


def make_model(n_class: int, use_relu: bool = False) -> nn.Module:
    return LeNet5(in_channels=3, n_class=n_class, use_relu=use_relu)


def load_model(path: Union[str, Path],
               model: nn.Module,
               optimizer: optim.Optimizer) -> Tuple[nn.Module, optim.Optimizer, int, float]:
    path = Path(path)
    loaded = torch.load(path)
    model.load_state_dict(loaded['model_state_dict'])
    optimizer.load_state_dict(loaded['optimizer_state_dict'])
    return model, optimizer, loaded['epochs'], loaded['best_acc']


def save_model(path: Union[str, Path],
               model: nn.Module,
               optimizer: optim.Optimizer,
               epochs: int,
               best_acc: float) -> None:
    path = Path(path)
    torch.save({
        'epochs': epochs,
        'best_acc': best_acc,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)


def main(*argv) -> None:

    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true',
                        help='Use GPU if avaiable')
    parser.add_argument('--cifar100', action='store_true',
                        help='Use CIFAR100 instead of CIFAR10')
    parser.add_argument('--relu', action='store_true',
                        help='Use ReLU as activation function for hidden layers')
    parser.add_argument('--batch-size', type=int, default=100,
                        metavar='<int>', help='Set batch size (default=100)')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        metavar='<int>', help='Set epochs (default=1)')
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

    train, test = load_cifar(args.batch_size, use_gpu, args.cifar100)

    model = make_model(100 if args.cifar100 else 10, args.relu)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    current_epoch = 0
    best_acc = 0

    if args.continual is not None:
        model, optimizer, current_epoch, best_acc = load_model(
            args.continual, model, optimizer)
        print(f'Model loaded: {args.continual}')

    if use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()

    for _ in range(current_epoch, args.epochs):
        current_epoch += 1
        print(f'Epoch {current_epoch}')
        model, (train_loss, train_acc) = simple_trainer(model, criterion, optimizer,
                                                        train, verbose=1, use_gpu=use_gpu)
        val_loss, val_acc = simple_validation(model, criterion, test,
                                              use_gpu=use_gpu)
        print(f'Train loss: {train_loss:.4f} Train accuracy: {train_acc:.4f} '
              f'Validation loss: {val_loss:.4f} Validation accuracy: {val_acc:.4f}')

        model_path = path / \
            f'lenet5{"_relu" if args.relu else ""}_cifar{100 if args.cifar100 else 10}_{current_epoch}epochs.pt'
        save_model(model_path, model, optimizer, current_epoch, best_acc)

        if val_acc > best_acc:
            model_path = path / \
                f'lenet5{"_relu" if args.relu else ""}_cifar{100 if args.cifar100 else 10}_best.pt'
            save_model(model_path, model, optimizer, current_epoch, best_acc)
            print(f'Best model saved: {model_path}')


if __name__ == '__main__':
    main(*sys.argv[1:])
