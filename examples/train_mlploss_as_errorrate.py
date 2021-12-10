from argparse import ArgumentParser
from typing import Tuple
from typing import Union
from isl.loss.mlp import MLPLossTrainer
from isl.trainer import mlploss_trainer
from isl.trainer import mlploss_validation
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LogitErrorDataset(Dataset):

    def __init__(self, n: int, n_class: int):
        self.data = []
        for _ in range(n):
            y_hat = torch.randn(n_class)
            y_hat = F.softmax(y_hat, dim=0)
            y_star = torch.randint(n_class, (1,))
            _, predicted = torch.max(y_hat, 0)
            error = (y_star != predicted).type(torch.float32)
            self.data.append((y_hat, y_star[0], error))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return self.data[idx]


def load_logit_error_data(n: int, n_class: int, batch_size: int = 100, use_gpu: bool = False) -> DataLoader:
    train = LogitErrorDataset(n, n_class)
    train = DataLoader(train, batch_size=batch_size, pin_memory=use_gpu)
    test = LogitErrorDataset(int(n * 0.2), n_class)
    test = DataLoader(test, batch_size=batch_size, pin_memory=use_gpu)
    return train, test


def make_model(n_class: int) -> nn.Module:
    return MLPLossTrainer(n_class)


def load_model(path: Union[str, Path],
               model: nn.Module,
               optimizer: optim.Optimizer) -> Tuple[nn.Module, optim.Optimizer, int]:
    path = Path(path)
    loaded = torch.load(path)
    model.load_state_dict(loaded['model_state_dict'])
    optimizer.load_state_dict(loaded['optimizer_state_dict'])
    return model, optimizer, loaded['epochs']


def save_model(path: Union[str, Path],
               model: nn.Module,
               optimizer: optim.Optimizer,
               epochs: int) -> None:
    path = Path(path)
    torch.save({
        'epochs': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)


def main(*argv) -> None:

    parser = ArgumentParser(
        description='ISL Example Script: Train LeNet5 with CIFAR')
    parser.add_argument('--cuda', action='store_true',
                        help='Use GPU if avaiable')
    parser.add_argument('--batch-size', type=int, default=100,
                        metavar='<int>', help='Set batch size (default=100)')
    parser.add_argument('-n', '--data-size', type=int, default=100000,
                        metavar='<int>', help='Numberof data to generate (default=100000)')
    parser.add_argument('-c', '--classes', type=int, default=10,
                        metavar='<int>', help='Number of classes (default=10)')
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

    train, test = load_logit_error_data(
        args.data_size, args.classes, args.batch_size, use_gpu)

    model = make_model(args.classes)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    current_epoch = 0

    if args.continual is not None:
        model, optimizer, current_epoch = load_model(
            args.continual, model, optimizer)
        print(f'Model loaded: {args.continual}')

    if use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()

    for _ in range(current_epoch, args.epochs):
        current_epoch += 1
        print(f'Epoch {current_epoch}')
        model, train_loss = mlploss_trainer(model, criterion, optimizer,
                                            train, verbose=1, use_gpu=use_gpu)
        val_loss = mlploss_validation(model, criterion, test, use_gpu=use_gpu)
        print(f'Train loss: {train_loss:.4f} Validation loss: {val_loss:.4f}')

        model_path = path / f'mlploss_{current_epoch}epochs.pt'
        save_model(model_path, model, optimizer, current_epoch)


if __name__ == '__main__':
    main(*sys.argv[1:])
