from typing import Tuple
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim


def simple_trainer(model: nn.Module,
                   criterion: nn.Module,
                   optimizer: optim.Optimizer,
                   dataset: torch.utils.data.DataLoader,
                   verbose: int = 0,
                   use_gpu: bool = False) -> Tuple[nn.Module, Tuple[float, float]]:

    if use_gpu:
        use_gpu = use_gpu and torch.cuda.is_available()

    if verbose > 0:
        dataset = tqdm.tqdm(dataset)

    total_loss = []
    total_acc = []

    model.train()
    criterion.train()

    for x, y_star in dataset:
        if use_gpu:
            x = x.cuda()
            y_star = y_star.cuda()

        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y_star)
        loss.backward()
        optimizer.step()

        total_loss.append(loss.cpu().item())

        _, y_hat = torch.max(y_hat, 1)
        acc = (y_hat == y_star).sum() / x.shape[0]
        total_acc.append(acc.cpu().item())

    total_loss = sum(total_loss) / len(total_loss)
    total_acc = sum(total_acc) / len(total_acc)

    return model, (total_loss, total_acc)


def simple_validation(model: nn.Module,
                      criterion: nn.Module,
                      dataset: torch.utils.data.DataLoader,
                      verbose: int = 0,
                      use_gpu: bool = False) -> Tuple[float, float]:

    if use_gpu:
        use_gpu = use_gpu and torch.cuda.is_available()

    if verbose > 0:
        dataset = tqdm.tqdm(dataset)

    total_loss = []
    total_acc = []

    model.eval()
    criterion.eval()

    with torch.no_grad():
        for x, y_star in dataset:
            if use_gpu:
                x = x.cuda()
                y_star = y_star.cuda()

            y_hat = model(x)
            loss = criterion(y_hat, y_star)

            total_loss.append(loss.cpu().item())

            _, y_hat = torch.max(y_hat, 1)
            acc = (y_hat == y_star).sum() / x.shape[0]
            total_acc.append(acc.cpu().item())

    total_loss = sum(total_loss) / len(total_loss)
    total_acc = sum(total_acc) / len(total_acc)

    return total_loss, total_acc
