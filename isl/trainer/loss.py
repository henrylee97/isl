from typing import Tuple
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim


def mlploss_trainer(loss_model: nn.Module,
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

    loss_model.train()
    criterion.train()

    for y_hat, y_star, error in dataset:
        if use_gpu:
            y_hat = y_hat.cuda()
            y_star = y_star.cuda()
            error.cuda()

        optimizer.zero_grad()
        pred = loss_model(y_hat, y_star)
        loss = criterion(pred, error)
        loss.backward()
        optimizer.step()

        total_loss.append(loss.cpu().item())

    total_loss = sum(total_loss) / len(total_loss)

    return loss_model, total_loss


def mlploss_validation(loss_model: nn.Module,
                       criterion: nn.Module,
                       dataset: torch.utils.data.DataLoader,
                       verbose: int = 0,
                       use_gpu: bool = False) -> Tuple[float, float]:

    if use_gpu:
        use_gpu = use_gpu and torch.cuda.is_available()

    if verbose > 0:
        dataset = tqdm.tqdm(dataset)

    total_loss = []

    loss_model.eval()
    criterion.eval()

    with torch.no_grad():
        for y_hat, y_star, error in dataset:
            if use_gpu:
                y_hat = y_hat.cuda()
                y_star = y_star.cuda()
                error = error.cuda()

            pred = loss_model(y_hat, y_star)
            loss = criterion(pred, error)

            total_loss.append(loss.cpu().item())

    total_loss = sum(total_loss) / len(total_loss)

    return total_loss
