from copy import deepcopy
from typing import Callable
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim


def spawn_child(criterion: nn.Module,
                stddiv: float = 0.1):
    child = deepcopy(criterion)
    with torch.no_grad():
        for param in criterion.parameters():
            param.add_(torch.randn_like(param) * stddiv)
    return child


def evaluate_candidate(criterion: nn.Module,
                       model: nn.Module,
                       optimizer: optim.Optimizer,
                       preferance: Callable[[torch.Tensor, torch.Tensor], float],
                       x: torch.Tensor,
                       y_star: torch.Tensor) -> float:
    # backup original model
    model_backup = deepcopy(model)
    optimizer_backup = deepcopy(optimizer)

    # one step training
    model.train()
    optimizer.zero_grad()
    y_hat = model(x)
    loss = criterion(y_hat, y_star)
    loss.backward()
    optimizer.step()

    # evaluate
    model.eval()
    with torch.no_grad():
        y_hat = model(x)
        score = preferance(y_hat, y_star)

    # restore model
    model.load_state_dict(model_backup.state_dict())
    optimizer.load_state_dict(optimizer_backup.state_dict())

    return score


def criterion_evolution(model: nn.Module,
                        optimizer: optim.Optimizer,
                        criterion: nn.Module,
                        preferance: Callable[[torch.Tensor, torch.Tensor], float],
                        x: torch.Tensor,
                        y_star: torch.Tensor,
                        n_candidates: int = 5,
                        stddiv: float = 0.1) -> nn.Module:

    childrens = [spawn_child(criterion, stddiv)
                 for _ in range(n_candidates - 1)]
    best_candidate = criterion
    best_preference = evaluate_candidate(
        best_candidate, model, optimizer, preferance, x, y_star)

    for child in childrens:
        child_preference = evaluate_candidate(
            child, model, optimizer, preferance, x, y_star)
        if child_preference > best_preference:
            best_preference = child_preference
            best_candidate = child

    return best_candidate


def swarm_trainer(model: nn.Module,
                  criterion: nn.Module,
                  optimizer: optim.Optimizer,
                  preferance: Callable[[torch.Tensor, torch.Tensor], float],
                  dataset: torch.utils.data.DataLoader,
                  valset: torch.utils.data.DataLoader,
                  n_candidates: int = 5,
                  stddiv: float = 0.1,
                  verbose: int = 0,
                  use_gpu: bool = False):

    if use_gpu:
        use_gpu = use_gpu and torch.cuda.is_available()

    if verbose > 0:
        dataset = tqdm.tqdm(dataset)

    iter_valset = iter(valset)

    model.eval()
    criterion.eval()

    total_loss = []
    total_acc = []

    for x, y_star in dataset:
        if use_gpu:
            x = x.cuda()
            y_star = y_star.cuda()

        try:
            valx, valy_star = next(iter_valset)
        except:
            iter_valset = iter(valset)
            valx, valy_star = next(iter_valset)
        if use_gpu:
            valx = valx.cuda()
            valy_star = valy_star.cuda()

        criterion = criterion_evolution(model, optimizer, criterion, preferance,
                                        valx, valy_star, n_candidates, stddiv)

        model.train()
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
