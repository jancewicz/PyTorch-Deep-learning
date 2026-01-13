import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset

from loguru import logger
from fundamentals.models.data import X_valid, y_valid, train_loader
from fundamentals.models.linear_regression.low_level_api.linear_regression import (
    n_features,
)
from fundamentals.models.training_and_evaluation.training import SimpleNNTrainer
from utils.device import get_device

"""
nn.Sequential is a module that chains multiple modules. It puts input into first module, then the output from the 
first module is passed as an input to second module and so on. It's like stacking modules one after another and passing
the outputs of one module as an input to the next one until neural network ends.

nn.Sequential is one of the most useful modules in PyTorch.
"""

device = get_device()

torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(n_features, 50), nn.ReLU(), nn.Linear(50, 40), nn.ReLU(), nn.Linear(40, 1)
)
model = model.to(device=device)

mse = nn.MSELoss()
learning_rate = 0.02
n_epochs = 20

# Some optimizers have internal state, and this state is allocated on the same device as the model
# Important to create the optimizer after model is moved to the GPU
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# instantiate trainer class
nn_trainer = SimpleNNTrainer(model, optimizer, mse, train_loader, device)


def evaluate(model, data_loader, metric_fn, aggregate_fn=torch.mean):
    """
    Evaluates model and collect metrics within each batch.
    :param model: current model to evaluate
    :param data_loader: dataloader for currently used dataset
    :param metric_fn: function to compute metric for given batch
    :param aggregate_fn: function to aggregate the batch metrics, computes mean by default
    :return: aggregated model metrics
    """
    model.eval()
    metrics = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric = metric_fn(y_pred, y_batch)
            metrics.append(metric)

    return aggregate_fn(torch.stack(metrics))


# Optionally rmse can be computed as a metric method instead of mse
def rmse(y_pred, y_true):
    """
    Computes root mean squared error.
    :param y_pred: list of predictions made by model
    :param y_true: list of true labels
    :return: returns calculated rmse for current model
    """
    return ((y_pred - y_true) ** 2).mean().sqrt()


# Or created from torchmetrics library methods
# Create rmse streaming metric, move it to the GPU
rmse_tm = torchmetrics.MeanSquaredError(squared=False).to(device)


# Alternative method: use torchmetrics library to evaluate model
def evaluate_tm(model, data_loader, metric):
    model.eval()
    metric.reset()  # at the beginning reset metric
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)  # update the metric at each iteration

    # compute final result
    return metric.compute()


valid_dataset = TensorDataset(X_valid, y_valid)
valid_loader = DataLoader(valid_dataset, batch_size=32)
valid_mse = evaluate(model, valid_loader, mse)


if __name__ == "__main__":
    nn_trainer.train(n_epochs)

    valid_mse = evaluate(
        model,
        valid_loader,
        mse,
        aggregate_fn=lambda metrics: torch.sqrt(torch.mean(metrics)),
    )
    valid_tm = evaluate_tm(model, valid_loader, rmse_tm)

    logger.info(valid_tm)
