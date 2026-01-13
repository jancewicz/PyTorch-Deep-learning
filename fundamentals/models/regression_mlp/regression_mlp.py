import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset

from loguru import logger
from fundamentals.models.data import X_valid, y_valid, train_loader
from fundamentals.models.linear_regression.low_level_api.linear_regression import (
    n_features,
)
from fundamentals.models.training_and_evaluation.evaluate import NNEvaluator
from fundamentals.models.training_and_evaluation.training import NNTrainer
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

valid_dataset = TensorDataset(X_valid, y_valid)
valid_loader = DataLoader(valid_dataset, batch_size=32)

# Create rmse streaming metric, move it to the GPU
rmse_tm = torchmetrics.MeanSquaredError(squared=False).to(device)

# instantiate trainer and evaluator class
nn_trainer = NNTrainer(model, optimizer, mse, train_loader, device)
nn_evaluator = NNEvaluator(model, valid_loader, rmse_tm, device)

valid_mse = nn_evaluator.evaluate()


if __name__ == "__main__":
    nn_trainer.train(n_epochs)

    valid_mse = nn_evaluator.evaluate(
        aggregate_fn=lambda metrics: torch.sqrt(torch.mean(metrics)),
    )
    valid_tm = nn_evaluator.evaluate_tm(rmse_tm)

    logger.info(valid_tm)
