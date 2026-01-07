import torch
import torch.nn as nn

from fundamentals.linear_regression.data import X_train, y_train
from fundamentals.linear_regression.low_level_api.linear_regression import n_features
from fundamentals.linear_regression.high_level_api.linear_regression import train_bdg

"""
nn.Sequential is a module that chains multiple modules. It puts input into first module, then the output from the 
first module is passed as an input to second module and so on. It's like stacking modules one after another and passing
the outputs of one module as an input to the next one until neural network ends.

nn.Sequential is one of the most useful modules in PyTorch.
"""

torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(n_features, 50),
    nn.ReLU(),
    nn.Linear(50, 40),
    nn.ReLU(),
    nn.Linear(40, 1)
)

learning_rate = 0.1
n_epochs = 20
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
mse = nn.MSELoss()

train_bdg(model, optimizer, mse, X_train=X_train, y_train=y_train, n_epochs=n_epochs)







