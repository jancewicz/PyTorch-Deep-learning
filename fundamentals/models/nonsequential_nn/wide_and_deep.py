import torch
import torch.nn as nn

from fundamentals.models.data import X_train, train_loader
from fundamentals.models.regression_mlp.regression_mlp import train

"""
Wide and Deep neural network (Heng-Tze Cheng 2016) - it connects all or part of the inputs to the output layer.
It allows nn to learn deep patterns through deep path and set of rules through short path. The short path can be used
to provide manually engineered features to neural net.

In contrast, in simple MLP all the data has to go through full stack of layers - simple pattern can be distorted by this
sequence of transformations.
"""

device = "cpu"
n_features = X_train.shape[1]


# Because wide and deep architecture is nonsequential a custom module has to be made
class WideAndDeep(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.deep_stack = nn.Sequential(
            nn.Linear(n_features, 50), nn.ReLU(),
            nn.Linear(50, 40), nn.ReLU(),
        )
        self.output_layer = nn.Linear(40 + n_features, 1)

    def forward(self, X):
        deep_output = self.deep_stack(X)
        wide_and_deep = torch.concat([X, deep_output], dim=1)
        return self.output_layer(wide_and_deep)


torch.manual_seed(42)
model = WideAndDeep(n_features).to(device)
learning_rate = 0.002

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
mse = nn.MSELoss()
n_epochs = 30

if __name__ == "__main__":
    train(model, optimizer, mse, train_loader=train_loader, n_epochs=n_epochs)
