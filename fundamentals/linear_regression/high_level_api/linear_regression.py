import torch
import torch.nn as nn

from fundamentals.linear_regression.data import X_train, y_train, X_test
from fundamentals.linear_regression.low_level_api.linear_regression import n_features

"""
Creating simple linear regression model with higher level pytorch api utilities.
This is second part, displaying powerful features of PyTorch.
"""

torch.manual_seed(42)
model = nn.Linear(in_features=n_features, out_features=1)

n_epochs = 20
learning_rate = 0.4

# Create optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
mse = nn.MSELoss()


def train_bdg(model, optimizer, criterion, X_train, y_train, n_epochs):
    for epoch in range(n_epochs):
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)  # in PyTorch loss function object is commonly called criterion
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}")


if __name__ == "__main__":
    train_bdg(model, optimizer, mse, X_train=X_train, y_train=y_train, n_epochs=n_epochs)

    # With trained model it's ready to predict unseen data
    X_new = X_test[:3]
    with torch.no_grad():
        y_pred = model(X_new)

    print(y_pred)
