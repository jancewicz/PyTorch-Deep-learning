import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fundamentals.linear_regression.data import X_train, y_train
from fundamentals.linear_regression.low_level_api.linear_regression import n_features

"""
nn.Sequential is a module that chains multiple modules. It puts input into first module, then the output from the 
first module is passed as an input to second module and so on. It's like stacking modules one after another and passing
the outputs of one module as an input to the next one until neural network ends.

nn.Sequential is one of the most useful modules in PyTorch.
"""

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(n_features, 50),
    nn.ReLU(),
    nn.Linear(50, 40),
    nn.ReLU(),
    nn.Linear(40, 1)
)
model = model.to(device=device)

mse = nn.MSELoss()
learning_rate = 0.02
n_epochs = 20

# Some optimizers have internal state, and this state is allocated on the same device as the model
# Important to create the optimizer after model is moved to the GPU
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


def train(model, optimizer, criterion, train_loader, n_epochs):
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        mean_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} / {n_epochs}, Loss: {mean_loss: .4f}")


if __name__ == "__main__":
    train(model, optimizer, mse, train_loader, n_epochs)




