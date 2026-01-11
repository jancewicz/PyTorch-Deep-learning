import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

from fundamentals.models.data import X_train, y_train
from utils.device import get_device

"""
Sometimes model require different input types that cannot be combined easily into single tensor. For example, if we want
to feed neural net with image and text data, those inputs are going to have completely different number of dimensions.
"""
device = get_device()


class WideAndDeepV3(nn.Module):
    # Constructor stays the same
    def __init__(self, n_features):
        super().__init__()
        self.deep_stack = nn.Sequential(
            nn.Linear(n_features, 50), nn.ReLU(),
            nn.Linear(50, 40), nn.ReLU(),
        )
        self.output_layer = nn.Linear(40 + n_features, 1)

    # Provide wide and deep data separately if inputs differ
    def forward(self, X_wide, X_deep):
        deep_output = self.deep_stack(X_deep)
        wide_and_deep = torch.concat([X_wide, deep_output], dim=1)
        return self.output_layer(wide_and_deep)


# Create a dataset that returns the wide and deep inputs separately
train_data_wd = TensorDataset(X_train[:, :5], X_train[:, 2:], y_train)
train_loader_wd = DataLoader(train_data_wd, batch_size=32, shuffle=True)


# Data loader now returns three tensors instead of two: training loop needs to be adjusted
def train_with_unpacking(model, optimizer, criterion, data_loader, n_epochs):
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        # Since order of inputs matches the order of forward() method args we can unpack using * operator
        for *X_batch_inputs, y_batch in data_loader:
            X_batch_inputs = [X.to(device) for X in X_batch_inputs]
            y_batch = y_batch.to(device)
            y_pred = model(*X_batch_inputs)

            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        mean_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1} / {n_epochs}, Loss: {mean_loss: .4f}")


class WideAndDeepDataset(Dataset):
    """
    When model has many inputs, it's easy to mess up their order. The recommended solution is to name each input type.
    Creating custom dataset class can provide multiple options for it: ex. returning dictionary with input names.
    """
    def __init__(self, X_wide, X_deep, y):
        self.X_wide = X_wide
        self.X_deep = X_deep
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        input_dict = {
            "X_wide": self.X_wide[idx],
            "X_deep": self.X_deep[idx]
        }
        return input_dict, self.y[idx]


# Then create dataset and dataloader
train_data_named = WideAndDeepDataset(X_wide=X_train[:, :5], X_deep=X_train[:, 2:], y=y_train)
train_loader_named = DataLoader(train_data_named, batch_size=32, shuffle=True)


# Adjust training loop
def train_with_named_inputs(model, optimizer, criterion, data_loader_named, n_epochs):
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        # Since order of inputs matches the order of forward() method args we can unpack using * operator
        for inputs, y_batch in data_loader_named:
            inputs = {name: X.to(device) for name, X in inputs}
            y_batch = y_batch.to(device)
            y_pred = model(X_wide=inputs["X_wide"], X_deep=inputs["X_deep"])

            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        mean_loss = total_loss / len(data_loader_named)
        print(f"Epoch {epoch + 1} / {n_epochs}, Loss: {mean_loss: .4f}")
