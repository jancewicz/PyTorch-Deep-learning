import torch
import torch.nn as nn

from utils.device import get_device

device = get_device()

class ImageClassifier(nn.Module):
    def __init__(self, n_inputs, n_hidden1, n_hidden2, n_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),  # reshape input to single dimension for linear layers
            nn.Linear(n_inputs, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_classes)
        )

    def forward(self, X):
        return self.mlp(X)


torch.manual_seed(42)
model = ImageClassifier(
    n_inputs=28 * 28,
    n_hidden1=300,
    n_hidden2=100,
    n_classes=10
)
model = model.to(device)

xentropy = nn.CrossEntropyLoss()
