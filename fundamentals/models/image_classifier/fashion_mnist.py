import torch
import torchmetrics
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

from utils.device import get_device
from fundamentals.models.regression_mlp.regression_mlp import train, evaluate
from fundamentals.models.image_classifier.image_classifier import model, xentropy

device = get_device()

# data processing function
toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

# transform fashion MNIST from PIL to PyTorch float tensors with scaled pixel values
train_and_validate_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=True, download=True, transform=toTensor
)
test_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=False, download=True, transform=toTensor
)

torch.manual_seed(42)
train_data, valid_data = torch.utils.data.random_split(
    train_and_validate_data, [55_000, 5_000]
)

# create data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

learning_rate = 0.002


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
n_epochs = 20

train(model, optimizer, criterion=xentropy, train_loader=train_loader, n_epochs=n_epochs)

# evaluate model function
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
evaluation = evaluate(
    model, valid_loader, accuracy
)

print(f"Accuracy on validation set: {evaluation}")
