import torch
import torchmetrics
import torchvision
import torchvision.transforms.v2 as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from loguru import logger


from fundamentals.models.training_and_evaluation.evaluate import NNEvaluator
from fundamentals.models.training_and_evaluation.training import NNTrainer
from utils.device import get_device
from fundamentals.models.image_classifier.fashion_mnist.image_classifier import model, xentropy

device = get_device()

# data processing function
toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

# transform fashion MNIST from PIL to PyTorch float tensors with scaled pixel values
train_and_valid_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=True, download=True, transform=toTensor
)
test_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=False, download=True, transform=toTensor
)

torch.manual_seed(42)
train_data, valid_data = torch.utils.data.random_split(
    train_and_valid_data, [55_000, 5_000]
)

# create data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

learning_rate = 0.002
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
n_epochs = 20

# evaluate model function
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)

# instantiate trainer and evaluator for neural network
nn_trainer = NNTrainer(model, optimizer, xentropy, train_loader, device)
nn_evaluator = NNEvaluator(model, valid_loader, accuracy, device)

if __name__ == "__main__":
    nn_trainer.train(n_epochs)
    evaluation = nn_evaluator.evaluate()

    logger.info(f"Accuracy on validation set: {evaluation}")
    model.eval()
    X_new, y_new = next(iter(valid_loader))
    X_new = X_new[:3].to(device)
    with torch.no_grad():
        y_pred_logits = model(X_new)
    y_pred = y_pred_logits.argmax(dim=1)  # get index of the largest logit
    logger.info(f"Predicted label: {y_pred}")

    # Compute softmax of logits manually and display values for first 3 records from validation set
    y_proba = F.softmax(y_pred_logits, dim=1)
    logger.info(
        f"Softmax from all values for classes probabilities: {y_proba.round(decimals=3)}"
    )

    # Get top k predictions of the model
    y_top4_logits, y_top4_indices = torch.topk(y_pred_logits, k=4, dim=1)
    y_top4_probas = F.softmax(y_top4_logits, dim=1)

    logger.info(f"Top 4 class probabilities ${y_top4_probas.round(decimals=3)}")
    logger.info(f"Top 4 indies: {y_top4_indices}")

    # save model data inside a dict
    model_data = {
        "model_state_dict": model.state_dict(),
        "hyperparameters": {
            "n_inputs": 1*28*28,
            "n_hidden1": 300,
            "n_hidden2": 100,
            "n_classes": 10
        }
    }
    torch.save(model_data, "checkpoints/fashion_mnist_weights.pt")
