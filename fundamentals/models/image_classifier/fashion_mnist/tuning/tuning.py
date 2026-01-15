import torch
import optuna
import torchmetrics
from loguru import logger

from fundamentals.models.training_and_evaluation.evaluate import NNEvaluator
from utils.device import get_device

from fundamentals.models.training_and_evaluation.training import NNTrainer
from fundamentals.models.image_classifier.fashion_mnist.fashion_mnist import (
    train_loader,
    valid_loader,
)
from fundamentals.models.image_classifier.fashion_mnist.image_classifier import (
    ImageClassifier,
    xentropy,
)

device = get_device()


def objective(trial, train_loader, valid_loader):
    """
    Objective function for fine-tuning hyperparameters of neural network - learning rate and number of hidden layers.

    Optuna is going to call objective function many times to perform tuning and fine optimal values.
    Function must evaluate the model and return metric: in this case it's accuracy.
    """
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    n_hidden = trial.suggest_int("n_hidden", 20, 300)
    model = ImageClassifier(
        n_inputs=1 * 28 * 28, n_hidden1=n_hidden, n_hidden2=n_hidden, n_classes=10
    ).to(device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)

    n_epochs = 20
    valid_set_accuracy = 0.0

    nn_trainer = NNTrainer(model, optimizer, xentropy, train_loader, device)
    nn_evaluator = NNEvaluator(model, valid_loader, accuracy, device)
    model.train()

    for epoch in range(n_epochs):
        mean_loss = nn_trainer.single_training_loop()
        logger.info(f"Epoch {epoch + 1} / {n_epochs}, Loss: {mean_loss: .4f}")

        # evaluate model function .item() to convert the resulting 0-dim tensor to a float
        valid_set_accuracy = nn_evaluator.evaluate_tm(accuracy).item()

        trial.report(valid_set_accuracy, epoch)
        if trial.should_prune():
            logger.info(f"Trial {trial.number} pruned at epoch {epoch}")
            raise optuna.TrialPruned()

    return valid_set_accuracy


if __name__ == "__main__":
    torch.manual_seed(42)
    # Tree-structured Parzen Estimator for hyperparameters optimization
    sampler = optuna.samplers.TPESampler(seed=42)

    # Lambda to pass additional args for objective function
    objective_with_data = lambda trial: objective(
        trial, train_loader=train_loader, valid_loader=valid_loader
    )

    # prunner instance to detect if trial goes bad
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=0, interval_steps=1
    )

    # start hyperparameter tuning, due to accuracy as metric direction is: maximize
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective_with_data, n_trials=5)

    logger.info(f"Best params found during tuning: {study.best_params}")
    logger.info(f"Best value found: {study.best_value}")
