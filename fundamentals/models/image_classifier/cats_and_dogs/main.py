import os

import torch
import torch.nn as nn
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from fundamentals.models.image_classifier.cats_and_dogs.dataloaders import (
    STD_NORMALIZATION_VALUES,
    MEAN_NORMALIZATION_VALUES,
    split_and_transform_trainset,
    cats_dogs_training_set_dir,
)
from fundamentals.models.image_classifier.cats_and_dogs.predictors.cats_dogs_conv_nn import (
    ConvNNCatsDogsClassifier,
)
from fundamentals.training_and_evaluation.training import NNTrainer
from utils.device import get_device

CATS_DOGS_CHECKPOINTS_DIR = os.getenv("CATS_DOGS_CHECKPOINTS_DIR")

def visualize_batch(data_loader):
    # batch has shape (images, labes)
    batch = next(iter(data_loader))
    # images_batch has shape [4, 3, 224, 224]
    images_batch, labels_batch = batch[:4]

    # convert std and mean lists to tensor + apply broadcasting for new tensor to match shape
    std = torch.tensor(STD_NORMALIZATION_VALUES).view(1, 3, 1, 1)
    mean = torch.tensor(MEAN_NORMALIZATION_VALUES).view(1, 3, 1, 1)

    denormalized_imgs = (images_batch * std) + mean
    denormalized_imgs = torch.clamp(denormalized_imgs, 0, 1)

    # move height and width to the middle, channel goes last, batch position stays the same
    permuted_images = denormalized_imgs.permute(0, 2, 3, 1)
    class_names = data_loader.dataset.data_subset.dataset.classes

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    for i in range(4):
        axes[i].imshow(permuted_images[i].numpy())
        label_idx = labels_batch[i].item()
        axes[i].set_title(class_names[label_idx])
        axes[i].axis("off")
    plt.show()


if __name__ == "__main__":
    load_dotenv()

    train_set, valid_set = split_and_transform_trainset(cats_dogs_training_set_dir)

    train_set_dataloader = DataLoader(train_set, 32, shuffle=True)
    valid_set_dataloader = DataLoader(valid_set, 32, shuffle=True)

    visualize_batch(train_set_dataloader)
    hermes_img = os.getenv("HERMES_JPG_DIR")

    conv_nn = ConvNNCatsDogsClassifier().to(device=get_device())
    xentropy = nn.CrossEntropyLoss()
    learning_rate = 0.0001
    optimizer = torch.optim.AdamW(conv_nn.parameters(), lr=learning_rate)
    n_epochs = 40

    nn_trainer = NNTrainer(
        model=conv_nn,
        optimizer=optimizer,
        criterion=xentropy,
        train_loader=train_set_dataloader,
        device=get_device(),
    )

    nn_trainer.train(n_epochs=n_epochs)
    torch.save(conv_nn.state_dict(), f"{CATS_DOGS_CHECKPOINTS_DIR}/cats_dogs_alexNet_weights.pt")
