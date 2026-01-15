import torch

from torch.utils.data import random_split
from torchvision import datasets
from torchvision.transforms import v2 as transforms_v2

from fundamentals.models.image_classifier.cats_and_dogs.image_processing.transform_subset import (
    SubsetTransformator,
)

# get both train set and test set dirs
cats_dogs_training_set_dir = "datasets/cats_and_dogs/training_set"
cats_dogs_test_set_dir = "datasets/cats_and_dogs/test_set"
SIZE = (224, 224)

torch.manual_seed(42)


def split_and_transform_trainset(cats_dogs_training_set: str):
    generator = torch.Generator().manual_seed(42)

    common_transforms = [
        transforms_v2.ToImage(),
        transforms_v2.Resize(size=SIZE, antialias=True),
    ]

    # create image processing pipeline for train set with augmentation and valid set with just resizing photos
    train_set_transform = transforms_v2.Compose(
        [
            *common_transforms,
            transforms_v2.RandomHorizontalFlip(0.5),
            transforms_v2.ToDtype(torch.float32, scale=True),
            # values from PyTorch docs
            transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    valid_set_transform = transforms_v2.Compose(
        [
            *common_transforms,
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    # split training set with 8/2 ratio
    raw_train_dataset = datasets.ImageFolder(cats_dogs_training_set)
    train_set_size = int(0.8 * len(raw_train_dataset))
    valid_set_size = len(raw_train_dataset) - train_set_size

    train_set, valid_set = random_split(
        dataset=raw_train_dataset,
        lengths=[train_set_size, valid_set_size],
        generator=generator,
    )

    train_set_transformed = SubsetTransformator(
        data_subset=train_set, transform=train_set_transform
    )
    valid_set_transformed = SubsetTransformator(
        data_subset=valid_set, transform=valid_set_transform
    )
    return train_set_transformed, valid_set_transformed
