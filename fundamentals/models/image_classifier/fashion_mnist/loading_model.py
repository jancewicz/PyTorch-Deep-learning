import torch

from fundamentals.models.image_classifier.fashion_mnist.image_classifier import ImageClassifier

"""
Loading model's weights from .pt file requires creating exact same model structure 
"""

if __name__ == "__main__":
    weights_path = "fundamentals/models/image_classifier/checkpoints/fashion_mnist_weights.pt"
    loaded_weights = torch.load(weights_path, weights_only=True)

    loaded_model = ImageClassifier(**loaded_weights["hyperparameters"])
    loaded_model.load_state_dict(loaded_weights["model_state_dict"])
    loaded_model.eval()
