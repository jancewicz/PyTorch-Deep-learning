import torch.nn as nn

"""
Using techniques like He initialization with activation function like ReLU can indeed reduce the risk of vanishing / 
exploding gradients it doesn't guarantee that this issue won't happen at some stage of the training.

The key problem: internal covariate shift: As the network learns, each layer's inputs keep changing because the previous
layers are also updating their weights. This makes training unstable and slow.

Batch normalization adds operation just before, or after the activations function of each hidden layer. This method 
zero-centers and normalizes each input, and then scales and shifts the result using new vectors for each layer.
First vector (γ) is used for scaling, second (β) is used for shifting. 

Long story short: It provides learnable, optimal scale and mean of each of the layers input for the model.

Batch norm provides a layer of regularization too, reducing needs for other forms of regularizers like dropout. It adds
more complexity to the model. The neural net predicts slower due to heavier computations at each layer. 
"""

model = nn.Sequential(
    nn.Flatten(),
    nn.BatchNorm1d(1 * 28 * 28),
    nn.Linear(1* 28 * 28, 300),
    nn.ReLU(),
    nn.BatchNorm1d(300),
    nn.Linear(300, 100),
    nn.ReLU(),
    nn.BatchNorm1d(100),
    nn.Linear(100, 10)
)

