import torch.nn as nn


"""
Xawier and Glorot proposed that we need variance of output of each layer to be equal to the variance of its inputs,
same for the gradients - equal variance before and after going through a layer.

Equal number of inputs and outputs - fan-in, fan-out of the layer.
Eq:
sigmoid**2 = 1 / fan_avg
fan_avg = (fan_in + fan_out) / 2

Methodology:
- Initialize weights from Gaussian or uniform distribution
- Scale the weights with respect to the number of inputs of the layer.
Example: For the first hidden layer - fan in is the number of features in dataset, for the second layer it is the number 
of the units in the first hidden layer. 
"""


def apply_xawier_golrot_uniform(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform(module.weight)
        nn.init.zeros_(module.bias)


model = nn.Sequential(nn.Linear(50, 40), nn.ReLU(), nn.Linear(40, 1), nn.ReLU())
model.apply(apply_xawier_golrot_uniform)
