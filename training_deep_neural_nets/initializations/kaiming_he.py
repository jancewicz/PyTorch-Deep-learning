import torch
import torch.nn as nn


"""
nn.Linear module's weights are being initialized  using Kaiming uniform initialization, except the fact that the weights
are being scaled down by √6. 
But this is not an optimal scale for modern activation functions.

For proper Kaiming / He initialization just multiply nn.Linear weights by √6 right after module initialization. 
"""
layer = nn.Linear(40, 10)
layer.weight.data *= (
    6**0.5
)  # He init (Also possible to do LeCun initialization if we use 3**0.5)
torch.zero_(layer.bias.data)

"""
The cleanest way to apply corrected initialization to nn.Linear layers of nn is to write simple function that will 
provide such logic for each nn.Linear module.
"""


def use_he_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight)
        nn.init.zeros_(module.bias)


# apply He init
model: nn.Module = nn.Sequential(
    nn.Linear(50, 40), nn.ReLU(), nn.Linear(40, 1), nn.ReLU()
)
model.apply(use_he_init)
