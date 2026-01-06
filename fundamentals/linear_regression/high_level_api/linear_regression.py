import torch
import torch.nn as nn

from fundamentals.linear_regression.low_level_api.linear_regression import n_features

"""
Creating simple linear regression model with higher level pytorch api utilities.
This is second part, displaying powerful features of PyTorch.
"""

torch.manual_seed(42)
model = nn.Linear(in_features=n_features, out_features=1)


