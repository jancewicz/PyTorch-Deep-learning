import torch
import torch.nn as nn

"""
What if we want to send subset of the features through wide path and other part (possibly overlapping) through the deep
path? One solution is to split the inputs inside forward() method
"""


class WideAndDeepV2(nn.Module):
    # Constructor stays the same
    def __init__(self, n_features):
        super().__init__()
        self.deep_stack = nn.Sequential(
            nn.Linear(n_features, 50), nn.ReLU(),
            nn.Linear(50, 40), nn.ReLU(),
        )
        self.output_layer = nn.Linear(40 + n_features, 1)

    def forward(self, X):
        # Define which slices of same data should go through wide path and deep path
        X_wide = X[:, :5]
        X_deep = X[:, 2:]
        deep_output = self.deep_stack(X_deep)
        wide_and_deep = torch.concat([X_wide, deep_output], dim=1)
        return self.output_layer(wide_and_deep)


"""
Note: This works fine, but usally it's better to let the model take two separate tensors as input.
"""
