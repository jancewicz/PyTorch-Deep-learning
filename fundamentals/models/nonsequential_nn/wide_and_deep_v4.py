import torch
import torch.nn as nn

"""
Multiple output models - Type of neural network that at the end has not only one, but two or more outputs.
For example when specific task is divided into smaller ones: find an element in the picture and classify it, you will 
need two outputs for this - it's regression and classification subtasks.

Other use case is regularization. You may want to add auxiliary output in a neural network to ensure that underlying 
part of nn learns something useful on its own.
"""


class WideAndDeepV4(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.deep_stack = nn.Sequential(
            nn.Linear(n_features, 50),
            nn.ReLU(),
            nn.Linear(50, 40),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(40 + n_features, 1)
        self.aux_output_layer = nn.Linear(40, 1)

    def forward(self, X_wide, X_deep):
        deep_output = self.deep_stack(X_deep)
        wide_and_deep = torch.concat([X_wide, deep_output], dim=1)
        main_output = self.output_layer(wide_and_deep)
        aux_output = self.aux_output_layer(deep_output)

        return main_output, aux_output
