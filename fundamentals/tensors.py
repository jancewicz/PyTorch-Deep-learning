import torch
import numpy as np

"""
Basic tensors operations with PyTorch
"""

# Create tensor
X = torch.tensor([[1.0, 4.0, 7.0], [2.0, 3.0, 6.0]])

# Get tensor shape and data type
X_shape, X_type = X.shape, X.dtype

# Math operations on tensors
10 * (X + 1.0)  # item-wise addition and multiplication

tensor_exp = X.exp()  # item-wise exponential
tensor_mean = X.mean()  # mean of tensor

# max values along dimension 0 - max value per column
tensor_max_vals_in_cols = X.max(dim=0)

# Matrix multiplication
res = X @ X.T

# Convert tensor to numpy array
np_arr = X.numpy()
# Crate tensor from numpy array
tensor_from_np = torch.tensor(np.array([[1.0, 4.0, 7.0], [2.0, 3.0, 6.0]]))


"""
Note: Basic precision point for PyTorch is 32bit where numpy use 64bits by default. It's generally better to use 
32bits in deep learning because it takes half the RAM and speeds up computations. Neural nets usually don't need 
extra precision that is given by 64bits floating points.

When calling torch.tensor() from numpy array, it's always good to specify dtype=torch.float32
"""

# Modify tensor in place using slicing
X[:, 1] = -99  # Change all values in column 1 to -99

# There are many methods that mutate tensor in place, like relu() which applies ReLU activation function and replacing
# all negative values with 0s.
X.relu_()

"""
Pytorch in place methods always ends with underscore - easier to spot
"""
