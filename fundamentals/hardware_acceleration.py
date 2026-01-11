import torch

"""
PyTorch can integrate with GPU, if the machine that is currently running program is GPU compatible.

Remember to run PyTorch on GPU locally, it's necessary to have all related libraries installed.
"""

if torch.cuda.is_available():
    device = "cuda"  # access nvidia GPU
elif torch.backends.mps.is_available():
    device = "mps"  # access apple MPS
else:
    device = "cpu"  # CPU fallback

# Create tensor on CPU
M = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# then copy it to the GPU
M = M.to(device)
# Lookup to on which device tensor lives
tensor_curr_device = M.device

# Alternatively it's possible to create tensor on GPU directly
M = torch.tensor([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]], device=device)

# If the tensor lives on GPU all the operations are will take place on the GPU
R = M @ M.T  # The result R also lives on the GPU
