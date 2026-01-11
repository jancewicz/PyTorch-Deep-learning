import torch


def get_device():
    """
    Get GPU device if is available otherwise CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")