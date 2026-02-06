import torch
if torch.cuda.is_available():
    print("Compute capability:", torch.cuda.get_device_capability(0))
