import torch, os
print("torch:", torch.__version__)
print("compiled cuda:", torch.version.cuda)
print("is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())