import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version used by PyTorch:", torch.version.cuda)
print("PyTorch version:", torch.__version__)

if torch.cuda.is_available():
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.current_device())
    print("GPU Name:", torch.cuda.get_device_name(0))