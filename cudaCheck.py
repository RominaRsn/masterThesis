import torch

if torch.cuda.is_available():
    print("CUDA is available.")
else:
    print("CUDA is not available.")


if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Use the first GPU
    tensor_on_gpu = torch.randn(3, 3).to(device)
    result = tensor_on_gpu * 2
    print(result)
else:
    print("No GPU available.")