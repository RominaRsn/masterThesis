# import torch
#
# if torch.cuda.is_available():
#     print("CUDA is available.")
# else:
#     print("CUDA is not available.")
#
#
# if torch.cuda.is_available():
#     device = torch.device("cuda:0")  # Use the first GPU
#     tensor_on_gpu = torch.randn(3, 3).to(device)
#     result = tensor_on_gpu * 2
#     print(result)
# else:
#     print("No GPU available.")


#
import torch
import torch.nn as nn

# Define a 1D convolutional layer with instance normalization
conv_layer = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
instance_norm = nn.InstanceNorm1d(32)  # Normalize along the channel dimension (32 channels)

# Generate some random 1D data (e.g., time series)
data = torch.randn(1, 1, 10)  # Batch size of 1, 1 channel, 10 time steps

# Apply the convolution followed by instance normalization
output = conv_layer(data)
output = instance_norm(output)


print(data)
print(output)