import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

learning_rate = 1e-6
epochs = 1

# Define the simple model with pytorch
input_shape = (500, 1)
input_channels = 1
# Define your model

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(500, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(96, 500, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize the model with your input shape
model = MyModel()

zero_input = torch.zeros(1,500,1)
yero_output = model(zero_input)
print(yero_output.shape)
print(yero_output)
#max_norm_value = 6  # Your specified maximum norm value

# Create a custom class for weight-normalized Conv1D
# class WeightNormConv1D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size):
#         super(WeightNormConv1D, self).__init__()
#         self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size)
#         self.weight_norm = nn.utils.weight_norm(self.conv1d, name='weight', dim=None)
#
#     def forward(self, x):
#         x = self.weight_norm(x)
#         return x
#
# # Example usage
# model = nn.Sequential(
#     WeightNormConv1D(1, 128, kernel_size=3),
#     nn.ReLU(),
#     WeightNormConv1D(128, 96, kernel_size=3),
#     nn.ReLU(),
#     nn.ConvTranspose1d(96, 96, kernel_size=3),
#     nn.ReLU(),
#     nn.ConvTranspose1d(96, 96, kernel_size=3),
#     nn.ReLU(),
#     WeightNormConv1D(96, 1, kernel_size=3),
#     nn.Sigmoid()
# )

# Now, the convolutional layers in the model have weight normalization applied


# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters())
# #
data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized.npy")
data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized.npy")

# Step 1: Split into training and test sets
noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized, data_clean_normalized, test_size=0.2, random_state=42)

# Step 2: Split the training set into training and validation sets
noisy_train, noisy_val, clean_train, clean_val = train_test_split(noisy_train, clean_train, test_size=0.1, random_state=42)

noisy_train_reshaped = noisy_train.reshape(noisy_train.shape[0], 500, 1)
clean_train_reshaped = clean_train.reshape(clean_train.shape[0], 500, 1)

print(noisy_train_reshaped.shape)
print(clean_train_reshaped.shape)
noisy_train_tensor = torch.from_numpy(noisy_train_reshaped)
noisy_train_tensor = noisy_train_tensor.double()
out1=model(noisy_train_tensor[0,:,:])

print(out1.shape)

# # print("Training set size: ", noisy_train.shape)
# # print("Validation set size: ", noisy_val.shape)
# # print("Test set size: ", noisy_test.shape)
# #
# # #
# # # nosiy_train_t = np.transpose(noisy_train)
# # # clean_train_t = np.transpose(clean_train)
# # #
# # #
# # reshaped_data_noisy = noisy_train.reshape(input_shape[0], input_shape[1])
# # reshaped_data_clean = clean_train.reshape(input_shape[0], input_shape[1])
# # #
# # #
#
#
# sampling_freq = 250
# y_axis = np.linspace(0, 2 * sampling_freq)
# for i in range(0,5):
#     fig, axes = plt.subplots(nrows=2, ncols=1)
#
#
#     row_index = np.random.randint(0, len(noisy_train))
#
#     axes[0].plot(clean_train[row_index, :], label = 'Clean Data')
#     axes[0].set_title('Clean data')
#     axes[0].set_ylabel('Signal amplitude')
#     axes[0].set_xlabel('Time')
#
#
#     axes[1].plot(noisy_train[row_index, :], label = 'Noisy Data')
#     axes[1].set_title('Noisy data')
#     axes[1].set_ylabel('Signal amplitude')
#     axes[1].set_xlabel('Time')
#
#     #test_array = np.array(noisy_dataF3[row_index, col_index : col_index + 500])
#     #print(test_array.shape())
#
#     # Add overall title
#     fig.suptitle('Comparison of clean and noisy data')
#
#     # Adjust layout to prevent overlap
#     plt.tight_layout()
#
#     # Show the plot
#     #plt.show()
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using {} device".format(device))
# model.to('cuda')
#
# # Check if the model and its parameters are on CUDA
# is_model_on_cuda = next(model.parameters()).is_cuda
# print(f"Model on CUDA: {is_model_on_cuda}")
#
# for name, param in model.named_parameters():
#     print(f"Parameter: {name}, Data Type: {param.dtype}")
#
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)



# ...
# print(noisy_train.shape)
# for epoch in range(epochs):
#     model.train()
#     running_loss = 0.0
#     for i in range(len(noisy_train)):
#         inputs = torch.from_numpy(noisy_train[i, :]).to(device)
#         labels = torch.from_numpy(clean_train[i, :]).to(device)
#         inputs = inputs.to(torch.float32)  # Convert input to torch.float32
#         labels = labels.to(torch.float32)
#         optimizer.zero_grad()
#         outputs = model(inputs.transpose)
#         outputs = outputs.to(device)
#         # outputs = outputs.to(torch.float32)
#         loss_fn = torch.nn.functional.mse_loss
#         # print("the shapes are: ")
#         # print(inputs.shape)
#         # print(labels.shape)
#         print(outputs)
#         #
#         loss = loss_fn(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     print("Epoch {} - Training loss: {}".format(epoch, running_loss / len(noisy_train)))




#
# # Training loop
# for epoch in range(1):
#     model.train()  # Set the model to training mode
#     running_loss = 0.0
#
#     for inputs, labels in zip(nosiy_train_t,clean_train_t) :
#         inputs = inputs.reshape(1, 1, -1)
#         inputs = torch.from_numpy(inputs).to(device)
#
#         labels = labels.reshape(1, 1, -1)
#         labels = torch.from_numpy(labels).to(device)
#
#         inputs = inputs.to(torch.float32)  # Convert input to torch.float32
#         labels = labels.to(torch.float32)
#
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#
#     print(f"Epoch {epoch+1}, Loss: {running_loss / len(reshaped_data_clean)}")
