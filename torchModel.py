import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

max_norm_value = 6  # Your specified maximum norm value

# Create a custom class for weight-normalized Conv1D
class WeightNormConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(WeightNormConv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.weight_norm = nn.utils.weight_norm(self.conv1d, name='weight', dim=None)

    def forward(self, x):
        x = self.weight_norm(x)
        return x

# Example usage
model = nn.Sequential(
    WeightNormConv1D(1, 128, kernel_size=3),
    nn.ReLU(),
    WeightNormConv1D(128, 96, kernel_size=3),
    nn.ReLU(),
    nn.ConvTranspose1d(96, 96, kernel_size=3),
    nn.ReLU(),
    nn.ConvTranspose1d(96, 96, kernel_size=3),
    nn.ReLU(),
    WeightNormConv1D(96, 1, kernel_size=3),
    nn.Sigmoid()
)

# Now, the convolutional layers in the model have weight normalization applied
device = torch.device("cuda")

model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized.npy")
data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized.npy")

# Step 1: Split into training and test sets
noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized, data_clean_normalized, test_size=0.2, random_state=42)

# Step 2: Split the training set into training and validation sets
noisy_train, noisy_val, clean_train, clean_val = train_test_split(noisy_train, clean_train, test_size=0.1, random_state=42)

nosiy_train_t = np.transpose(noisy_train)
clean_train_t = np.transpose(clean_train)


reshaped_data_noisy = nosiy_train_t.reshape(np.shape(noisy_train)[0], np.shape(noisy_train)[1], 1)
reshaped_data_clean = clean_train_t.reshape(np.shape(noisy_train)[0], np.shape(noisy_train)[1], 1)


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Training loop
for epoch in range(1):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for inputs, labels in zip(nosiy_train_t,clean_train_t) :
        inputs = inputs.reshape(1, 1, -1)
        inputs = torch.from_numpy(inputs).to(device)

        labels = labels.reshape(1, 1, -1)
        labels = torch.from_numpy(labels).to(device)

        inputs = inputs.to(torch.float32)  # Convert input to torch.float32
        labels = labels.to(torch.float32)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(reshaped_data_clean)}")
