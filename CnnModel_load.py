from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split
from scipy.signal import butter,filtfilt,iirnotch
from masterThesis.metrics import metrics
from keras.utils import plot_model
import os

# clean_data = np.load(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\data_file\clean_0.npy')
# noisy_data = np.load(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\data_file\noisy_0.npy')
#
# print(np.max(clean_data))
# print(np.max(noisy_data))

# file_path_clean = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\data_file\clean_0.npy'
# file_path_noisy = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\data_file\noisy_0.npy'

data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized.npy")
data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized.npy")




# Step 1: Split into training and test sets
noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized, data_clean_normalized, test_size=0.2, random_state=42)

# Step 2: Split the training set into training and validation sets
noisy_train, noisy_val, clean_train, clean_val = train_test_split(noisy_train, clean_train, test_size=0.1, random_state=42)

# Now, you have:
# - X_train, y_train: Training set
# - X_val, y_val: Validation set
# - X_test, y_test: Test set

# You can print the shapes to check the sizes of the sets
# print("Training set shape:", noisy_train.shape)
# print("Validation set shape:", noisy_val.shape)
# print("Test set shape:", noisy_test.shape)

# reshaped_data_noisy = nosiy_train_t.reshape(np.shape(noisy_train)[0], np.shape(noisy_train)[1], 1)
# reshaped_data_clean = clean_train_t.reshape(np.shape(noisy_train)[0], np.shape(noisy_train)[1], 1)

reshaped_noisy_test = noisy_test.reshape(noisy_test.shape[0], noisy_test.shape[1], 1)
model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\my_model_Cnn.h5")

# Plot the model and save it to a file (optional)
##plot test_prediction with time from 0 to 500 ms


x = np.linspace(0, 500, 500)
plot_model(model, to_file='model_plot_CnnModel.png', show_shapes=True, show_layer_names=True)
test_input = np.zeros((1, 500))
test_prediction = model.predict(test_input)
print(test_prediction)

test_input = np.ones((1, 500))
test_prediction1 = model.predict(test_input)
print(test_prediction)



fig, axes = plt.subplots(nrows=2, ncols=1)

#col_index = np.random.randint(0, 11520000/500)

axes[0].plot(test_prediction[0, :], label = 'zero Data')
axes[0].set_title('zero data')
axes[0].set_ylabel('Signal amplitude')
axes[0].set_xlabel('Time')


axes[1].plot(test_prediction1[0, :], label = 'one Data')
axes[1].set_title('one data')
axes[1].set_ylabel('Signal amplitude')
axes[1].set_xlabel('Time')

plt.tight_layout()

# Show the plot
plt.show()

# result = model.predict(reshaped_noisy_test[0:2039, :, :]);
# print(result.shape)
# print(result)
# #result = result.squeeze(axis=-1)
#
#
# filteredSignal = metrics.filtering_signals(noisy_test, 250, 45, 0.5, 50, 4)
#
# sampling_freq = 250
# y_axis = np.linspace(0, 2 * sampling_freq)
# for i in range(10,15):
#     fig, axes = plt.subplots(nrows=4, ncols=1)
#
#
#     row_index = i
#     #col_index = np.random.randint(0, 11520000/500)
#
#     axes[0].plot(noisy_test[row_index, :], label = 'Noisy Data')
#     axes[0].set_title('Noisy data')
#     axes[0].set_ylabel('Signal amplitude')
#     axes[0].set_xlabel('Time')
#
#
#     axes[1].plot(clean_test[row_index, :], label = 'clean Data')
#     axes[1].set_title('clean data')
#     axes[1].set_ylabel('Signal amplitude')
#     axes[1].set_xlabel('Time')
#
#     axes[2].plot(result[row_index, :], label = 'clean Data_ predicted')
#     axes[2].set_title('cleaned data')
#     axes[2].set_ylabel('Signal amplitude')
#     axes[2].set_xlabel('Time')
#
#
#     axes[3].plot(filteredSignal[row_index, :], label = 'filtered signal')
#     axes[3].set_title('filtered signal')
#     axes[3].set_ylabel('Signal amplitude')
#     axes[3].set_xlabel('Time')
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
#     plt.show()
#
# # Get the user's home directory
# user_home = os.path.expanduser("~")
#
# # Specify the file path in the Downloads directory
# file_path = os.path.join(user_home, "Downloads", "your_file.txt")
#
#
# #Calculating the snr
# for i in range(0,2039):
#     a = metrics.snr(result[i, :], clean_test[i, :])
#
#     # Open the file for appending ('a' mode)
#     with open(file_path, 'a') as fm:
#         fm.write("SNRNoisy: %f\n" % a)
#         # Write other data if needed
