from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split
from scipy.signal import butter,filtfilt,iirnotch
import metrics
from keras.utils import plot_model
import os
import model
import spicy

model = model.simpleModel_modified2()
#model.load_weights(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\my_model_modified_simple.h5")



data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized.npy")
data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized.npy")




# Step 1: Split into training and test sets
noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized, data_clean_normalized, test_size=0.2, random_state=42)

# Step 2: Split the training set into training and validation sets
noisy_train, noisy_val, clean_train, clean_val = train_test_split(noisy_train, clean_train, test_size=0.1, random_state=42)



reshaped_data_noisy = noisy_train.reshape(np.shape(noisy_train)[0], np.shape(noisy_train)[1], 1)
reshaped_data_clean = clean_train.reshape(np.shape(noisy_train)[0], np.shape(noisy_train)[1], 1)

# sample1 = reshaped_data_noisy[0].transpose()
#

a = 100000
smaller_reshaped_data_clean = reshaped_data_clean[0:a]
smaller_reshaped_data_noisy = reshaped_data_noisy[0:a]
# # Desired new shape
# new_shape = (1, 504, 1)
#
# # Calculate the padding size
# padding = (new_shape[1] - smaller_reshaped_data_clean.shape[1], 0)
#
# # Pad the array with zeros
# padded_array = np.pad(smaller_reshaped_data_clean, pad_width=((0, 0), padding, (0, 0)), mode='constant')
#
# # Check the shape of the padded array
# print(padded_array.shape)  # It should be (1, 504, 1)
#
#



#smaller_reshaped_data_clean = reshaped_data_clean[0:a]
#print(smaller_reshaped_data_clean.shape)
#smaller_reshaped_data_clean_padded = np.concatenate([smaller_reshaped_data_clean, np.zeros(1,4, 1)])

#print(smaller_reshaped_data_clean_padded.shape)
#model.compile(optimizer='adam', loss='mean_squared_error')
model.optimizer.learning_rate = 1e-6
model.fit(
    smaller_reshaped_data_noisy,
    smaller_reshaped_data_clean,
    epochs=1,
    batch_size=32,
    validation_split=0.1,

)

x = np.arange(500)
result = model.predict(noisy_train[0, :].reshape(1, 500, 1))
# print(result)
# result =  result.reshape(500, 1)
cross_corr = spicy.signal.correlate(result, clean_train[0, :].reshape(1, 500, 1), mode='same')
#plt.plot(cross_corr.reshape(500,1))

coref = np.corrcoef(result.squeeze(axis= -1), clean_train[0, :])

print(coref)

sampling_freq = 250

#
# fft_result = np.fft.fft(np.fft.fft(clean_train[200, :].transpose()))
# frequencies = np.fft.fftfreq(len(fft_result))
# print(frequencies)
# magnitude = np.abs(fft_result)
#
# # Find the peaks in the FFT magnitude
# peaks, _ = spicy.signal.find_peaks(magnitude, height=0)  # You may need to adjust the height threshold
#
# # Find the index of the highest peak (excluding the DC component)
# fundamental_frequency_index = peaks[1] if len(peaks) > 1 else peaks[0]
#
# # Convert the index to the corresponding frequency
# fundamental_frequency = frequencies[fundamental_frequency_index]
#
# # Create a plot for visualization
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.plot(frequencies, magnitude)
# plt.plot(frequencies[fundamental_frequency_index], magnitude[fundamental_frequency_index], 'ro')
# plt.title('FFT with Fundamental Frequency')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
#
#
# t = np.linspace(0, 1, 500, endpoint=False)  # Time values
# # Plot the original signal
# plt.subplot(1, 2, 2)
# plt.plot(t, clean_train[200, :])
# plt.title('Original Signal')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
#
# plt.tight_layout()
# plt.show()
#
# print("Fundamental Frequency: {:.2f} Hz".format(abs(fundamental_frequency)))
# # print(fft_result)
# # # Plot the FFT magnitude
# # plt.subplot(1, 3, 1)
# # plt.plot(np.abs(fft_result))
# # plt.title('FFT Magnitude')
# # plt.xlabel('Frequency (Hz)')
# # plt.ylabel('Magnitude')
# #
# # # Plot the original signal
# # plt.subplot(1, 3, 2)
# # plt.plot(np.fft.fft(noisy_train[0, :]))
# # plt.title('Original Signal')
# # plt.xlabel('Time')
# # plt.ylabel('Amplitude')
# #
# #
# # plt.subplot(1, 3, 3)
# # plt.plot(np.fft.fft(clean_train[0, :]))
# # plt.title('Original Signal')
# # plt.xlabel('Time')
# # plt.ylabel('Amplitude')
# #
# # plt.tight_layout()
# # plt.show()
# #
# #
#
# # Calculate the FFT
# #fft_result = np.fft.fft(signal)
# frequencies = np.fft.fftfreq(len(fft_result))
#
# # Calculate the magnitude of the FFT components
# # Create a histogram of magnitude values
# # plt.hist(magnitude, bins=50, range=(0, max(magnitude)), edgecolor='black')
# # plt.title('FFT Magnitude Histogram')
# # plt.xlabel('Magnitude')
# # plt.ylabel('Frequency')
# #
# plt.show()
#
for i in range(0,5):
    fig, axes = plt.subplots(nrows=3, ncols=1)

    #row_index = 5
    row_index = np.random.randint(0, a)
    #col_index = np.random.randint(0, 11520000/500)

    axes[0].plot(clean_test[row_index, :], label = 'Clean Data')
    axes[0].set_title('Clean data')
    axes[0].set_ylabel('Signal amplitude')
    axes[0].set_xlabel('Time')

    print(smaller_reshaped_data_clean[row_index, :].shape)


    axes[1].plot(noisy_test[row_index, :], label = 'Noisy Data')
    axes[1].set_title('Noisy data')
    axes[1].set_ylabel('Signal amplitude')
    axes[1].set_xlabel('Time')

    result = model.predict(noisy_test[row_index, :].reshape(1, 500, 1))
    result = result.reshape(500, 1)
    #result = result.transpose()

    axes[2].plot(result/3, label='predicted data')
    axes[2].set_title('predicted data')
    axes[2].set_ylabel('Signal amplitude')
    axes[2].set_xlabel('Time')

    #test_array = np.array(noisy_dataF3[row_index, col_index : col_index + 500])
    #print(test_array.shape())

    # Add overall title
    fig.suptitle('Comparison of clean and noisy data')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


#
# result = model.predict(noisy_train[1, :].reshape(1, 500, 1))
# print(result)
# result =  result.reshape(500, 1)
# x = np.arange(500)
# sampling_freq = 250
# plt.plot(x, result)
# plt.show()
#
# print(result)

# sample1 = reshaped_data_noisy[0]
# print(sample1)
# print(sample1.shape)
#
# plt.plot(sample1)
# plt.show()
#
#
# sample1 = reshaped_data_clean[0]
# print(sample1)
# print(sample1.shape)
#
# plt.plot(sample1)
# plt.show()

#
# reshaped_data_noisy_validation = noisy_val.reshape(np.shape(clean_val)[0], np.shape(clean_val)[1], 1)
# reshaped_data_clean_validation = clean_val.reshape(np.shape(clean_val)[0], np.shape(clean_val)[1], 1)
#
# random_array = np.random.rand(1, 500, 1)
# result = model.predict(random_array)
# result = result.squeeze(axis=-1)
# sampling_freq = 250
# #y_axis = np.linspace(0, 2 * sampling_freq, 1)
#
# x = np.arange(500)
#
# random_array = random_array.squeeze(axis=0)
#
# plt.plot(x, random_array)
# plt.show()
#
# plt.plot(result)
# plt.show()
#
# print(model.weights)