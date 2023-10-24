import mne
import graphviz
import time
from keras.models import load_model
import keras
import model
import tensorflow
from sklearn.model_selection import train_test_split
from metrics import metrics
from keras.models import Sequential, save_model
from keras.layers import Conv1D, Conv1DTranspose, MaxPooling1D
from keras.constraints import max_norm
from scipy.signal import butter,filtfilt,iirnotch
from keras.utils import plot_model
#from numba import jit, cuda

import numpy as np
import matplotlib.pyplot as plt
#
# sample_data_folder = mne.datasets.sample.data_path()
# sample_data_raw_file = (
#     sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
# )
# raw = mne.io.read_raw_fif(sample_data_raw_file)
#
# print(raw.info)
#
# EEG_Clean1 = mne.io.read_raw_brainvision("D:\data\denoising\datafortraining\clean\subclinical182_ch3.vhdr");
# signals = mne.ioraw_data = EEG_Clean1.get_data()
# sampling_freq = 250
#
# print(signals[1])
# plt.plot(signals[1])
# #plt.show()
#
# def loadclean():
#     EEG_Clean1 = mne.io.read_raw_brainvision("D:\data\denoising\datafortraining\clean\subclinical182_ch1.vhdr")
#     EEG_Clean2 = mne.io.read_raw_brainvision("D:\data\denoising\datafortraining\clean\subclinical182_ch2.vhdr")
#     EEG_Clean3 = mne.io.read_raw_brainvision("D:\data\denoising\datafortraining\clean\subclinical182_ch3.vhdr")
#     signals = mne.ioraw_data = EEG_Clean1.get_data()
#     clean_data2 = mne.ioraw_data = EEG_Clean2.get_data()
#     signals = np.append(signals, clean_data2, axis=1)
#     clean_data2 = mne.ioraw_data = EEG_Clean3.get_data()
#     signals = np.append(signals, clean_data2, axis=1)
#     sampling_freq = 250
#     return (signals, sampling_freq)
#
#
# def loadnoisy():
#     EEG_NoisyF1 = mne.io.read_raw_brainvision(
#         r"D:\data\denoising\datafortraining\noisy_f3\subclinical182_newnoisescaledfhp30iir_3_1.vhdr")
#     EEG_NoisyF2 = mne.io.read_raw_brainvision(
#         r"D:\data\denoising\datafortraining\noisy_f3\subclinical182_newnoisescaledfhp30iir_3_2.vhdr")
#     EEG_NoisyF3 = mne.io.read_raw_brainvision(
#         r"D:\data\denoising\datafortraining\noisy_f3\subclinical182_newnoisescaledfhp30iir_3_3.vhdr")
#
#     noisy_signals = mne.ioraw_data = EEG_NoisyF1.get_data()
#     noisy_dataF4 = mne.ioraw_data = EEG_NoisyF2.get_data()
#     noisy_signals = np.append(noisy_signals, noisy_dataF4, axis=1)
#
#     noisy_dataF4 = mne.ioraw_data = EEG_NoisyF3.get_data()
#     noisy_signals = np.append(noisy_signals, noisy_dataF4, axis=1)
#     sampling_freq = 250
#     return (noisy_signals, sampling_freq)
#

# Model configuration
#input_shape = (500, 1)
batch_size = 32
#no_epochs = 30
no_epochs = 1


#train_test_split = 0.4
validation_split = 0.1
verbosity = 1
max_norm_value = 6.0

#
# start_time = time.time()
# #load EEG
# clean_EEG = loadclean()
# #clean_EEG = loadsmallclean()
# clean_data = clean_EEG[0]
# sampling_freq = clean_EEG[1]
#
# #noisy_data = loadnoisy()
# noisy_data = loadnoisy()
# #noisy_data = loadsmallnoisy()
#
# noisy_dataF3 = noisy_data[0]
#
#
# end_time = time.time()
# elapsed_time = end_time - start_time
#
# print(f"Elapsed time 1: {elapsed_time} seconds")
# num_chanClean, num_SamplesClean = clean_data.shape
# nChan = num_chanClean
# num_chanNoisy, num_SamplesNoisy = noisy_dataF3.shape
#
# val_clean = []
# val_noisy = []
#
# start_time = time.time()
#
# for i in range(0, 180):
#     for j in range(0, 23040):
#         val_clean.append(clean_data[i, j * 2 * sampling_freq : (j + 1) * 2 * sampling_freq])
#         val_noisy.append(noisy_dataF3[i, j * 2 * sampling_freq : (j + 1) * 2 * sampling_freq])
#
#
# data_cleaned = np.array(val_clean)
# data_noisy = np.array(val_noisy)
# # Specify the file path
# file_path_clean = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\data_file\clean_0.npy'
# file_path_noisy = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\data_file\noisy_0.npy'
# # Save the matrix to a NumPy binary file
# np.save(file_path_clean, data_cleaned)
# np.save(file_path_noisy, data_noisy)
#
#
# print(data_cleaned.shape)
# print(data_noisy.shape)
#
#
# end_time = time.time()
# elapsed_time = end_time - start_time
#
# print(f"Elapsed time: {elapsed_time} seconds")

metrics = metrics()

#Loading normalized data

start_time = time.time()

data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized.npy")
data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized.npy")


end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")




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

# nosiy_train_t = np.transpose(noisy_train)
# clean_train_t = np.transpose(clean_train)


reshaped_data_noisy = noisy_train.reshape(np.shape(noisy_train)[0], np.shape(noisy_train)[1], 1)
reshaped_data_clean = clean_train.reshape(np.shape(noisy_train)[0], np.shape(noisy_train)[1], 1)

reshaped_data_noisy_validation = noisy_val.reshape(np.shape(clean_val)[0], np.shape(clean_val)[1], 1)
reshaped_data_clean_validation = clean_val.reshape(np.shape(clean_val)[0], np.shape(clean_val)[1], 1)

print(reshaped_data_noisy.shape)
print(reshaped_data_clean.shape)

# start_time = time.time()
#
# snr_matrix = metrics.snr(noisy_train, clean_train)
#
# print(snr_matrix)
# end_time = time.time()
# elapsed_time = end_time - start_time
#
# print(f"snr calculation: {elapsed_time} seconds")



#
# sampling_freq = 250
# y_axis = np.linspace(0, 2 * sampling_freq)
# for i in range(0,5):
#     fig, axes = plt.subplots(nrows=2, ncols=1)
#
#
#     row_index = np.random.randint(0, len(data_clean_normalized))
#     axes[0].plot(data_clean_normalized[row_index], label = 'Clean Data')
#     axes[0].set_title('Clean data')
#     axes[0].set_ylabel('Signal amplitude')
#     axes[0].set_xlabel('Time')
#
#
#     axes[1].plot(data_noisy_normalized[row_index], label = 'Noisy Data')
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

    # Show the plot
    #plt.show()

#
# #Encoder model
nLatentNeurons = 96
input_shape = (500, 1)
with tensorflow.device('/device:GPU:0'):
    #model = model.Novel_CNN((500,1))
    model = Sequential()
    model.add(Conv1D(128, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
                     kernel_initializer='he_uniform', input_shape=input_shape))
    model.add(Conv1D(nLatentNeurons, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
                     kernel_initializer='he_uniform'))
    model.add(
        Conv1DTranspose(nLatentNeurons, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
                        kernel_initializer='he_uniform'))
    model.add(
        Conv1DTranspose(nLatentNeurons, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
                        kernel_initializer='he_uniform'))
    model.add(
        Conv1D(1, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='sigmoid', padding='same'))

    model.summary()

    #Print model summary to see the architecture
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    #
    model.compile(optimizer='adam', loss='mean_squared_error')
    # model.save("C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis")
    #
    # x_train_1 = np.transpose(noisy_train)[:,0:1000]
    # print(reshaped_data_noisy.shape)
    # print(reshaped_noisy_train.shape)
    #
    #
    # start_time = time.time()
    history = model.fit(
        reshaped_data_noisy,
        reshaped_data_clean,
        epochs=1,
        batch_size=64,
        validation_data=(reshaped_data_noisy_validation,reshaped_data_clean_validation)
    )
    model.save('my_model_simple.h5')

    # Access training and validation loss from model.history
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Create a plot of training and validation loss
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    # # Record the end time
    # end_time = time.time()
    #
    # # Calculate and print the elapsed time
    # elapsed_time = end_time - start_time
    # print(f"Elapsed Time: {elapsed_time} seconds")
    # model.save('my_model.h5')

#
#     nLatentNeurons = 96
#
#     input_shape = (500, 1)
#
#     model = Sequential()
#     model.add(Conv1D(128, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
#                      kernel_initializer='he_uniform', input_shape=input_shape))
#     model.add(Conv1D(nLatentNeurons, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
#                      kernel_initializer='he_uniform'))
#     model.add(Conv1DTranspose(nLatentNeurons, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
#                               kernel_initializer='he_uniform'))
#     model.add(Conv1DTranspose(nLatentNeurons, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='relu',
#                               kernel_initializer='he_uniform'))
#     model.add(Conv1D(1, kernel_size=3, kernel_constraint=max_norm(max_norm_value), activation='sigmoid', padding='same'))
#
#     model.summary()
#
#     # Note: You can customize the number of filters, kernel size, activation function, etc., based on your requirements
#
#     # Print model summary to see the architecture
#     plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     #model.save("C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis")
#
#     # x_train_1 = np.transpose(noisy_train)[:,0:1000]
#     # print(reshaped_data_noisy.shape)
#     # print(reshaped_noisy_train.shape)
#     #
#     #
#     start_time = time.time()
#     model.fit(
#         reshaped_data_noisy,
#         reshaped_data_clean,
#         epochs=no_epochs,
#         batch_size=32,
#         validation_split=validation_split
#     )
#
#     # Record the end time
#     end_time = time.time()
#
#     # Calculate and print the elapsed time
#     elapsed_time = end_time - start_time
#     print(f"Elapsed Time: {elapsed_time} seconds")
#     model.save('my_model.h5')

#model = load_model(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\my_model.h5")

# Plot the model and save it to a file (optional)
#plot_model(model, to_file='model_plot1.png', show_shapes=True, show_layer_names=True)


# print("shape")
# print(noisy_test.shape)
# a = noisy_test[100, :].reshape(1, 500, 1)
# ab = model.predict(a)
#
#
# #Plotting
# plt.figure(figsize=(10, 5))
#
# # Plot input data
# plt.subplot(2, 1, 1)
# plt.plot(a.flatten(), label='Input Data')
# plt.title('Input Data')
# plt.legend()
#
# # Plot predictions
# plt.subplot(2, 1, 2)
# plt.plot(ab.flatten(), label='Predictions')
# plt.title('Predictions')
# plt.legend()
#
# plt.tight_layout()
# plt.show()
#
#




# plt.plot(a)
# plt.show()
# print(model.get_weights())



#
# #comparing the metrics of the test sets
noisy_test_t = np.transpose(noisy_test)
clean_test_t = np.transpose(clean_test)
#
reshaped_data_noisy_test = noisy_test_t.reshape(np.shape(noisy_test)[0], np.shape(noisy_test)[1], 1)
print(reshaped_data_noisy_test)
# reshaped_data_clean_test = clean_test_t.reshape(np.shape(noisy_test)[0], np.shape(noisy_test)[1], 1)
# prediction_clean = model.predict(reshaped_data_noisy_test)
#
#
#file_path = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\predictionOfTestSet.npy'
# # Save the matrix to a NumPy binary file
# np.save(file_path, prediction_clean)
prediction_clean = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\predictionOfTestSetComplexModel.npy")
print(prediction_clean.shape)
print(prediction_clean)
# print(prediction_clean[1000, : ,0])
# plt.plot(prediction_clean[1000, : ,0])
# plt.show()
#prediction_clean_resized = np.reshape(prediction_clean.shape[0], prediction_clean.shape[1])
# prediction_clean = model.predict(reshaped_data_noisy_test)
#reshaped_array_alternative = prediction_clean.squeeze(axis=-1)
# file_path = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\predictionOfTestSetComplexModel.npy'
# # Save the matrix to a NumPy binary file
# np.save(file_path, prediction_clean)
#print(reshaped_array_alternative.shape)
#metrics to compare cleaned signal with the encoder

# filteredSignal = metrics.filtering_signals(noisy_test, 250, 45, 0.5, 50, 4)
# snr = metrics.snr(prediction_clean[0,:], clean_test[0,:])
#
# print(snr)
#
# #
# sampling_freq = 250
# y_axis = np.linspace(0, 2 * sampling_freq)
# for i in range(10,15):
#     fig, axes = plt.pyplot.subplots(nrows=4, ncols=1)
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
#     axes[2].plot(prediction_clean[row_index, :], label = 'clean Data_ predicted')
#     axes[2].set_title('cleaned data')
#     axes[2].set_ylabel('Signal amplitude')
#     axes[2].set_xlabel('Time')
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
#     plt.pyplot.tight_layout()
#
#     # Show the plot
#     plt.pyplot.show()
#

# #
#
# print("********************")
# print(clean_data.shape)
# print(num_chanClean)
# print(num_SamplesClean)
#
#
# # check dimensions of clean and noisy datafiles
# if num_chanClean != num_chanNoisy:
#     exit()
#
# if num_SamplesClean != num_SamplesNoisy:
#     exit()
#
# pmaxn = np.max(noisy_dataF3)
# pminn = np.min(noisy_dataF3)
#
# #pmaxn = np.max(clean_data)
# #pminn = np.min(clean_data)
#
# #estimate value range for noisy and clean data
# pure_probe = np.zeros((1, sampling_freq))
# noisy_probe = np.zeros((1, sampling_freq))
# nOffset = sampling_freq/2
#
# for s in range(0, sampling_freq-1):
#     pure_probe[0, s] = clean_data[0, s + 120]
#
# for s in range(0, sampling_freq-1):
#     noisy_probe[0, s] = noisy_dataF3[0, s + 120]
#
# minc = 3 * np.min(pure_probe)
# maxc = 3 * np.max(pure_probe)
#
# minn = 3 * np.min(noisy_probe)
# maxn = 3 * np.max(noisy_probe)
#
# #determine number of epochs with data
# ns = 128 * 3
# #ns = 128
# # for small dataset ns = 128
# #ns=64
# nb = 30
#
# nDataEpochs = ns * nb * num_chanClean
# #y_val_noisy = data_noisy[:,1]
# y_val_noisy = np.zeros((nDataEpochs,  2 * sampling_freq))
# noisy_sample = np.zeros((2 * sampling_freq, 1))
# #y_val_pure = data_pure[:,1]
# y_val_pure = np.zeros((nDataEpochs, 2 * sampling_freq))
# pure_sample = np.zeros((2 * sampling_freq, 1))
# # Reshape data
# y_val_noisy_r = []
# y_val_pure_r = []
# # for i in range(0, len(y_val_noisy)):
# index = 0;
# nSeizureStartOffset = 60 * sampling_freq
# stddnoisy = 1.e-25
# stddpure = 1.e-25
# pmaxn = 1.e-25
# pmax = 1.e-25
# pminn= -100000000000
# pmin = -100000000000
#
# #Number of times to go over each row to complete the row of 11520000 columns
# n_s = np.int_(11520000/500)
#
#
# print(sampling_freq)
#
# val_data = np.zeros((23040, 0))
#
# for i in range(0, n_s):
#     for row in range(0, 180):
#
#         #rand_index = np.random.randint(0, 11520000 - 500)
#         val_data[i] = clean_data[row , n_s * 2 * sampling_freq : n_s * 2 * sampling_freq + 500]
#
# val_data = np.array(val_data)
# print(val_data.shape)
# #print(signals)
# #plt.plot(signals[1:5])
# #plt.show()
# #raw.compute_psd(fmax=50).plot(picks="data", exclude="bads")
# #raw.plot(duration=5, n_channels=30)
# #plt.show()