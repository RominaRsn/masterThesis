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
import tensorflow as tf
import pickle
import masterThesis.metrics as metrics
import neurokit2 as nk
from keras.callbacks import ModelCheckpoint
from scipy.ndimage import convolve1d
from scipy.signal import butter,filtfilt,iirnotch, convolve, cheby2, sosfiltfilt


#model = model.simpleModel_modified2()
#model.load_weights(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\my_model_modified_simple.h5")



# data_clean_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_new.npy")
# data_noisy_normalized = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_new.npy")


data_clean_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\clean_normalized_cheby_filtered_new.npy")
data_noisy_normalized_cheby = np.load(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\data_file\noisy_normalized_cheby_filtered_new.npy")



# Step 1: Split into training and test sets
noisy_train, noisy_test, clean_train, clean_test = train_test_split(data_noisy_normalized_cheby, data_clean_normalized_cheby, test_size=0.2, random_state=42)

# Step 2: Split the training set into training and validation sets
#noisy_train, noisy_val, clean_train, clean_val = train_test_split(noisy_train, clean_train, test_size=0.1, random_state=42)

num_zeros = (0, 12)

# Pad the array with zeros
padded_noisy_train = np.pad(noisy_train, ((0, 0), num_zeros), mode='constant')
padded_clean_train = np.pad(clean_train, ((0, 0), num_zeros), mode='constant')

padded_noisy_test = np.pad(noisy_test, ((0, 0), num_zeros), mode='constant')
padded_clean_test = np.pad(clean_test, ((0, 0), num_zeros), mode='constant')

# #training with all the skip connections, more epochs
# model_1 = load_model(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\ae_skip_layers_checkpoint.h5')
# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1)
#
# checkpoint_path = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\ae_skip_layers_checkpoint.h5'
#
# checkpoint = ModelCheckpoint(checkpoint_path,
#                              monitor='val_loss',  # You can choose a different metric, e.g., 'val_accuracy'
#                              save_best_only=True,  # Save only if the validation performance improves
#                              mode='min',  # 'min' for loss, 'max' for accuracy, 'auto' will infer automatically
#                              verbose=1)
#
# model_1.optimizer.learning_rate = 1e-6
# model_1.fit(
#     padded_noisy_train,
#     padded_clean_train,
#     epochs=5,
#     batch_size=32,
#     validation_split=0.1,
#     callbacks=[callback, checkpoint],
#     shuffle=True
#
# )
# model_1.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\ae_skip_layers.h5")
#
#
# #training with only the input as skip connections
#
# model_2 = model.encoder_with_5_layers_skip_only_input()
# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1)
#
# checkpoint_path = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\ae_skip_layers_checkpoint_oisk.h5'
#
# checkpoint = ModelCheckpoint(checkpoint_path,
#                              monitor='val_loss',  # You can choose a different metric, e.g., 'val_accuracy'
#                              save_best_only=True,  # Save only if the validation performance improves
#                              mode='min',  # 'min' for loss, 'max' for accuracy, 'auto' will infer automatically
#                              verbose=1)
#
# model_2.optimizer.learning_rate = 1e-6
# model_2.fit(
#     padded_noisy_train,
#     padded_clean_train,
#     epochs=5,
#     batch_size=32,
#     validation_split=0.1,
#     callbacks=[callback, checkpoint],
#     shuffle=True
#
# )
# model_2.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\ae_skip_layers_oisk.h5")
#
# #training with 2 skip connections
#
# model_3 = model.encoder_with_5_layers_2_skip()
# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1)
#
# checkpoint_path = r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\ae_skip_layers_checkpoint_2s.h5'
#
# checkpoint = ModelCheckpoint(checkpoint_path,
#                              monitor='val_loss',  # You can choose a different metric, e.g., 'val_accuracy'
#                              save_best_only=True,  # Save only if the validation performance improves
#                              mode='min',  # 'min' for loss, 'max' for accuracy, 'auto' will infer automatically
#                              verbose=1)
#
# model_3.optimizer.learning_rate = 1e-6
# model_3.fit(
#     padded_noisy_train,
#     padded_clean_train,
#     epochs=5,
#     batch_size=32,
#     validation_split=0.1,
#     callbacks=[callback, checkpoint],
#     shuffle=True
#
# )
# model_3.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\ae_skip_layers_2s.h5")
#
#
# #predicting the test set
# results_skip_layers = model_1.predict(padded_noisy_test)
# np.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\results\ae_skip_layers.npy", results_skip_layers)
#
# results_skip_layers_oisk = model_2.predict(padded_noisy_test)
# np.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\results\ae_skip_layers_oisk.npy", results_skip_layers_oisk)
#
# results_skip_layers_2s = model_3.predict(padded_noisy_test)
# np.save(r"C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\results\ae_skip_layers_2s.npy", results_skip_layers_2s)


model_1 = load_model(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\ae_skip_layers_checkpoint.h5')
model_2 = load_model(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\ae_skip_layers_checkpoint_oisk.h5')
model_3 = load_model(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\ae_skip_layers_checkpoint_2s.h5')
model_4 = load_model(r'C:\Users\RominaRsn\PycharmProjects\MyMasterThesis\masterThesis\trained_models\ae_cheby_checkpoint.h5')


sharpening_kernel = np.array([0, -1, 2, -1, 0])
def sharpenSignal(data):
    data = np.array(data)
    data_sharpened = np.empty_like(data)
    for i in range(0, len(data)):
        data_sharpened[i, :] = convolve1d(data[i, :], weights=sharpening_kernel, mode='constant', cval=0.0)
    return data_sharpened

def cheby2_lowpass(data):
    order = 10
    stopband_attenuation = 40  # in dB
    f1 = 0.1  # Lower stopband frequency in Hz
    f2 = 48.0  # Upper stopband frequency in Hz
    fs = 512  # Sampling frequency in Hz
    sos = cheby2(N=order, rs=stopband_attenuation, Wn=[f1/ (0.5 * fs), f2/ (0.5 * fs)], btype='band', analog=False, output='sos')
    filtered_data = np.empty_like(data)
    for i in range(data.shape[0]):
        filtered_data[i, :] = sosfiltfilt(sos, data[i, :])
        print(f"filtering {i}th row of data")


    return filtered_data


result_1 = model_1.predict(padded_noisy_test[0:1000])
result_1 = result_1.squeeze(-1)
result_2 = model_2.predict(padded_noisy_test[0:1000])
result_2 = result_2.squeeze(-1)
result_3 = model_3.predict(padded_noisy_test[0:1000])
result_3 = result_3.squeeze(-1)
result_4 = model_4.predict(noisy_test[0:1000])
result_4 = result_4.squeeze(-1)


sharpedened_result_1 = sharpenSignal(result_1)
sharpedened_result_2 = sharpenSignal(result_2)
sharpedened_result_3 = sharpenSignal(result_3)

result_1 = cheby2_lowpass(result_1)
result_2 = cheby2_lowpass(result_2)
result_3 = cheby2_lowpass(result_3)
#result_4 = cheby2_lowpass(result_4)


#sharpedened_result_4 = sharpenSignal(result_4)

# sharpedened_result_1_filtered = cheby2_lowpass(sharpedened_result_1)
# sharpedened_result_2_filtered = cheby2_lowpass(sharpedened_result_2)
# sharpedened_result_3_filtered = cheby2_lowpass(sharpedened_result_3)
# #sharpedened_result_4_filtered = cheby2_lowpass(sharpedened_result_4)










#plotting the results
signalIndexVector = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
#signalIndexVector = [0, 1, 3, 4]

for i in signalIndexVector:
    fig, axes = plt.subplots(nrows=6, ncols=1, sharey='col')

    row_index = i
    #row_index = np.random.randint(0, a)
    #col_index = np.random.randint(0, 11520000/500)

    axes[0].plot(padded_clean_test[row_index, :], label = 'Clean Data')
    axes[0].set_title('Clean data')
    axes[0].set_ylabel('Signal amplitude')
    axes[0].set_xlabel('Time')

    #print(smaller_reshaped_data_clean_test[row_index, :].shape)


    axes[1].plot(padded_noisy_test[row_index, :], label = 'Noisy Data')
    axes[1].set_title('Noisy data')
    axes[1].set_ylabel('Signal amplitude')
    axes[1].set_xlabel('Time')

    #result = model.predict(result)
    #result = result.transpose()

    axes[2].plot(result_1[row_index, :], label='predicted data- all skip connections')
    axes[2].plot(sharpedened_result_1[row_index, :], label='sharpened')
    axes[2].set_title('predicted data - all skip connections')
    axes[2].set_ylabel('Signal amplitude')
    axes[2].set_xlabel('Time')
    axes[2].legend(loc='lower right')

    #
    axes[3].plot(result_2[row_index, :], label ='predicted data- only input skip connection')
    axes[3].plot(sharpedened_result_2[row_index, :], label='sharpened')
    axes[3].set_title('predicted data - only input skip connection')
    axes[3].set_ylabel('Signal amplitude')
    axes[3].set_xlabel('Time')
    axes[3].legend(loc='lower right')

    axes[4].plot(result_3[row_index, :], label ='predicted data- 2 skip connection')
    axes[4].plot(sharpedened_result_3[row_index, :], label='sharpened')
    axes[4].set_title('predicted data - 2 skip connection')
    axes[4].set_ylabel('Signal amplitude')
    axes[4].set_xlabel('Time')
    axes[4].legend(loc='lower right')

    axes[5].plot(result_4[row_index, :], label='ae normal')
    axes[5].set_title('ae normal')
    axes[5].set_ylabel('Signal amplitude')
    axes[5].set_xlabel('Time')





    #test_array = np.array(noisy_dataF3[row_index, col_index : col_index + 500])
    #print(test_array.shape())

    # Add overall title
    fig.suptitle('Comparison of clean and noisy data')

    # Adjust layout to prevent overlap
    #plt.tight_layout()

    # Show the plot
    plt.show()

